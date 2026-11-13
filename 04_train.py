#!/usr/bin/env python
"""Step 4: train E0 (CE only) or E1 (CE + order + continuity).

E0:  python 04_train.py --data-dir D --out $WS/octb_runs/E0
E1:  python 04_train.py --data-dir D --out $WS/octb_runs/E1 --lambda-order 0.1 --lambda-smooth 0.01
Everything else (split, augmentation, epochs, seed) must stay identical.
"""
import argparse
import math
import os
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import (BIDX, BOUNDARIES, load_json, read_csv, save_json, set_seed,  # noqa: E402
                         sha256_file, write_csv, write_test)
from octb.dataset import BoundaryDataset  # noqa: E402
from octb.losses import column_ce, decode, order_loss, smooth_loss  # noqa: E402
from octb.metrics import Aggregator, sample_geom, w1_columns  # noqa: E402
from octb.model import build_model  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="ffc_unet")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--ffc-from", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16, help="effective batch size per optimizer step")
    ap.add_argument("--micro-bs", type=int, default=0,
                    help="samples per forward pass (0 = bs). bs/micro-bs steps are accumulated; with GroupNorm\nthis is equivalent to a true batch of bs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=-1, help="-1 = 5%% of all optimizer steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--vshift", type=int, default=40)
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--no-intensity", action="store_true")
    ap.add_argument("--ce-balance", choices=["column", "channel"], default="column")
    ap.add_argument("--lambda-order", type=float, default=0.0)
    ap.add_argument("--lambda-smooth", type=float, default=0.0)
    ap.add_argument("--constraint-start-epoch", type=int, default=2)
    ap.add_argument("--constraint-ramp-epochs", type=int, default=3)
    ap.add_argument("--gate-thr", type=float, default=0.7, help="confidence (local mass) gate")
    ap.add_argument("--gate-win", type=int, default=5, help="half window (px) for local soft-argmax")
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=0, help="smoke tests only")
    ap.add_argument("--max-val-batches", type=int, default=0, help="smoke tests only")
    return ap.parse_args(argv)


def lr_lambda(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    t = (step - warm) / max(total - warm, 1)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def validate(model, loader, device, amp_ctx, pairs, args):
    model.eval()
    agg = Aggregator(pairs)
    ce_sum, n = 0.0, 0
    for k, batch in enumerate(loader):
        if args.max_val_batches and k >= args.max_val_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        tgt = batch["target"].to(device)
        val = batch["valid"].to(device)
        with amp_ctx():
            logits = model(x)
        ce_sum += column_ce(logits, tgt, val).item()
        n += 1
        yhat, conf, p = decode(logits, args.gate_win)
        w1 = w1_columns(p, tgt, val)
        for i in range(x.shape[0]):
            g = sample_geom(yhat[i].cpu().numpy(), conf[i].cpu().numpy(), batch["gt_y"][i].numpy(),
                            batch["single"][i].numpy(), pairs, args.gate_thr)
            agg.add(g, w1[i].sum(-1).cpu().numpy(), val[i].float().sum(-1).cpu().numpy())
    s = agg.summary()
    s["ce"] = ce_sum / max(n, 1)
    return s


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out
    if os.path.exists(os.path.join(out, "best.pt")) and not (args.resume or args.force):
        sys.exit(f"{out}/best.pt exists. Use a new --out, or --resume. Never overwrite a finished baseline.")
    os.makedirs(out, exist_ok=True)
    write_test(out)

    rows = read_csv(os.path.join(args.data_dir, "manifest.csv"))
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    cons = load_json(os.path.join(args.data_dir, "order_constraints.json"))
    all_pairs = [(BIDX[a], BIDX[b]) for a, b in cons["pairs"]]
    smooth_ch = [BIDX[c] for c in cons["chain"] if c not in cons["excluded_from_constraints"]]

    ds_tr = BoundaryDataset(tr, args.data_dir, train=True, sigma=args.sigma, hflip=not args.no_hflip,
                            vshift=args.vshift, intensity=not args.no_intensity)
    ds_va = BoundaryDataset(va, args.data_dir, train=False, sigma=args.sigma)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)

    def _wi(wid):
        np.random.seed((torch.initial_seed() + wid) % 2 ** 32)

    micro = args.micro_bs or args.bs
    if args.bs % micro:
        sys.exit(f"--bs {args.bs} must be a multiple of --micro-bs {micro}")
    if len(tr) < args.bs:
        sys.exit(f"only {len(tr)} training samples < batch size {args.bs}")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, worker_init_fn=_wi, **kw)
    dl_va = DataLoader(ds_va, batch_size=micro, shuffle=False, **kw)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(args.arch, n_out=len(BOUNDARIES), base=args.base, depth=args.depth,
                        ffc_from=args.ffc_from).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    iters_per_epoch = len(dl_tr) if not args.max_train_batches else min(len(dl_tr), args.max_train_batches)
    total = args.epochs * iters_per_epoch
    if args.warmup_iters < 0:
        args.warmup_iters = max(1, int(0.05 * total))
    print(f"[train] effective bs {args.bs} = {args.bs // micro} x micro {micro}; "
          f"{iters_per_epoch} optimizer steps/epoch, {total} total, warmup {args.warmup_iters}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, args.warmup_iters))
    use_amp = args.amp == "bf16" and device.type == "cuda"

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

    start_epoch, best = 0, float("inf")
    hist_path = os.path.join(out, "history.csv")
    history = []
    if args.resume and os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        history = read_csv(hist_path) if os.path.exists(hist_path) else []
        print(f"[resume] from epoch {start_epoch}, best {best:.4f}")

    cfg = dict(args=vars(args), n_params=n_par, n_train=len(tr), n_val=len(va),
               manifest_sha256=sha256_file(os.path.join(args.data_dir, "manifest.csv")),
               order_constraints=cons, constraint_pairs=[[BOUNDARIES[a], BOUNDARIES[b]] for a, b in all_pairs],
               smooth_channels=[BOUNDARIES[c] for c in smooth_ch], device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "", torch=torch.__version__,
               python=platform.python_version(), slurm_job=os.environ.get("SLURM_JOB_ID", ""),
               hostname=platform.node(), boundaries=BOUNDARIES)
    save_json(cfg, os.path.join(out, "config.json"))
    print(f"[train] {len(tr)} train / {len(va)} val, {n_par / 1e6:.1f}M params, device {device} {cfg['gpu']}")
    print(f"[train] lambda_order={args.lambda_order} lambda_smooth={args.lambda_smooth} "
          f"pairs={cfg['constraint_pairs']}")

    def save_ck(path, epoch):
        tmp = path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best=best, args=dict(vars(args), n_out=len(BOUNDARIES)),
                        boundaries=BOUNDARIES, manifest_sha256=cfg["manifest_sha256"]), tmp)
        os.replace(tmp, path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        ramp = 0.0
        if epoch >= args.constraint_start_epoch:
            ramp = min(1.0, (epoch - args.constraint_start_epoch + 1) / max(args.constraint_ramp_epochs, 1))
        sums = dict(loss=0.0, ce=0.0, order=0.0, smooth=0.0, gate=0.0)
        nb = 0
        for k, batch in enumerate(dl_tr):
            if args.max_train_batches and k >= args.max_train_batches:
                break
            v_total = max(float(batch["valid"].float().sum()), 1.0)
            n_acc = args.bs // micro
            opt.zero_grad(set_to_none=True)
            step = dict(loss=0.0, ce=0.0, order=0.0, smooth=0.0, gate=0.0)
            for j in range(0, args.bs, micro):
                x = batch["image"][j:j + micro].to(device, non_blocking=True)
                tgt = batch["target"][j:j + micro].to(device, non_blocking=True)
                val = batch["valid"][j:j + micro].to(device, non_blocking=True)
                with amp_ctx():
                    logits = model(x)
                ce = column_ce(logits, tgt, val, args.ce_balance)
                # weight by this chunk's share of annotated columns -> exact batch-level column mean
                w = (val.float().sum() / v_total) if args.ce_balance == "column" else 1.0 / n_acc
                loss = ce * w
                lo = ls = torch.zeros((), device=device)
                gate_frac = 0.0
                if args.lambda_order > 0 or args.lambda_smooth > 0:
                    yhat, conf, _ = decode(logits, args.gate_win)
                    lo, gate_frac = order_loss(yhat, conf, all_pairs, args.gate_thr)
                    ls = smooth_loss(yhat, conf, smooth_ch, args.gate_thr)
                    loss = loss + ramp * (args.lambda_order * lo + args.lambda_smooth * ls) / n_acc
                loss.backward()
                step["loss"] += loss.item(); step["ce"] += ce.item() / n_acc; step["order"] += lo.item() / n_acc
                step["smooth"] += ls.item() / n_acc; step["gate"] += gate_frac / n_acc
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for kk in sums:
                sums[kk] += step[kk]
            nb += 1
            if k % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
                print(f"  ep{epoch} step{k}/{iters_per_epoch} loss {step['loss']:.4f} ce {step['ce']:.4f} "
                      f"order {step['order']:.4f} smooth {step['smooth']:.4f} gate {step['gate']:.3f} "
                      f"peak_mem {mem:.1f}GB", flush=True)
        tr_stats = {f"train_{k}": v / max(nb, 1) for k, v in sums.items()}
        vs = validate(model, dl_va, device, amp_ctx, all_pairs, args)
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], ramp=ramp, **tr_stats,
                   val_ce=vs["ce"], val_macro_w1=vs["macro_w1"], val_macro_mae=vs["macro_mae"],
                   val_crossing_gt=vs["crossing_rate_gt"], val_crossing_conf=vs["crossing_rate_conf"],
                   val_jump_rate=vs["jump_rate"], time_s=time.time() - t0,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0)
        for name, d in vs["per_boundary"].items():
            row[f"val_w1_{name}"] = d["w1"]
        history.append(row)
        write_csv(history, hist_path)
        save_ck(os.path.join(out, "last.pt"), epoch)
        if np.isfinite(vs["macro_w1"]) and vs["macro_w1"] < best:
            best = vs["macro_w1"]
            save_ck(os.path.join(out, "best.pt"), epoch)
            save_json(dict(epoch=epoch, **vs), os.path.join(out, "best_val_metrics.json"))
        print(f"[ep {epoch}] train_ce {tr_stats['train_ce']:.4f} | val W1 {vs['macro_w1']:.3f}px "
              f"MAE {vs['macro_mae']:.3f}px cross(gt) {vs['crossing_rate_gt']:.4f} "
              f"cross(conf) {vs['crossing_rate_conf']:.4f} | best {best:.3f} | {row['time_s']:.0f}s", flush=True)
    save_json(dict(finished=True, epochs=args.epochs, best_val_macro_w1=best), os.path.join(out, "DONE.json"))
    print("[done]", out)


if __name__ == "__main__":
    main()
