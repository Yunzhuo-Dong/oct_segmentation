#!/usr/bin/env python
"""Step 2: build the dataset directly from raw MATLAB annotations + TIFFs.

Produces in --out:
  samples/*.npz          image (uint8, canonical HxW) + runs [b, x, y_center, len]
  manifest.csv           one row per sample incl. animal_id, scan_key, split
  skipped.csv            every .mat not used and why
  duplicates.csv         identical images referenced by several .mat files
  unknown_names.csv      label names that did not map to the 11 boundaries
  boundary_inventory.csv per boundary: images, column coverage, multi-curve share
  order_table.csv        pairwise above/below/touch counts (TRAIN split only)
  order_constraints.json data-derived chain (top->bottom in array y),
                         orientation, constraint pairs used by E1 / E0-PP
  summary.json           counts, split mode, animal sets, overlap checks

python 02_prepare.py --mat-root "$MAT_BASE" --out "$WS/octb_data_v1" --split-mode animal
"""
import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import (BIDX, BOUNDARIES, DATE_RE, DEFAULT_ANIMAL_RE, parse_ids, read_csv,  # noqa: E402
                         save_json, set_seed, write_csv, write_test)
from octb.matio import basename_any, index_images, load_annotation, read_image_gray, resize_gray  # noqa: E402
from octb.targets import points_to_runs, project_points  # noqa: E402

_CTX = {}


def _init(ctx):
    _CTX.update(ctx)


def _pick_image(mat_path, src_file):
    by_name, by_stem = _CTX["by_name"], _CTX["by_stem"]
    cands = []
    if src_file:
        bn = basename_any(src_file).lower()
        cands = by_name.get(bn, []) or by_stem.get(os.path.splitext(bn)[0], [])
    if not cands:
        stem = os.path.splitext(os.path.basename(mat_path))[0].lower()
        for suf in ("_gtruth", "_groundtruth", "_label", "_labels", "_gt"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        cands = by_stem.get(stem, [])
    if len(cands) > 1:
        md = os.path.dirname(mat_path)
        same = [c for c in cands if os.path.dirname(c) == md]
        if len(same) == 1:
            return same, None
        return cands, "ambiguous"
    return cands, None


def load_one(mat_path):
    r = dict(mat_path=mat_path, status="ok", reason="")
    try:
        a = load_annotation(mat_path)
    except ValueError as e:
        r.update(status="skip", reason=str(e))
        return r
    r["source_file"] = a["source_file"]
    r["annotator"] = a["annotator"]
    cands, amb = _pick_image(mat_path, a["source_file"])
    if not cands:
        r.update(status="skip", reason="image_not_found")
        return r
    img, info, note = None, {}, ""
    if amb:
        loaded = {}
        for c in sorted(cands):
            try:
                loaded[c] = read_image_gray(c)
            except Exception:  # noqa: BLE001
                continue
        lm_shapes = {np.squeeze(ld["mask"]).shape for ld in a["labels"]
                     if ld["mask"] is not None and np.squeeze(ld["mask"]).ndim == 2}
        ok = [c for c, (im, _) in loaded.items() if not lm_shapes or im.shape in lm_shapes]
        if not ok:
            r.update(status="skip", reason=f"ambiguous_image:{len(cands)}_candidates_no_shape_match")
            return r
        md = os.path.dirname(mat_path)

        def closeness(c):
            return len(os.path.commonpath([md, os.path.dirname(c)]))

        ok.sort(key=closeness, reverse=True)
        same_content = len({hashlib.md5(loaded[c][0].tobytes()).hexdigest() for c in ok}) == 1
        if same_content:
            note = f"identical_copies:{len(ok)}"
        elif len(ok) > 1 and closeness(ok[0]) == closeness(ok[1]):
            r.update(status="skip", reason=f"ambiguous_image:{len(ok)}_different_images_equally_close")
            return r
        else:
            note = f"closest_of_{len(ok)}_different_images"
        r["image_path"] = ok[0]
        img, info = loaded[ok[0]]
    else:
        r["image_path"] = cands[0]
        try:
            img, info = read_image_gray(cands[0])
        except Exception as e:  # noqa: BLE001
            r.update(status="skip", reason=f"image_read_failed:{type(e).__name__}")
            return r
    r["image_note"] = note
    r["image"] = img
    r["image_info"] = info
    pts = defaultdict(list)
    unknown, bad = [], []
    for ld in a["labels"]:
        m = ld["mask"]
        if ld["name"] is None:
            unknown.append(ld["raw_name"])
            continue
        if m is not None and m.ndim != 2:
            m = np.squeeze(m)
        if m is None or m.ndim != 2:
            bad.append(f"{ld['raw_name']}:shape{None if m is None else tuple(m.shape)}")
            continue
        if m.shape != img.shape:
            bad.append(f"{ld['raw_name']}:shape{m.shape}!=img{img.shape}")
            continue
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        pts[ld["name"]].append((ys, xs))
    r["unknown"] = unknown
    r["bad_labels"] = bad
    r["points"] = {k: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v]))
                   for k, v in pts.items()}
    if not r["points"]:
        r.update(status="skip", reason="no_valid_boundary_mask" + (f"({';'.join(bad)})" if bad else ""))
    return r


def assign_split_animal(groups, fracs, seed):
    """Greedy group assignment toward target image fractions."""
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # big groups first, random tie-break order kept
    names = list(fracs)
    total = sum(len(v) for v in groups.values())
    count = {n: 0 for n in names}
    out = {}
    for k in keys:
        deficit = {n: fracs[n] * total - count[n] for n in names}
        n = max(names, key=lambda s: deficit[s])
        out[k] = n
        count[n] += len(groups[k])
    return out


def order_statistics(train_runs, C, tol=1.0):
    """Pairwise stats over columns where both boundaries have exactly one run."""
    above = np.zeros((C, C), np.int64)   # y_a < y_b - tol
    touch = np.zeros((C, C), np.int64)   # |y_a - y_b| <= tol
    ysum = np.zeros(C)
    ycnt = np.zeros(C)
    for runs, W in train_runs:
        if len(runs) == 0:
            continue
        b = runs[:, 0].astype(int)
        x = runs[:, 1].astype(int)
        n = np.zeros((C, W), int)
        np.add.at(n, (b, x), 1)
        Y = np.full((C, W), np.nan)
        s = n[b, x] == 1
        Y[b[s], x[s]] = runs[s, 2]
        for c in range(C):
            f = np.isfinite(Y[c])
            ysum[c] += Y[c][f].sum()
            ycnt[c] += f.sum()
        for i in range(C):
            fi = np.isfinite(Y[i])
            if not fi.any():
                continue
            for j in range(C):
                if i == j:
                    continue
                m = fi & np.isfinite(Y[j])
                if not m.any():
                    continue
                d = Y[i][m] - Y[j][m]
                above[i, j] += int((d < -tol).sum())
                touch[i, j] += int((np.abs(d) <= tol).sum())
    mean_y = np.where(ycnt > 0, ysum / np.maximum(ycnt, 1), np.nan)
    return above, touch, mean_y, ycnt


def derive_chain(above, mean_y, present):
    """Order boundaries top->bottom: pairwise win fraction, mean y as tiebreak."""
    C = len(BOUNDARIES)
    score = np.zeros(C)
    for i in range(C):
        ws = []
        for j in range(C):
            if i != j and (above[i, j] + above[j, i]) > 0:
                ws.append(above[i, j] / (above[i, j] + above[j, i]))
        score[i] = np.mean(ws) if ws else np.nan
    idx = [c for c in range(C) if present[c]]
    idx.sort(key=lambda c: (-(score[c] if np.isfinite(score[c]) else 0.5), mean_y[c]))
    return idx, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-root", required=True)
    ap.add_argument("--image-roots", nargs="+", default=None, help="where TIFFs live (default: mat-root)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-mode", choices=["animal", "source"], default="animal",
                    help="animal: new animal-disjoint split of everything; source: MATLAB 'testing' folder = test")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--test-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gap-px", type=float, default=3.0, help="vertical gap that separates two runs")
    ap.add_argument("--canonical-hw", default="auto", help="'auto' = most common native size, or 'H,W'")
    ap.add_argument("--animal-regex", default=DEFAULT_ANIMAL_RE)
    ap.add_argument("--animal-map", default=None, help="csv with columns image_basename,animal_id (overrides regex)")
    ap.add_argument("--order-min-support", type=int, default=200)
    ap.add_argument("--order-max-viol", type=float, default=0.01)
    ap.add_argument("--multi-curve-thr", type=float, default=0.05,
                    help="boundaries with more multi-run columns than this are excluded from order constraints")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.out
    if os.path.exists(os.path.join(out, "manifest.csv")) and not args.force:
        sys.exit(f"{out}/manifest.csv exists. Use a new --out (never overwrite a frozen split) or --force.")
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    write_test(out)

    mats = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.mat_root)
                  for f in fs if f.lower().endswith(".mat"))
    if args.limit:
        mats = mats[: args.limit]
    roots = args.image_roots or [args.mat_root]
    by_name, by_stem = index_images(roots)
    print(f"[prepare] {len(mats)} .mat, {sum(map(len, by_name.values()))} images indexed")

    ctx = dict(by_name=by_name, by_stem=by_stem)
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(ctx,)) as pool:
            results = list(pool.imap(load_one, mats, chunksize=4))
    else:
        _init(ctx)
        results = [load_one(m) for m in mats]

    skipped = [dict(mat_path=r["mat_path"], reason=r["reason"], source_file=r.get("source_file", ""))
               for r in results if r["status"] != "ok"]
    ok = [r for r in results if r["status"] == "ok"]
    unknown = Counter(n for r in ok for n in r["unknown"])
    print(f"[prepare] loaded {len(ok)}, skipped {len(skipped)}: {Counter(s['reason'].split(':')[0] for s in skipped)}")
    if not ok:
        write_csv(skipped, os.path.join(out, "skipped.csv"))
        sys.exit("no usable samples")

    # canonical size
    shapes = Counter(r["image"].shape for r in ok)
    if args.canonical_hw == "auto":
        H, W = shapes.most_common(1)[0][0]
    else:
        H, W = map(int, args.canonical_hw.split(","))
    print(f"[prepare] native shapes {dict(shapes)} -> canonical {(H, W)}")

    amap = {}
    if args.animal_map:
        for row in read_csv(args.animal_map):
            amap[row["image_basename"].lower()] = row["animal_id"].strip().upper()

    # duplicates by image content
    hashes = defaultdict(list)
    for r in ok:
        hashes[hashlib.md5(r["image"].tobytes()).hexdigest()].append(r)
    dups = []
    for h, rs in hashes.items():
        if len(rs) > 1:
            for r in rs:
                dups.append(dict(hash=h, mat_path=r["mat_path"], image_path=r["image_path"], n=len(rs)))

    rows, runs_by_id, seen_exact = [], {}, {}
    inv = {n: dict(images=0, cols=0, multi_cols=0, px=0) for n in BOUNDARIES}
    for k, r in enumerate(sorted(ok, key=lambda r: r["mat_path"])):
        img = r["image"]
        src_hw = img.shape
        img_c = resize_gray(img, (H, W))
        all_runs = []
        for name, (ys, xs) in r["points"].items():
            yp, xp, sy = project_points(ys, xs, src_hw, (H, W))
            ru = points_to_runs(BIDX[name], yp, xp, gap_px=args.gap_px)
            all_runs.append(ru)
            n_per_col = np.bincount(ru[:, 1].astype(int), minlength=W)
            inv[name]["images"] += 1
            inv[name]["cols"] += int((n_per_col > 0).sum())
            inv[name]["multi_cols"] += int((n_per_col > 1).sum())
            inv[name]["px"] += int(ys.size)
        runs = np.concatenate(all_runs) if all_runs else np.zeros((0, 4), np.float32)
        dkey = hashlib.md5(img_c.tobytes() + np.ascontiguousarray(runs).tobytes()).hexdigest()
        if dkey in seen_exact:
            skipped.append(dict(mat_path=r["mat_path"], reason=f"exact_duplicate_of:{seen_exact[dkey]}",
                                source_file=r.get("source_file", "")))
            r.pop("image", None)
            continue
        seen_exact[dkey] = r["mat_path"]
        img_base = os.path.basename(r["image_path"])
        animal, eye, date = parse_ids(img_base, args.animal_regex)
        if not animal:
            animal, eye2, date2 = parse_ids(r["source_file"] or r["mat_path"], args.animal_regex)
            eye, date = eye or eye2, date or date2
        animal = amap.get(img_base.lower(), animal)
        rel = os.path.relpath(r["mat_path"], args.mat_root)
        if not date:
            dm = DATE_RE.search(rel)
            date = dm.group(1) if dm else ""
        stem = os.path.splitext(img_base)[0]
        parsed = bool(animal)
        if not animal:
            animal = f"UNPARSED:{stem}"
        src_split = "testing" if "testing" in [p.lower() for p in rel.split(os.sep)] else (
            "training" if "training" in [p.lower() for p in rel.split(os.sep)] else "other")
        parts = rel.split(os.sep)
        top_is_split = parts[0].lower() in ("training", "testing")
        if top_is_split:
            experiment = parts[1] if len(parts) > 2 else ""
        else:
            experiment = parts[0] if len(parts) > 1 else ""
        sid = f"{k:04d}_{stem}"[:120]
        npz_rel = os.path.join("samples", sid + ".npz")
        np.savez_compressed(os.path.join(out, npz_rel), image=img_c, runs=runs)
        runs_by_id[sid] = runs
        rows.append(dict(
            sample_id=sid, npz=npz_rel, image_path=r["image_path"], mat_path=r["mat_path"],
            source_split=src_split, experiment=experiment, animal_id=animal, animal_parsed=int(parsed), eye=eye, date=date,
            scan_key=f"{date}_{animal}_{eye}", annotator=r.get("annotator", ""),
            native_h=src_hw[0], native_w=src_hw[1], n_boundaries=len(r["points"]),
            boundaries="|".join(sorted(r["points"])), bad_labels=";".join(r["bad_labels"]),
            image_dtype=r["image_info"].get("orig_dtype", ""), image_note=r.get("image_note", ""), split=""))
        r.pop("image", None)

    # ---------------- split ----------------
    groups = defaultdict(list)
    for row in rows:
        groups[row["animal_id"]].append(row)
    if args.split_mode == "animal":
        fr = {"train": 1 - args.val_frac - args.test_frac, "val": args.val_frac, "test": args.test_frac}
        assign = assign_split_animal(groups, fr, args.seed)
        for row in rows:
            row["split"] = assign[row["animal_id"]]
    else:
        test_animals = {row["animal_id"] for row in rows if row["source_split"] == "testing"}
        pool_groups = {}
        for a, rs in groups.items():
            if a in test_animals:
                for row in rs:
                    if row["source_split"] == "testing":
                        row["split"] = "test"
                    else:
                        row["split"] = "excluded"
                        skipped.append(dict(mat_path=row["mat_path"], reason="animal_overlap_with_source_test",
                                            source_file=row["image_path"]))
            else:
                pool_groups[a] = rs
        vf = args.val_frac / max(1 - args.test_frac, 1e-6)
        assign = assign_split_animal(pool_groups, {"train": 1 - vf, "val": vf}, args.seed)
        for a, rs in pool_groups.items():
            for row in rs:
                row["split"] = assign[a]
    rows_used = [r for r in rows if r["split"] in ("train", "val", "test")]

    # overlap checks
    sets = {s: {r["animal_id"] for r in rows_used if r["split"] == s} for s in ("train", "val", "test")}
    scans = {s: {r["scan_key"] for r in rows_used if r["split"] == s} for s in ("train", "val", "test")}
    hsplit = defaultdict(set)
    for h, rs in hashes.items():
        for r in rs:
            for row in rows_used:
                if row["mat_path"] == r["mat_path"]:
                    hsplit[h].add(row["split"])
    overlap = dict(
        animal_train_val=sorted(sets["train"] & sets["val"]),
        animal_train_test=sorted(sets["train"] & sets["test"]),
        animal_val_test=sorted(sets["val"] & sets["test"]),
        scan_train_test=sorted(scans["train"] & scans["test"]),
        identical_image_across_splits=sum(1 for v in hsplit.values() if len(v) > 1))

    # ---------------- order statistics (train only) ----------------
    C = len(BOUNDARIES)
    train_runs = [(runs_by_id[r["sample_id"]], W) for r in rows_used if r["split"] == "train"]
    above, touch, mean_y, ycnt = order_statistics(train_runs, C)
    multi_frac = np.array([inv[n]["multi_cols"] / inv[n]["cols"] if inv[n]["cols"] else 0 for n in BOUNDARIES])
    present = ycnt > 0
    chain, score = derive_chain(above, mean_y, present)
    chain_names = [BOUNDARIES[c] for c in chain]
    excluded = [BOUNDARIES[c] for c in range(C) if multi_frac[c] > args.multi_curve_thr]
    order_rows, pairs = [], []
    pos = {c: i for i, c in enumerate(chain)}
    for i in range(C):
        for j in range(C):
            if i >= j or not (present[i] and present[j]):
                continue
            a, b = (i, j) if pos[i] < pos[j] else (j, i)
            sup = above[a, b] + above[b, a] + touch[a, b]
            viol = above[b, a] / sup if sup else float("nan")
            use = (sup >= args.order_min_support and viol <= args.order_max_viol
                   and BOUNDARIES[a] not in excluded and BOUNDARIES[b] not in excluded)
            if use:
                pairs.append([BOUNDARIES[a], BOUNDARIES[b]])
            order_rows.append(dict(upper=BOUNDARIES[a], lower=BOUNDARIES[b], support_cols=int(sup),
                                   upper_above=int(above[a, b]), touch=int(touch[a, b]),
                                   reversed=int(above[b, a]), reversed_rate=viol,
                                   mean_gap_px=float(mean_y[b] - mean_y[a]), used_as_constraint=int(use)))
    orient = "unknown"
    if "RNFL" in chain_names and ("RPE" in chain_names or "CHOROID" in chain_names):
        deep = "RPE" if "RPE" in chain_names else "CHOROID"
        orient = "vitreous_top" if chain_names.index("RNFL") < chain_names.index(deep) else "vitreous_bottom"
    cons = dict(chain=chain_names, orientation=orient, pairs=pairs,
                excluded_from_constraints=excluded,
                multi_curve_fraction={n: float(multi_frac[c]) for c, n in enumerate(BOUNDARIES)},
                mean_y_train={n: float(mean_y[c]) for c, n in enumerate(BOUNDARIES)},
                criteria=dict(min_support=args.order_min_support, max_reversed_rate=args.order_max_viol,
                              multi_curve_thr=args.multi_curve_thr, computed_on="train"))

    # ---------------- write ----------------
    write_csv(rows, os.path.join(out, "manifest.csv"))
    write_csv(skipped, os.path.join(out, "skipped.csv"), ["mat_path", "reason", "source_file"])
    write_csv(dups, os.path.join(out, "duplicates.csv"), ["hash", "mat_path", "image_path", "n"])
    write_csv([dict(raw_name=k, count=v) for k, v in unknown.items()], os.path.join(out, "unknown_names.csv"),
              ["raw_name", "count"])
    write_csv([dict(boundary=n, **inv[n], multi_curve_fraction=float(multi_frac[c]),
                    mean_col_coverage=inv[n]["cols"] / (inv[n]["images"] * W) if inv[n]["images"] else 0)
               for c, n in enumerate(BOUNDARIES)], os.path.join(out, "boundary_inventory.csv"))
    write_csv(order_rows, os.path.join(out, "order_table.csv"))
    save_json(cons, os.path.join(out, "order_constraints.json"))
    cnt = Counter(r["split"] for r in rows)
    summary = dict(
        args=vars(args), canonical_hw=[H, W], native_shapes={f"{k[0]}x{k[1]}": v for k, v in shapes.items()},
        n_mat=len(mats), n_loaded=len(ok), n_skipped=len(skipped), split_counts=dict(cnt),
        animals={s: sorted(v) for s, v in sets.items()},
        n_animals={s: len(v) for s, v in sets.items()},
        unparsed_animal_images=sum(1 for r in rows if not int(r["animal_parsed"])),
        overlap=overlap, orientation=orient, chain=chain_names, n_constraint_pairs=len(pairs),
        unknown_names=dict(unknown), n_duplicate_entries=len(dups),
        experiments={e: dict(Counter(r["split"] for r in rows if r["experiment"] == e))
                     for e in sorted({r["experiment"] for r in rows})})
    save_json(summary, os.path.join(out, "summary.json"))

    print(f"[split] {dict(cnt)}  animals {summary['n_animals']}")
    print(f"[split] overlap check: {overlap}")
    if summary["unparsed_animal_images"]:
        print(f"[WARN] {summary['unparsed_animal_images']} images without parsed animal id "
              f"-> each is its own group; supply --animal-map to avoid leakage")
    print(f"[order] orientation={orient}  chain(top->bottom in array y)={chain_names}")
    print(f"[order] {len(pairs)} constraint pairs; excluded (multi-curve): {excluded}")
    if unknown:
        print(f"[WARN] unknown label names: {dict(unknown)}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
