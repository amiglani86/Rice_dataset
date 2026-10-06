"""
finetune_cross_variety.py
-------------------------
Domain-adaptation demonstration: fine-tune the trained BPT5204 ResNet18 on a
labelled subset of the new multi-variety captures (whatever is in ROOT), and
evaluate on a held-out split of the same captures.

Methodology (reviewer-proof):
  - stratified grain-level split 70 / 15 / 15  (train / val / test)
  - fine-tune layer4 + head (same recipe as the headline model)
  - model selection on VALIDATION macro-F1; single final evaluation on TEST
  - reports zero-shot vs fine-tuned on the identical held-out test

All outputs are written to  Outputs/finetune_crossvariety_seed<SEED>/ :
    finetuned_model.pth, report.txt, summary.txt,
    predictions_test.csv, split.csv, train_log.csv,
    per_class_zeroshot.csv, per_class_finetuned.csv,
    confusion_zeroshot.csv, confusion_finetuned.csv

Run from the project root:
    python "finetune_cross_variety.py"

Dependencies: torch, torchvision, scikit-learn, pandas, pillow, numpy
"""

import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import rice_common as rc

# -----------------------------------------------------------------------------
ROOT     = Path("Data/Test on another dataset/Mixture_Segmented")   # PR14 removed manually
CKPT     = Path("Checkpoints/models/resnet18_6class_headline.pth")

SEED      = 999
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
EPOCHS    = 15
LR        = 1e-4
WEIGHT_DECAY = 1e-4
BATCH     = 16

OUT_DIR  = Path(f"Outputs/finetune_crossvariety_seed{SEED}")
OUT_MODEL = OUT_DIR / "finetuned_model.pth"

ABBR = {"HY": "Healthy", "PC": "Partial chalky", "FC": "Full chalky",
        "DC": "Discolored", "BK": "Broken", "BR": "Broken"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

device  = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = rc.CLASSES_6


# -----------------------------------------------------------------------------
def _md5(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def gather():
    """Each record gets a leakage-safe group id:
       same grain stem (variety+gid) OR identical image content -> same group,
       so no physical grain / duplicate frame can span splits."""
    rows, skipped = [], {}
    hash2group = {}
    dups = 0
    for p in sorted(ROOT.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(ROOT).parts
        if len(rel) < 2:
            continue
        cls = ABBR.get(rel[-2].upper())
        if cls is None:
            skipped[rel[-2]] = skipped.get(rel[-2], 0) + 1
            continue
        variety, gid = rel[0], p.stem
        base_group = f"{variety}/{gid}"          # same stem -> same group
        h = _md5(p)
        if h in hash2group:                       # identical content -> merge groups
            group = hash2group[h]; dups += 1
        else:
            group = base_group
            hash2group[h] = group
        rows.append({"path": str(p), "label": cls, "gid": gid,
                     "variety": variety, "group": group})
    if skipped:
        print("WARNING: skipped unmapped class folders:", skipped)
    if dups:
        print(f"NOTE: {dups} exact-duplicate image(s) detected -> merged into their group (kept, not split apart).")
    return rows


def ds_rows(recs):
    return [(r["path"], r["label"], r["gid"]) for r in recs]


def make_loader(recs, tf, shuffle):
    GrainDataset = rc.make_dataset_class()
    return DataLoader(GrainDataset(ds_rows(recs), CLASSES, tf),
                      batch_size=BATCH, shuffle=shuffle, num_workers=0,
                      worker_init_fn=rc.seed_worker,
                      generator=rc.torch_generator() if shuffle else None,
                      pin_memory=(device == "cuda"))


def load_base():
    model = rc.build_model("resnet18", len(CLASSES)).to(device)
    ck = torch.load(CKPT, map_location=device)
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_all(model, recs):
    model.eval()
    _, eval_tf = rc.make_transforms()
    yt, yp = [], []
    for x, y, _ in make_loader(recs, eval_tf, shuffle=False):
        yp += model(x.to(device)).argmax(1).cpu().tolist()
        yt += y.tolist()
    return [CLASSES[i] for i in yt], [CLASSES[i] for i in yp]


def save_eval(recs, yt, yp, tag):
    labels = [c for c in CLASSES if c in set(yt)]
    rep_txt = classification_report(yt, yp, labels=labels, zero_division=0, digits=3)
    rep_d = classification_report(yt, yp, labels=labels, zero_division=0, output_dict=True)
    acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
    macro = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    pd.DataFrame(rep_d).T.to_csv(OUT_DIR / f"per_class_{tag}.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUT_DIR / f"confusion_{tag}.csv")
    block = f"==== {tag.upper()}  (acc={acc:.3f}, macro-F1={macro:.3f}) ====\n{rep_txt}\n" \
            f"confusion rows=true cols=pred {labels}\n{np.array2string(cm)}\n"
    print(block)
    return block, acc, macro


def main():
    assert ROOT.exists(), f"ROOT not found: {ROOT.resolve()}"
    assert CKPT.exists(), f"Checkpoint not found: {CKPT.resolve()}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc.set_seed(SEED)

    recs = gather()
    labels_all = [r["label"] for r in recs]
    dist = dict(zip(*np.unique(labels_all, return_counts=True)))
    print(f"{len(recs)} grains | class distribution: {dist}")
    tiny = {k: v for k, v in dist.items() if v < 10}
    if tiny:
        print(f"WARNING: very small classes {tiny} -- treat results as a demonstration, report seeds.")

    # ---- leakage-safe GROUP-level stratified split (70/15/15) ----
    # one row per group; a group's label = its records' label (warn if mixed)
    groups, glabel = {}, {}
    for r in recs:
        groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        labs = {r["label"] for r in rs}
        if len(labs) > 1:
            print(f"WARNING: group {g} has mixed labels {labs} -- check your folders.")
        glabel[g] = rs[0]["label"]
    gkeys = sorted(groups)
    gy = [glabel[g] for g in gkeys]
    print(f"{len(gkeys)} groups from {len(recs)} images "
          f"({len(recs) - len(gkeys)} images share a group)")

    try:
        g_trval, g_test = train_test_split(gkeys, test_size=TEST_FRAC,
                                           random_state=SEED, stratify=gy)
        g_tr, g_val = train_test_split(
            g_trval, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=SEED,
            stratify=[glabel[g] for g in g_trval])
    except ValueError as e:
        print(f"Stratified group split failed ({e}); falling back to random group split.")
        g_trval, g_test = train_test_split(gkeys, test_size=TEST_FRAC, random_state=SEED)
        g_tr, g_val = train_test_split(g_trval, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                       random_state=SEED)

    g_tr, g_val, g_test = set(g_tr), set(g_val), set(g_test)
    tr  = [r for r in recs if r["group"] in g_tr]
    val = [r for r in recs if r["group"] in g_val]
    test = [r for r in recs if r["group"] in g_test]
    print(f"train={len(tr)}  val={len(val)}  test={len(test)}")

    # ---- HARD leakage checks (abort if anything overlaps) ----
    assert g_tr.isdisjoint(g_val) and g_tr.isdisjoint(g_test) and g_val.isdisjoint(g_test), \
        "GROUP leakage: a group appears in more than one split"
    s_tr  = {r["path"] for r in tr}
    s_val = {r["path"] for r in val}
    s_te  = {r["path"] for r in test}
    assert s_tr.isdisjoint(s_val) and s_tr.isdisjoint(s_te) and s_val.isdisjoint(s_te), \
        "FILE leakage: an image appears in more than one split"
    gid_tr = {(r["variety"], r["gid"]) for r in tr} | {(r["variety"], r["gid"]) for r in val}
    gid_te = {(r["variety"], r["gid"]) for r in test}
    assert gid_tr.isdisjoint(gid_te), "GRAIN-ID leakage: a grain id is in both train/val and test"
    print("LEAKAGE CHECK PASSED: splits are disjoint by group, grain-id, and file.")
    # save split membership
    with open(OUT_DIR / "split.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "variety", "true", "split"])
        for name, part in [("train", tr), ("val", val), ("test", test)]:
            for r in part:
                w.writerow([Path(r["path"]).name, r["variety"], r["label"], name])

    # --- zero-shot on test ---
    model = load_base()
    yt0, yp0 = predict_all(model, test)
    block0, acc0, macro0 = save_eval(test, yt0, yp0, "zeroshot")

    # --- fine-tune layer4 + head, select best epoch by val macro-F1 ---
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    _, eval_tf = rc.make_transforms(); train_tf, _ = rc.make_transforms()
    tl = make_loader(tr, train_tf, shuffle=True)

    best_f1, best_state, log = -1.0, None, []
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        sched.step()
        yv, pv = predict_all(model, val)
        vf1 = f1_score(yv, pv, labels=[c for c in CLASSES if c in set(yv)],
                       average="macro", zero_division=0)
        trloss = tot / max(len(tr), 1)
        log.append((ep + 1, trloss, vf1))
        print(f"epoch {ep+1:2d}/{EPOCHS}  train_loss={trloss:.4f}  val_macroF1={vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(log, columns=["epoch", "train_loss", "val_macroF1"]).to_csv(
        OUT_DIR / "train_log.csv", index=False)

    # --- fine-tuned on the SAME test ---
    ytf, ypf = predict_all(model, test)
    blockf, accf, macrof = save_eval(test, ytf, ypf, "finetuned")

    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "seed": SEED}, OUT_MODEL)

    with open(OUT_DIR / "predictions_test.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "variety", "true", "zeroshot_pred", "finetuned_pred"])
        for r, p0, pf in zip(test, yp0, ypf):
            w.writerow([Path(r["path"]).name, r["variety"], r["label"], p0, pf])

    summary = (f"SEED={SEED}  n={len(recs)}  train/val/test={len(tr)}/{len(val)}/{len(test)}\n"
               f"zero-shot : acc={acc0:.3f}  macro-F1={macro0:.3f}\n"
               f"fine-tuned: acc={accf:.3f}  macro-F1={macrof:.3f}\n"
               f"best val macro-F1 = {best_f1:.3f}\n")
    (OUT_DIR / "report.txt").write_text(block0 + "\n" + blockf, encoding="utf-8")
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"all outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()