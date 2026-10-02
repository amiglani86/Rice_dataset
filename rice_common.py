"""
rice_common.py
==============
Shared utilities for the rice-grain technical-validation notebooks
(02_Unsupervised_validation.ipynb and 03_Supervised_validation.ipynb).

Keeping every shared function here means there is a SINGLE definition of the
data listing, the grain-ID parser, the leakage-safe splitter, the transforms
and the model factory. The notebooks only orchestrate; they never redefine
these. Place this file in the same folder as `config.py`.

Design notes tied to reviewer comments
---------------------------------------
* Labels come from the FOLDER name (the curated ground truth). The class
  acronym embedded in each filename is only used to (a) build a stable
  grain-ID and (b) CROSS-CHECK the folder label. The variety number in the
  filename (e.g. BPT5204 vs BPT5201) is deliberately ignored, so a stray
  numbering difference cannot mislabel a grain.  [robustness]
* `grain_id()` returns the filename stem, which is identical for an Original
  image and its Segmented counterpart. Splitting on this guarantees a physical
  grain never lands in two partitions at once.            [R4 #8, leakage]
* Every listing is sorted deterministically, so results do not depend on the
  operating system's directory order.                     [reproducibility]
"""

from __future__ import annotations
import os, random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

import config as cfg  # your existing config.py

# Natural sorting (BPT5204_FC_2 before BPT5204_FC_10). Falls back to plain
# sorted() if natsort is not installed, so the module still imports cleanly.
try:
    from natsort import natsorted, natsort_keygen
    _NAT_KEY = natsort_keygen()
    _HAVE_NATSORT = True
except Exception:  # pragma: no cover
    _HAVE_NATSORT = False
    def natsorted(seq, key=None):
        return sorted(seq, key=key)
    def _NAT_KEY(x):
        return str(x)

# ----------------------------------------------------------------------
# Class definitions
# ----------------------------------------------------------------------
# BPT 5204 lives under this sub-directory inside "Original image"/"Segmented image".
BPT_SUBDIR = "BPT5204"

# Folder name -> canonical class label (folders use mixed casing/underscores)
BPT_DIR_TO_LABEL: Dict[str, str] = {
    "Full_chalky":    "Full chalky",
    "Partial_chalky": "Partial chalky",
    "Healthy":        "Healthy",
    "Discolored":     "Discolored",
    "Peck_damage":    "Peck damage",
    "Broken":         "Broken",
}

# Filename acronym -> canonical class label (used for the cross-check only)
ACR_TO_LABEL: Dict[str, str] = {
    "FC": "Full chalky",
    "PC": "Partial chalky",
    "HY": "Healthy",
    "DC": "Discolored",
    "PD": "Peck damage",
    "BK": "Broken",
}

# The two class sets we report.
#   CLASSES_5 : the discrete-phenotype set used for the headline numbers.
#   CLASSES_6 : adds "Partial chalky" (the chalkiness continuum) and is the
#               key new experiment requested by four reviewers.  [R1#4,R3#2,R4#6,R5§2.1]
CLASSES_5: List[str] = ["Full chalky", "Healthy", "Discolored", "Peck damage", "Broken"]
CLASSES_6: List[str] = ["Full chalky", "Partial chalky", "Healthy", "Discolored", "Peck damage", "Broken"]

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int = None) -> None:
    """Seed Python, NumPy and Torch and put cuDNN in deterministic mode."""
    if seed is None:
        seed = cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding for reproducible shuffling/augmentation."""
    s = cfg.SEED + worker_id
    np.random.seed(s)
    random.seed(s)


def torch_generator():
    """A seeded torch.Generator for DataLoader shuffling."""
    import torch
    g = torch.Generator()
    g.manual_seed(cfg.SEED)
    return g


# ----------------------------------------------------------------------
# Grain identity and label parsing
# ----------------------------------------------------------------------
def grain_id(path) -> str:
    """Physical-grain identifier = filename stem (shared by original & segmented)."""
    return Path(path).stem


def parse_bpt_acronym(path) -> str | None:
    """Return the class acronym token (FC/PC/HY/DC/PD/BK) from a BPT filename,
    independent of the variety number. Returns None if no acronym is present."""
    for tok in Path(path).stem.split("_"):
        if tok.upper() in ACR_TO_LABEL:
            return tok.upper()
    return None


# ----------------------------------------------------------------------
# Dataset listing (deterministic, leakage-aware)
# ----------------------------------------------------------------------
def list_bpt(image_root, classes: List[str]) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    List BPT grain images under `image_root`/BPT_SUBDIR.

    Parameters
    ----------
    image_root : path to "Segmented image" (or "Original image")
    classes    : which class labels to include (CLASSES_5 or CLASSES_6)

    Returns
    -------
    rows       : sorted list of (path, label, grain_id)
    mismatches : number of files whose filename-acronym disagreed with the
                 folder label (should be 0; printed as a data-integrity check)
    """
    base = Path(image_root) / BPT_SUBDIR
    rows, mismatches = [], 0
    for folder in natsorted(os.listdir(base)):
        label = BPT_DIR_TO_LABEL.get(folder)
        if label is None or label not in classes:
            continue
        for p in natsorted((base / folder).glob("*"), key=lambda x: x.name):
            if p.suffix.lower() not in VALID_EXT:
                continue
            acr = parse_bpt_acronym(p)
            if acr is not None and ACR_TO_LABEL[acr] != label:
                mismatches += 1
            rows.append((str(p), label, grain_id(p)))
    # deterministic global order: by label, then natural filename order
    rows.sort(key=lambda r: (classes.index(r[1]), _NAT_KEY(r[0])))
    return rows, mismatches


def balanced_subset(rows, classes, max_per_class: int, seed: int = None):
    """
    Cap each class at `max_per_class` grains to build the balanced pool.
    Selection is deterministic (grains chosen by sorted grain-ID), so the
    same subset is reproduced on every run.                     [reproducibility]
    """
    if seed is None:
        seed = cfg.SEED
    by_class = {c: [] for c in classes}
    for r in rows:
        by_class[r[1]].append(r)
    out = []
    for c in classes:
        items = natsorted(by_class[c], key=lambda r: r[2])  # by grain-ID, natural order
        if len(items) > max_per_class:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(items), size=max_per_class, replace=False))
            items = [items[i] for i in idx]
        out.extend(items)
    out.sort(key=lambda r: (classes.index(r[1]), _NAT_KEY(r[2])))
    return out


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def make_transforms():
    """Return (train_tf, eval_tf). ImageNet normalisation for pretrained backbones."""
    import torchvision.transforms as T
    eval_tf = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])
    train_tf = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------
# Torch Dataset
# ----------------------------------------------------------------------
def make_dataset_class():
    """Factory so torch is only imported when needed."""
    from torch.utils.data import Dataset
    from PIL import Image

    class GrainDataset(Dataset):
        def __init__(self, rows, classes, transform):
            self.rows = rows
            self.classes = classes
            self.transform = transform
            self.cls2idx = {c: i for i, c in enumerate(classes)}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, label, gid = self.rows[i]
            img = Image.open(path).convert("RGB")
            return self.transform(img), self.cls2idx[label], gid

    return GrainDataset


# ----------------------------------------------------------------------
# Model factory  (headline ResNet18 + extra architectures for R2 #3)
# ----------------------------------------------------------------------
def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """
    Build a classifier with the backbone frozen except its last block + head.
    Supported: 'resnet18' (headline), 'resnet50', 'mobilenet_v3_large', 'vit_b_16'.
    """
    import torch.nn as nn
    import torchvision.models as models

    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        for p in m.parameters(): p.requires_grad = False
        for p in m.features[-2:].parameters(): p.requires_grad = True
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

    elif arch == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vit_b_16(weights=w)
        for p in m.parameters(): p.requires_grad = False
        for p in m.encoder.layers[-1].parameters(): p.requires_grad = True
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)

    elif arch == "swin_v2_b":
            w = models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.swin_v2_b(weights=w)
            for p in m.parameters(): p.requires_grad = False
            for p in m.features[-1].parameters(): p.requires_grad = True   # last Swin stage
            m.head = nn.Linear(m.head.in_features, num_classes)

    else:
        raise ValueError(f"unknown arch: {arch}")

    return m


def feature_extractor(pretrained: bool = True):
    """ResNet50 with the classifier removed -> 2048-d embeddings (unsupervised)."""
    import torch.nn as nn
    import torchvision.models as models
    w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Identity()
    m.eval()
    return m


# ----------------------------------------------------------------------
# Multi-backbone feature factory (unsupervised comparison, R2 #3/#4)
# ----------------------------------------------------------------------
# ResNet50 uses IMAGENET1K_V1 (as requested); the transformers/ConvNeXt use
# their best available ImageNet-1k weights. Each backbone returns its OWN
# preprocessing transform via weights.transforms(), so the (differing) input
# resolutions and normalisations are always correct.
FEATURE_BACKBONES = ["resnet50", "vit_b_16", "swin_v2_b", "convnext_base"]

def feature_backbone(arch: str, pretrained: bool = True):
    """
    Return (model, transform, embed_dim) for a headless feature extractor.
    `transform` is the backbone's native preprocessing (torchvision weights.transforms()).
    Supported: resnet50 (2048-d, V1), vit_b_16 (768-d), swin_v2_b (1024-d),
    convnext_base (1024-d).
    """
    import torch.nn as nn
    import torchvision.models as models

    if arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1
        m = models.resnet50(weights=w if pretrained else None)
        m.fc = nn.Identity(); dim = 2048
    elif arch == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1
        m = models.vit_b_16(weights=w if pretrained else None)
        m.heads = nn.Identity(); dim = 768
    elif arch == "swin_v2_b":
        w = models.Swin_V2_B_Weights.IMAGENET1K_V1
        m = models.swin_v2_b(weights=w if pretrained else None)
        m.head = nn.Identity(); dim = 1024
    elif arch == "convnext_base":
        w = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        m = models.convnext_base(weights=w if pretrained else None)
        m.classifier[2] = nn.Identity(); dim = 1024
    else:
        raise ValueError(f"unknown feature backbone: {arch}")

    m.eval()
    transform = w.transforms()          # native preprocessing for this backbone
    return m, transform, dim
