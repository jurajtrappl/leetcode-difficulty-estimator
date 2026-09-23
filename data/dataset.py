"""Shared data loading and the one train/(val)/test split that every model in this repo uses.

All settings (seed, split sizes, text cleaning, class weighting, dry-run sample size) come from
config.py, so every model sees exactly the same data.

No downsampling: all problems are kept. The class imbalance (~51% Medium) is handled with class
weights instead, and models are compared on macro-F1 on a test set that has the real class mix.

    from data.dataset import load_split, sample_weights, report
    X_train, X_test, y_train, y_test = load_split()
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(with_val=True)
"""
import json
import re

import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from config import CFG

LABELS = list(CFG.labels)


def html_to_text(html: str, keep_sup: bool = None) -> str:
    """Strip HTML. keep_sup=True writes 10<sup>5</sup> as 10^5 (plain get_text() gives '105')."""
    keep_sup = CFG.keep_superscripts if keep_sup is None else keep_sup
    soup = BeautifulSoup(html, "html.parser")
    if keep_sup:
        for tag in soup.find_all("sup"):
            tag.replace_with("^" + tag.get_text())
    return soup.get_text().replace("\xa0", " ")


def compact_statement(text: str) -> str:
    """Statement + constraints (+ follow-up), without the worked examples.

    The examples are long (arrays, explanations) and say little about difficulty; the constraints
    (n <= 10^5 ...) say a lot. Used by models with a short input window (Laya, the RNN).
    """
    statement = re.split(r"\n\s*Example\s*1\s*:", text, maxsplit=1)[0]
    m = re.search(r"Constraints\s*:(.*)", text, flags=re.S)
    tail = ("\nConstraints:" + m.group(1)) if m else ""
    s = statement.strip() + "\n" + tail
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def load_problems(keep_sup: bool = None):
    """All non-premium problems in file order -> (slugs, texts, labels); labels are 0/1/2 = Easy/Medium/Hard.

    With CFG.sample_limit = N > 0, a stratified sample of N problems (for quick dry runs).
    """
    with open(CFG.dataset_path, "r") as f:
        problems = json.load(f)
    slugs, texts, labels = [], [], []
    for slug, p in problems.items():
        if not p["content"]:  # premium problems have no content
            continue
        slugs.append(slug)
        texts.append(html_to_text(p["content"], keep_sup))
        labels.append(LABELS.index(p["difficulty"]))
    slugs, texts, labels = np.array(slugs), np.array(texts, dtype=object), np.array(labels)
    if 0 < CFG.sample_limit < len(labels):
        keep, _ = train_test_split(np.arange(len(labels)), train_size=CFG.sample_limit,
                                   random_state=CFG.seed, stratify=labels)
        keep = np.sort(keep)  # keep file order
        slugs, texts, labels = slugs[keep], texts[keep], labels[keep]
    return slugs, texts, labels


def split_indices(labels, with_val: bool = False):
    """Stratified row indices. The test part (CFG.test_size) never changes; with_val carves the
    validation set (CFG.val_size of the training part) out of the training part."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train, test = train_test_split(idx, test_size=CFG.test_size, random_state=CFG.seed, stratify=labels)
    if not with_val:
        return train, test
    train, val = train_test_split(train, test_size=CFG.val_size, random_state=CFG.seed, stratify=labels[train])
    return train, val, test


def load_split(with_val: bool = False, keep_sup: bool = None):
    """(X_train, X_test, y_train, y_test) or, with_val=True, (X_train, X_val, X_test, y_train, y_val, y_test)."""
    _, texts, labels = load_problems(keep_sup)
    parts = split_indices(labels, with_val)
    return tuple(texts[p] for p in parts) + tuple(labels[p] for p in parts)


def sample_weights(y):
    """Per-example weights for sklearn's fit(..., sample_weight=...), following CFG.class_weighting."""
    if CFG.class_weighting == "none":
        return np.ones(len(y))
    return compute_sample_weight(CFG.class_weighting, y)


def class_weights(y):
    """{class: weight} dict for Keras model.fit(..., class_weight=...), following CFG.class_weighting."""
    classes = np.unique(y)
    if CFG.class_weighting == "none":
        return {int(c): 1.0 for c in classes}
    w = compute_class_weight(CFG.class_weighting, classes=classes, y=y)
    return dict(zip(classes.tolist(), w.tolist()))


def report(y_true, y_pred, name: str = "") -> dict:
    """Metrics to compare models by. Always predicting Medium already gives ~51% accuracy,
    so look at macro-F1 (each class counts equally) and QWK (Easy->Hard is a worse mistake
    than Easy->Medium)."""
    r = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    print((name + ": " if name else "") + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    return r
