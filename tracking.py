"""Experiment tracking with MLflow: every model logs its runs the same way.

Layout in the MLflow UI:

    experiment  leetcode-difficulty-estimator/<family>          (sklearn, bert, rnn, laya, llama)
    run name    <family>-<model>[-<variant>]-s<seed>            e.g. sklearn-lsvm-tuned-s42

Every run gets tags (family, model, variant, seed, job_type, dry_run when SAMPLE_LIMIT is set, the git
commit) and all of config.py as parameters under `settings.*`, so any number can be traced back to
the seed, split and class weighting that produced it.

Storage is local and needs no server and no account: runs go to ./mlflow/mlflow.db, files (tables,
figures, confusion matrices) to ./mlflow/artifacts. To browse them, start the UI with
`docker compose up -d` (http://localhost:5050) or `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.

    from tracking import start_run, log_classification
    with start_run("sklearn", "lsvm", "tuned", config={"C": 0.1}) as run:
        ...
        run.summary["cv/macro_f1"] = 0.51          # number -> metric, text -> tag
        log_classification(run, y_test, predictions)

TRACKING_ENABLED=false turns logging off; without mlflow installed every call is a silent no-op.
"""
import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import CFG, ROOT
from data.dataset import LABELS, report

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # no anonymous usage stats to mlflow.org
os.environ.setdefault("DO_NOT_TRACK", "true")
try:
    import mlflow
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient
except ImportError:  # tracking is optional
    mlflow = None

MAX_PARAM_LEN = 6000


def _key(k: str) -> str:
    """MLflow keys allow letters, digits, _ - . : / and spaces: "a + b (c)" -> "a and b c"."""
    k = re.sub(r"\s*\+\s*", " and ", str(k)).replace("(", "").replace(")", "")
    return re.sub(r"[^\w\-.:/ ]", "_", k)[:250]


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        k = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flatten(v, k + ".")
        else:
            out[_key(k)] = str(v)[:MAX_PARAM_LEN]
    return out


class _Section:
    """run.summary / run.config: dict-like, writes through to MLflow."""

    def __init__(self, run, kind):
        self._run, self._kind, self._data = run, kind, {}

    def __setitem__(self, key, value):
        self.update({key: value})

    def __getitem__(self, key):
        return self._data[key]

    def update(self, values: dict):
        self._data.update(values)
        if self._run.run_id is None:
            return
        if self._kind == "config":
            self._run._log_params(_flatten(values))
        else:
            metrics = {k: v for k, v in values.items() if isinstance(v, (int, float)) and v == v}
            tags = {k: v for k, v in values.items() if k not in metrics}
            self._run.log(metrics)
            self._run._set_tags({f"summary.{k}": v for k, v in tags.items()})


class Run:
    """One MLflow run. Also works as a do-nothing stand-in when tracking is off (run_id None)."""

    def __init__(self, client=None, run_id=None):
        self.client, self.run_id = client, run_id
        self.summary, self.config = _Section(self, "summary"), _Section(self, "config")

    @property
    def active(self) -> bool:
        return self.run_id is not None

    def log(self, metrics: dict, step: int = 0):
        if self.active and metrics:
            ts = int(pd.Timestamp.now().timestamp() * 1000)
            self.client.log_batch(self.run_id, metrics=[Metric(_key(k), float(v), ts, step) for k, v in metrics.items()])

    def _log_params(self, params: dict):
        if self.active and params:
            items = list(params.items())
            for i in range(0, len(items), 100):  # MLflow accepts up to 100 params per batch
                self.client.log_batch(self.run_id, params=[Param(k, v) for k, v in items[i:i + 100]])

    def _set_tags(self, tags: dict):
        if self.active and tags:
            self.client.log_batch(self.run_id, tags=[RunTag(_key(k), str(v)[:MAX_PARAM_LEN]) for k, v in tags.items()])

    def finish(self, status: str = "FINISHED"):
        if self.active:
            self.client.set_terminated(self.run_id, status)
            self.run_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.finish("FAILED" if exc_type else "FINISHED")
        return False


def run_name(family: str, model: str, variant: str = None) -> str:
    return "-".join([family, model] + ([variant] if variant else []) + [f"s{CFG.seed}"])


def settings_dict() -> dict:
    """config.py as plain values (paths as strings)."""
    return {f.name: (str(v) if isinstance(v, Path) else v)
            for f in dataclasses.fields(CFG) for v in [getattr(CFG, f.name)]}


def _client():
    uri = CFG.mlflow_tracking_uri
    local = not uri
    if local:
        CFG.tracking_dir.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{CFG.tracking_dir / 'mlflow.db'}"
    return MlflowClient(tracking_uri=uri), local


def _experiment_id(client, name: str, local: bool) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    location = (CFG.tracking_dir / "artifacts" / name.replace("/", "_")).as_uri() if local else None
    return client.create_experiment(name, artifact_location=location)


def _git_tags() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return {"mlflow.source.git.commit": commit, "git_dirty": str(bool(dirty)).lower()} if commit else {}
    except OSError:
        return {}


def start_run(family: str, model: str, variant: str = None, config: dict = None,
              job_type: str = "train", notes: str = None) -> Run:
    """Start an MLflow run with the project-wide experiment, name, tags and settings.

    Use as a context manager (`with start_run(...) as run:`) or call `run.finish()` yourself.
    """
    if mlflow is None or not CFG.tracking_enabled:
        return Run()
    client, local = _client()
    exp_id = _experiment_id(client, f"{CFG.mlflow_experiment_prefix}/{family}", local)
    tags = {"family": family, "model": model, "variant": variant or "", "seed": str(CFG.seed),
            "job_type": job_type, **_git_tags()}
    if CFG.sample_limit:
        tags["dry_run"] = f"true (SAMPLE_LIMIT={CFG.sample_limit})"
    if notes:
        tags["mlflow.note.content"] = notes
    mlrun = client.create_run(exp_id, run_name=run_name(family, model, variant), tags=tags)
    run = Run(client, mlrun.info.run_id)
    run.config.update({"settings": settings_dict(), **(config or {})})
    print(f"MLflow run {run_name(family, model, variant)} ({mlrun.info.run_id[:8]}) in {CFG.mlflow_experiment_prefix}/{family}")
    return run


def log_classification(run: Run, y_true, y_pred, prefix: str = "test") -> dict:
    """Accuracy, macro-F1, QWK and per-class precision/recall/F1 as metrics, plus a confusion-matrix figure."""
    metrics = report(y_true, y_pred, name=prefix)
    if not run.active:
        return metrics
    prefix = _key(prefix)
    per_class = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS,
                                      output_dict=True, zero_division=0)
    values = {f"{prefix}/{k}": v for k, v in metrics.items()}
    for label in LABELS:
        for m, key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            values[f"{prefix}/{label}_{m}"] = per_class[label][key]
    run.log(values)

    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(list(y_true), list(y_pred), labels=[0, 1, 2])  # rows = true label
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(prefix)
    fig.tight_layout()
    run.client.log_figure(run.run_id, fig, f"{prefix}/confusion_matrix.png")
    plt.close(fig)
    run.client.log_text(run.run_id, json.dumps({"labels": LABELS, "rows_true_cols_pred": cm.tolist()}),
                        f"{prefix}/confusion_matrix.json")
    return metrics


def log_table(run: Run, key: str, df: pd.DataFrame):
    """A DataFrame as an MLflow table (viewable in the UI) and as a CSV next to it."""
    if run.active:
        df = df.astype({c: str for c in df.columns if df[c].dtype == object})
        run.client.log_table(run.run_id, data=df, artifact_file=f"tables/{_key(key)}.json")
        run.client.log_text(run.run_id, df.to_csv(index=False), f"tables/{_key(key)}.csv")


def log_image(run: Run, key: str, figure_or_path):
    """A matplotlib figure or an image file, stored as <key>.png."""
    if not run.active:
        return
    if isinstance(figure_or_path, (str, Path)):
        from PIL import Image
        run.client.log_image(run.run_id, Image.open(figure_or_path), f"{_key(key)}.png")
    else:
        run.client.log_figure(run.run_id, figure_or_path, f"{_key(key)}.png")


def keras_callbacks(run: Run) -> list:
    """A Keras callback that logs loss/metrics after every epoch as epoch/<name> ([] when tracking is off)."""
    if not run.active:
        return []
    import tensorflow as tf

    class MlflowEpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            run.log({f"epoch/{k}": float(v) for k, v in (logs or {}).items()}, step=epoch)

    return [MlflowEpochLogger()]
