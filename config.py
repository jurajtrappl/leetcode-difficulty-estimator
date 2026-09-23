"""Project-wide settings: one place for everything that must be the same across all models.

Every script and notebook does

    from config import CFG, set_seed
    set_seed()                      # after importing numpy / tensorflow / torch

and reads CFG.seed, CFG.test_size, CFG.figures_dir, ... instead of hard-coding them.

Override any value without touching code: set an environment variable with the field's name
in upper case, in the shell or in .env, e.g.

    SEED=7 jupyter lab
    SAMPLE_LIMIT=300 jupyter lab          # quick dry run on 300 problems

Model-specific knobs (batch size, epochs, tuner search ranges, TF-IDF settings) stay next to
their model on purpose.
"""
import os
import random
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv missing: plain environment variables still work
    pass


@dataclass(frozen=True)
class Settings:
    # --- reproducibility --------------------------------------------------------------------
    seed: int = 42                      # every split, CV fold, model init and tuner uses this

    # --- data -------------------------------------------------------------------------------
    dataset_path: Path = ROOT / "data" / "leetcode_problems_dataset.json"
    keep_superscripts: bool = True      # 10<sup>5</sup> -> "10^5" (False reproduces the old "105")
    sample_limit: int = 0               # 0 = all problems; N = stratified sample of N (dry runs)

    # --- evaluation protocol ----------------------------------------------------------------
    test_size: float = 0.2              # held-out test part, identical for every model
    val_size: float = 0.15              # validation part, carved out of the training part
    cv_folds: int = 5                   # k for cross-validation on the training part
    class_weighting: str = "balanced"   # "balanced" (rare classes count more) or "none"

    # --- pretrained models ------------------------------------------------------------------
    bert_model: str = "bert-base-uncased"
    laya_model: str = "convaiinnovations/laya"
    llama_model: str = "meta-llama/Llama-2-13b-chat-hf"
    max_tokens: int = 512               # truncation length for transformer inputs

    # --- outputs ----------------------------------------------------------------------------
    figures_dir: Path = ROOT / "confusion_matrices"     # plots (committed)
    models_dir: Path = ROOT / "trained_models"           # saved Keras models + best hyperparameters
    embeddings_dir: Path = ROOT / "bert_embeddings"      # embedding-projector TSVs
    results_dir: Path = ROOT / "results"                 # caches, tuner trials, result tables
    fig_dpi: int = 200

    # --- experiment tracking (MLflow, see tracking.py) ------------------------------------------
    tracking_enabled: bool = True       # False = run everything without logging
    mlflow_experiment_prefix: str = "leetcode-difficulty-estimator"   # experiments: <prefix>/<family>
    mlflow_tracking_uri: str = ""       # "" = local store in ./mlflow (no server needed);
                                        # or e.g. http://localhost:5050 to log through a server
    tracking_dir: Path = ROOT / "mlflow"   # local store: mlflow.db (runs, params, metrics) + artifacts/

    @property
    def labels(self):
        return ("Easy", "Medium", "Hard")

    @property
    def tuner_dir(self) -> Path:
        return self.results_dir / "tuner"


def _cast(value: str, default):
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, Path):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p
    return type(default)(value)


def _from_env(base: Settings) -> Settings:
    overrides = {}
    for f in fields(base):
        raw = os.environ.get(f.name.upper())
        if raw not in (None, ""):
            overrides[f.name] = _cast(raw, getattr(base, f.name))
    return replace(base, **overrides)


CFG = _from_env(Settings())

for _d in (CFG.figures_dir, CFG.models_dir, CFG.embeddings_dir, CFG.results_dir):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = None) -> int:
    """Seed Python, NumPy and whichever of TensorFlow / PyTorch is already imported.

    It never imports a framework itself (importing TensorFlow before PyTorch/transformers can
    crash the process), so call it after your own imports.
    """
    seed = CFG.seed if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects subprocesses started from here
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].keras.utils.set_random_seed(seed)  # also seeds python + numpy
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


def describe() -> str:
    """One line per setting, for printing at the top of a run."""
    return "\n".join(f"{f.name:18} {getattr(CFG, f.name)}" for f in fields(CFG))


if __name__ == "__main__":
    print(describe())
