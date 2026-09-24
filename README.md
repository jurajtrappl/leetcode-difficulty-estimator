# leetcode-difficulty-estimator

Semestral project for Neural Networks course at MFF, Charles University.

Authors: Mihal Filip, Trappl Juraj 2024.

The work is summarized in `slides.pdf`.

## Task

Given a text description of a programming problem, predict its difficulty - Easy/Medium/Hard. We tried both clasiffication and regression approaches.

## Data

Our dataset consists of 2366 free programming problems from the [LeetCode](https://leetcode.com/). We queried the LeetCode GraphQL API to get the data.
Class imbalance present (~51% Medium, 27% Easy, 22% Hard).

All models use `data/dataset.py`: it keeps **all** problems (no downsampling), makes one stratified 80/20 train/test split
(validation, when needed, is carved out of the training part), and handles the imbalance with balanced class weights.
Always predicting Medium already gives ~51% accuracy, so models are compared by **macro-F1** (every class counts equally)
and **QWK** (quadratic weighted kappa: Easy→Hard is a worse mistake than Easy→Medium) on the same 474-problem test set.
With that test size, the 95% interval on macro-F1 is roughly ±5 points.

## Models

### Classification

- Perceptron, Linear SVM, RBF SVM, MLP classifier (`sklearn_pipeline.ipynb`)
- Bidirectional LSTM/GRU trained from scratch (`rnn.ipynb`)
- Llama 3.1 8B Instruct, zero- and few-shot (`llama_few_shot.ipynb`)
- Laya decision model: zero-shot, calibrated, and as a feature extractor (`laya_experiments.ipynb`)
- Laya fine-tuned: decision head + top encoder layers trained on our training split (`laya_finetune.ipynb`)

### Regression

- MLP regressor

## Configuration

Settings that must be identical for every model live in `config.py`: seed, split sizes, CV folds, class weighting,
text cleaning, pretrained model IDs, output folders and plot DPI. Every script and notebook imports `CFG` from there
and calls `set_seed()`, which seeds Python, NumPy and whichever of TensorFlow / PyTorch is loaded.
Model-specific knobs (batch size, epochs, tuner ranges, TF-IDF settings) stay next to their model.

Override any setting without editing code, in the shell or in `.env` (see `.env.example`):

```sh
python config.py                          # print all settings
SEED=7 jupyter lab                        # different seed for split + models
SAMPLE_LIMIT=300 jupyter lab              # dry run on a stratified 300-problem sample
```

Outputs: plots → `confusion_matrices/`, saved models and best hyperparameters → `trained_models/`,
caches, result tables and tuner trials → `results/`.

Re-running a notebook reuses what was already trained, as long as its settings haven't changed: the sklearn
searches (`trained_models/sklearn_<model>.joblib`, only the best setting is refitted), the RNN re-check and model
(`rnn.keras` + `rnn_meta.json`), the fine-tuned Laya (`laya_finetuned.pt`), and the cached Llama / Laya answers in
`results/`. A changed setting retrains that part automatically; `RESEARCH = True` / `RETRAIN = True` at the top
of a notebook forces it.

## Experiment tracking (MLflow)

Every notebook logs its runs through `tracking.py` into a local MLflow store in `mlflow/` (no server, no account,
no license). Experiments are per model family, runs share one naming scheme: `<family>-<model>[-<variant>]-s<seed>`.

| Notebook | Experiment / runs | What is logged |
|---|---|---|
| `sklearn_pipeline.ipynb` | `.../sklearn`: `sklearn-<model>-tuned-s42` ×5, `sklearn-comparison-s42` | search space, best setting, CV + test scores, confusion matrix, every tried setting as a table |
| `rnn.ipynb` | `.../rnn`: `rnn-bi<lstm\|gru>-tuned-s42` | every tried setting, CV re-check of the top 3, per-epoch curves, test scores, confusion matrix |
| `laya_experiments.ipynb` | `.../laya`: `laya-<checkpoint>-s42`, `laya-<checkpoint>-extras-s42` | the full results table, calibration metrics, all figures; F1 bias tuning and the new-problems table |
| `laya_finetune.ipynb` | `.../laya`: `laya-finetune-top<k>-s42` | per-epoch loss / validation macro-F1, chosen variant, test + new-problem scores, confusion matrices |
| `llama_few_shot.ipynb` | `.../llama`: `llama-llama3.1-8b-<0\|6>shot-s42` | raw + calibrated test scores, log-loss / ECE, confusion matrices, prompt settings |

Every run also stores all of `config.py` as parameters (`settings.*`) and the git commit, and runs with
`SAMPLE_LIMIT` set are tagged `dry_run`, so you can filter them out.

Browse the runs with `docker compose up -d` → http://localhost:5050 (or, without Docker,
`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`). `TRACKING_ENABLED=false` turns logging off; without
mlflow installed everything still runs, just without logging.

## Results

Everything below is measured on the **same 474-problem test set** (the real class mix: 129 Easy / 242 Medium /
103 Hard), with settings chosen on training data only. **Macro-F1** is the main score (each class counts one
third); always answering Medium gets 0.511 accuracy but only 0.225 macro-F1. With 474 test problems the 95%
interval on macro-F1 is about **±0.05**, so smaller differences are noise.

### Overview

| Model | Notebook | Accuracy | Macro-F1 | QWK |
|---|---|---|---|---|
| Always Medium | – | 0.511 | 0.225 | 0.000 |
| Laya zero-shot (`choice` question) | `laya_experiments` | 0.468 | 0.300 | 0.097 |
| Llama 3.1 8B zero-shot, calibrated | `llama_few_shot` | 0.548 | 0.470 | 0.363 |
| TF-IDF + linear SVM (one fixed setting) | `laya_experiments` | 0.544 | 0.477 | 0.362 |
| Laya embeddings + signal questions → logistic regression | `laya_experiments` | 0.494 | 0.494 | 0.398 |
| MLP classifier, tuned (best sklearn model by CV) | `sklearn_pipeline` | 0.530 | 0.506 | 0.389 |
| BiGRU, tuned | `rnn` | 0.544 | 0.509 | 0.351 |
| Llama 3.1 8B 6-shot (variant picked on the calibration set) | `llama_few_shot` | 0.589 | **0.586** | **0.563** |
| Llama 3.1 8B 6-shot, calibrated + Easy/Hard shift for macro-F1 | `llama_few_shot` | 0.648 | **0.617** | **0.577** |

**What it says, in short**

1. **Every model that only reads the text lands at about 0.50 macro-F1.** TF-IDF, the tuned MLP, the RNN and a
   linear probe on Laya's encoder are all within 0.48–0.51, inside each other's noise band. How the text is turned
   into numbers matters much less than one would hope.
2. **Llama is the only model clearly above that, and the lead is real.** It *has* memorised LeetCode: given only
   the title, it is almost as good as with the whole statement. But on 876 problems published after its training
   cutoff it does just as well (6-shot macro-F1 0.63 vs 0.55 for a TF-IDF model), so its score with the full
   statement doesn't depend on that memory (see the Llama section).
3. **Hard is the weak class for everyone.** Hard recall is 0.29–0.44 for the sklearn models and the RNN; most Hard
   problems are called Medium.
4. The old ~58–62% accuracy numbers of the 2024 version were inflated by a leak in the downsampling (below).

### Shallow models (`sklearn_pipeline.ipynb`)

Five models, each with the same random-search budget (30 settings × 5-fold CV on the training part): TF-IDF word
and character n-grams, lowercasing, feature reduction (all n-grams / top-k% by label association / SVD),
class-weight strength, and each model's own knobs. The model to trust is the one with the best **CV** score; the
test columns only confirm.

| Model | CV macro-F1 | Test accuracy | Test macro-F1 | Test QWK | Recall Easy / Medium / Hard |
|---|---|---|---|---|---|
| MLP classifier | **0.565 ± 0.026** | 0.530 | 0.506 | 0.389 | 0.55 / 0.58 / 0.39 |
| MLP regressor (ordinal cut-offs) | 0.549 ± 0.027 | 0.555 | 0.530 | 0.439 | 0.55 / 0.62 / 0.41 |
| RBF SVM | 0.544 ± 0.018 | 0.576 | 0.533 | 0.447 | 0.62 / 0.67 / 0.29 |
| Linear SVM | 0.541 ± 0.012 | 0.544 | 0.506 | 0.390 | 0.57 / 0.63 / 0.31 |
| Perceptron | 0.523 ± 0.014 | 0.525 | 0.496 | 0.375 | 0.54 / 0.60 / 0.34 |

All five are within 0.04 of each other on test, i.e. statistically tied. The regressor, which knows that
Easy < Medium < Hard, has the best QWK together with the RBF SVM. Best settings: `trained_models/sklearn_best_params.json`;
every tried setting: `results/sklearn_search_<model>.csv` and MLflow. Confusion matrices:
`confusion_matrices/sklearn_tuned_all.png`.

### RNN (`rnn.ipynb`)

Bidirectional LSTM/GRU trained from scratch on statement + constraints, numbers turned into order-of-magnitude
tokens (`10^5` → `<1e5>`). 20 random settings (11 min), the top 3 re-checked with 3-fold CV, the winner trained
once more and tested once.

* Winner: **BiGRU**, 64 units, mean pooling, 64-dim embeddings, dropout 0.3, class-weight power 0.5
  (`trained_models/rnn_best_hp.json`).
* CV macro-F1 0.522 ± 0.008 → test accuracy 0.544, **macro-F1 0.509**, QWK 0.351.
* Recall Easy / Medium / Hard: 0.42 / 0.66 / 0.44. It overfits quickly (training macro-F1 ~0.85 after a few
  epochs vs ~0.48 on validation), which early stopping catches.

![RNN confusion matrix](confusion_matrices/rnn_bigru.png)

### Llama 3.1 8B Instruct (`llama_few_shot.ipynb`)

4-bit, run locally with MLX (≤16 GB). No text generation: we read the model's next-token probabilities for
`Easy` / `Medium` / `Hard` (they get 99.8% of the probability mass). Calibration (temperature + class bias) is
fitted on 600 training problems; the variant to report is chosen on those, not on test.

| Variant | Calibration-set macro-F1 | Test accuracy | Test macro-F1 | Test QWK | Test ECE |
|---|---|---|---|---|---|
| 0-shot raw | 0.385 | 0.544 | 0.369 | 0.217 | 0.222 |
| 0-shot calibrated | 0.495 | 0.548 | 0.470 | 0.363 | 0.049 |
| **6-shot raw** (chosen) | **0.626** | 0.589 | **0.586** | **0.563** | 0.142 |
| 6-shot calibrated | 0.583 | 0.637 | 0.572 | 0.531 | 0.047 |
| 6-shot calibrated + Easy/Hard shift | – | 0.648 | 0.617 | 0.577 | – |

* Two solved examples per class help a lot (0.47 → 0.59). Calibration fixes the confidence (ECE 0.14 → 0.05)
  but, as it optimises log-loss, it trades Hard predictions for Medium; the extra Easy/Hard shift (tuned for
  macro-F1 on the calibration set) buys them back: Hard recall 24% → 46%, macro-F1 0.617.

**Does it read the problem, or remember it?** Same model, but the prompt contains only the problem *title*:

| Input | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| Llama 0-shot, full statement | 0.548 | 0.470 | 0.363 |
| Llama 0-shot, **title only** | 0.582 | 0.493 | 0.396 |
| Llama 6-shot, full statement | 0.637 | 0.572 | 0.531 |
| Llama 6-shot, **title only** | 0.616 | 0.539 | 0.462 |
| TF-IDF + logistic regression, title only (no memory) | 0.401 | 0.380 | 0.144 |

A title alone carries little information (the no-memory model gets 0.38), yet Llama is about as good with the title
as with the whole statement: the LeetCode problems and their labels were in its training data, and it remembers them.

**Does the memory inflate the real score?** The fair test is on problems it cannot have seen:
`data/fetch_new_problems.py` fetched the 876 problems with ID 3000–4059 (184 Easy / 433 Medium / 259 Hard), all
published after Llama 3.1's training cutoff (December 2023). Nothing is re-fitted: the same prompts, calibration and
Easy/Hard shift as above.

| Input | Old test macro-F1 | New problems macro-F1 | Change | New QWK |
|---|---|---|---|---|
| TF-IDF + logistic regression, full statement (no-memory control) | 0.515 | 0.554 | +0.04 | 0.451 |
| Llama 0-shot calibrated, full statement | 0.470 | 0.533 | +0.06 | 0.437 |
| Llama 6-shot calibrated, full statement | 0.572 | **0.633** | +0.06 | 0.587 |
| Llama 6-shot calibrated + Easy/Hard shift | 0.617 | 0.622 | +0.00 | **0.593** |
| Llama 0-shot calibrated, **title only** | 0.493 | 0.429 | **−0.06** | 0.292 |
| Llama 6-shot calibrated, **title only** | 0.539 | 0.454 | **−0.09** | 0.334 |

* The new problems are a bit *easier* to classify for everyone (the no-memory control gains 0.04).
* **Title only** is the one input that gets worse: on unseen problems a title can't trigger a memory. That confirms
  the memorisation.
* **Full statement** gains as much as the control, so it was not being carried by memory. Llama's lead over the
  no-memory model is, if anything, larger on unseen problems (0.08 vs 0.06 macro-F1). It really reads the problem.
* With the shift, Hard recall on the new problems is 69% (179 / 259). The cost is Medium problems called Hard
  (163 / 433): the new set has more Hard problems (30% vs 22%), and the shift was tuned on the old mix.

![Llama on new problems](confusion_matrices/llama3.1-8b_new_problems.png)

### Laya (`laya_experiments.ipynb`)

Laya is a ModernBERT-large encoder with a small decision head: one forward pass returns calibrated probabilities
for multiple-choice, score and yes/no questions. We use it three ways:

| Use | Test accuracy | Test macro-F1 | QWK |
|---|---|---|---|
| Zero-shot, `choice` question (Easy / Medium / Hard with descriptions) | 0.468 | 0.300 | 0.097 |
| Zero-shot, `score` question (ordinal 0–2) | 0.462 | 0.311 | 0.135 |
| Zero-shot + temperature/bias calibration | 0.508 | 0.225 | ≈ 0 |
| 12 yes/no **signal questions** ("needs DP?", "input ≥ 10^5?", ...) → logistic regression | 0.420 | 0.416 | 0.271 |
| Encoder embeddings → logistic regression (linear probe) | 0.475 | 0.476 | 0.373 |
| Embeddings + signals → logistic regression | 0.494 | **0.494** | **0.398** |

* **Zero-shot doesn't work**: it answers Medium 85% of the time, and calibration collapses it to always-Medium,
  which shows its difficulty answer carries almost no information. The model card warns about this: the base
  checkpoint is "a fast base to specialise, not a zero-shot decision engine".
* **Its encoder does know something**: a plain linear model on its embeddings matches TF-IDF and is within noise of
  the tuned RNN, and its probabilities are well calibrated (ECE 0.043).
* **The signals are readable**: "Can the input size be 10^5 or larger?" is the strongest push towards Hard, then
  dynamic programming, advanced data structures and "combines several ideas"; "design a class", "brute force is
  fast enough" and a confident Easy answer push towards Easy (`confusion_matrices/laya_signal_weights.png`).
  Each signal alone differs only a little between classes (e.g. large input: 13% / 25% / 27% of Easy / Medium /
  Hard problems).
* Laya shows no sign of having memorised LeetCode, so its numbers are "reading" numbers.

*Pending*: Easy/Hard bias tuning and the new-problems check (Extras of `laya_experiments.ipynb`), and fine-tuning
Laya's head + top encoder layers on our training split (`laya_finetune.ipynb`).

### What was wrong in the 2024 version

* **Downsampling leak.** The balanced set kept the *first* 514 problems of each class in LeetCode-ID order: all Hard
  problems, but Medium only from the oldest ~940. Problem age then predicted the label (a decision tree on the
  problem position alone reached ~56%), which inflated the reported ~58–62% accuracy, and a third of the problems
  (824 of 2366) were thrown away. Now all problems are kept, class weights handle the imbalance, and macro-F1 is reported.
* **HTML cleaning** turned `10<sup>5</sup>` into `105`, so "n ≤ 10^5" read as "n ≤ 105" in ~65% of problems. Fixed
  (`10^5`); for TF-IDF it hardly matters (0.484 vs 0.477 macro-F1), for models that read it does.
* **Llama 2 prompt bug.** The 2024 prompt showed the same Easy example three times, labelled easy, medium and hard;
  its ~40% accuracy was below always guessing Medium (51%).
* **RNN.** Vocabulary built from all problems (test included), stopwords like "at most" removed, every number
  replaced by one `<NUMBER>` token (losing the constraint sizes), training data never shuffled.

## Final words

- generally a hard task: difficulty labels are partly subjective, and a problem's text says only so much about them
- with an honest setup (no leak, one shared test set, tuned by CV) every model that only reads the text ends up
  around 0.50 macro-F1; more data or a model that understands algorithms would be needed to go further
- the only model clearly above that is Llama 3.1 8B; it has memorised LeetCode, but it keeps its lead on problems
  published after its training, so the lead comes from reading, not remembering
- fun project :-)
