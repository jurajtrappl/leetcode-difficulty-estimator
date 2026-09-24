# LeetCode difficulty estimator

Predicts the difficulty of a LeetCode problem (Easy / Medium / Hard) from its statement. Tuned scikit-learn models,
a recurrent network and Llama 3.1 8B are compared on one shared test set.

**Key findings**

- **Text-only models plateau at about 0.50 macro-F1.** Five tuned TF-IDF + scikit-learn models and a BiGRU trained
  from scratch all score 0.50–0.53. Always answering "Medium" scores 0.23.
- **Llama 3.1 8B reaches 0.62**, run locally (4-bit, MLX) with six solved examples in the prompt, calibration and a
  decision rule tuned for macro-F1.
- **Llama has memorised LeetCode, but its score doesn't depend on it.** Given only the title, it does almost as well
  as with the full statement. On 876 problems published after its training cutoff, the title-only score drops by
  0.09 while the full-statement score holds (0.63).
- **Hard is the weak class for every model.** Most Hard problems are predicted as Medium (Hard recall 0.29–0.44 for
  the scikit-learn models and the RNN).

![Summary of results](confusion_matrices/summary.png)

## Authors

The project started in 2023–24 as a semestral project for the Neural Networks course at MFF, Charles University
(Faculty of Mathematics and Physics), by **Juraj Trappl** and **Filip Mihal**. Together we collected the problems
through LeetCode's GraphQL API and compared TF-IDF features with scikit-learn models, MLPs, RNNs, BERT embeddings and
few-shot prompting of Llama 2.

In 2026 I revisited it, reviewed the original work, fixed its mistakes and extended it:

- one evaluation protocol for all models: every problem kept, one stratified train/test split, class weights,
  macro-F1 and QWK, shared config and seeding;
- text cleaning that keeps exponents (`10<sup>5</sup>` used to become `105`);
- the same random-search budget for every scikit-learn model;
- a rebuilt RNN: statement and constraints only, numbers as order-of-magnitude tokens (`10^5` → `<1e5>`), vocabulary
  from the training data, masked BiLSTM/GRU, tuned and checked with cross-validation;
- Llama 3.1 8B in place of Llama 2, with calibration and a memorisation check on newly fetched problems;
- MLflow experiment tracking, resumable caches, and re-runs that reuse already trained models.

## Quick start

Python 3.12. The Llama notebook needs a Mac with Apple Silicon (MLX); everything else runs anywhere.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/rebuild_dataset.py     # downloads the problem statements (not in the repo, see Data), ~1 hour
jupyter lab                        # open any notebook and run it
docker compose up -d               # optional: MLflow UI on http://localhost:5050
```

## Data

2,366 free problems downloaded in December 2023 (51% Medium, 27% Easy, 22% Hard), plus 876 problems published later
(IDs 3000–4059) for the memorisation check.

The statements are LeetCode's content and are not in the repository. It contains only the problem lists with their
labels, in the original order (`data/problem_list.json`, `data/new_problem_list.json`), and
`python data/rebuild_dataset.py` downloads the statements for exactly those problems. LeetCode occasionally edits a
statement, so a rebuilt dataset can differ slightly.

## Evaluation

`data/dataset.py` makes one stratified 80/20 train/test split that every model uses; validation, when needed, comes
out of the training part. Settings are chosen on the training part, and each model sees the 474-problem test set
once. Always predicting Medium gives 51% accuracy, so the main score is **macro-F1** (each class counts equally),
with **QWK** (quadratic weighted kappa: Easy→Hard is a worse mistake than Easy→Medium) as the second. With 474 test
problems, the 95% interval on macro-F1 is about ±0.05.

## Notebooks

| Notebook | Models |
|---|---|
| `sklearn_pipeline.ipynb` | perceptron, linear SVM, RBF SVM, MLP classifier, MLP regressor (ordinal cut-offs) |
| `rnn.ipynb` | bidirectional LSTM/GRU trained from scratch |
| `llama_few_shot.ipynb` | Llama 3.1 8B Instruct, zero- and few-shot |

## Configuration

Shared settings live in `config.py`: seed, split sizes, CV folds, class weighting, text cleaning, model IDs and output
folders. Every notebook imports `CFG` and calls `set_seed()`; model-specific settings stay in their notebook. Any
shared setting can be overridden in the shell or in `.env` (see `.env.example`):

```sh
python config.py                  # print all settings
SEED=7 jupyter lab                # different seed for the split and the models
SAMPLE_LIMIT=300 jupyter lab      # dry run on a stratified sample of 300 problems
```

Outputs go to `confusion_matrices/` (plots), `trained_models/` (models and best settings) and `results/` (caches
and tables). `python plot_summary.py` redraws the summary figure from the result tables. Re-running a notebook reuses the saved searches and models as long as the settings haven't changed; a
changed setting retrains that part, and `RESEARCH = True` / `RETRAIN = True` at the top of a notebook forces it.

## Experiment tracking

Runs are logged with MLflow (`tracking.py`) into a local store in `mlflow/`, one experiment per model family, with
run names like `rnn-bigru-tuned-s42`. Each run stores the settings, the git commit, the scores, confusion matrices and
every tried setting; dry runs (`SAMPLE_LIMIT`) are tagged. Browse them with `docker compose up -d` at
http://localhost:5050, or with `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`. `TRACKING_ENABLED=false`
turns logging off.

## Results

All numbers are on the shared 474-problem test set (129 Easy / 242 Medium / 103 Hard).

| Model | Notebook | Accuracy | Macro-F1 | QWK |
|---|---|---|---|---|
| Always Medium | – | 0.511 | 0.225 | 0.000 |
| Llama 3.1 8B zero-shot, calibrated | `llama_few_shot` | 0.548 | 0.470 | 0.363 |
| MLP classifier, tuned (best scikit-learn model by CV) | `sklearn_pipeline` | 0.530 | 0.506 | 0.389 |
| Perceptron / linear SVM / RBF SVM / MLP regressor, tuned | `sklearn_pipeline` | 0.525–0.576 | 0.496–0.533 | 0.375–0.447 |
| BiGRU, tuned | `rnn` | 0.544 | 0.509 | 0.351 |
| Llama 3.1 8B 6-shot (variant picked on the calibration set) | `llama_few_shot` | 0.589 | **0.586** | **0.563** |
| Llama 3.1 8B 6-shot, calibrated + Easy/Hard shift for macro-F1 | `llama_few_shot` | 0.648 | **0.617** | **0.577** |

### scikit-learn (`sklearn_pipeline.ipynb`)

Five models, each with 30 random settings × 5-fold CV on the training part: word and character n-grams,
lowercasing, feature reduction (all n-grams, top-k% by label association, or SVD), class-weight strength and the
model's own parameters. The best model is picked by CV score; the test score only confirms it.

| Model | CV macro-F1 | Test accuracy | Test macro-F1 | Test QWK | Recall Easy / Medium / Hard |
|---|---|---|---|---|---|
| MLP classifier | **0.565 ± 0.026** | 0.530 | 0.506 | 0.389 | 0.55 / 0.58 / 0.39 |
| MLP regressor (ordinal cut-offs) | 0.549 ± 0.027 | 0.555 | 0.530 | 0.439 | 0.55 / 0.62 / 0.41 |
| RBF SVM | 0.544 ± 0.018 | 0.576 | 0.533 | 0.447 | 0.62 / 0.67 / 0.29 |
| Linear SVM | 0.541 ± 0.012 | 0.544 | 0.506 | 0.390 | 0.57 / 0.63 / 0.31 |
| Perceptron | 0.523 ± 0.014 | 0.525 | 0.496 | 0.375 | 0.54 / 0.60 / 0.34 |

All five are within 0.04 of each other on test, which is within noise. The regressor, which knows that
Easy < Medium < Hard, and the RBF SVM have the best QWK. Best settings: `trained_models/sklearn_best_params.json`;
every tried setting: `results/sklearn_search_<model>.csv`.

### RNN (`rnn.ipynb`)

Bidirectional LSTM/GRU trained from scratch on the statement and constraints, with numbers as order-of-magnitude
tokens (`10^5` → `<1e5>`). 20 random settings, the top 3 re-checked with 3-fold CV, the winner trained once more
and tested once.

- Winner: BiGRU with 64 units, mean pooling, 64-dim embeddings, dropout 0.3 and class-weight power 0.5
  (`trained_models/rnn_best_hp.json`).
- CV macro-F1 0.522 ± 0.008; test accuracy 0.544, macro-F1 0.509, QWK 0.351.
- Recall Easy / Medium / Hard: 0.42 / 0.66 / 0.44. It overfits within a few epochs (training macro-F1 ~0.85,
  validation ~0.48); early stopping keeps the best epoch.

![RNN confusion matrix](confusion_matrices/rnn_bigru.png)

### Llama 3.1 8B Instruct (`llama_few_shot.ipynb`)

4-bit, run locally with MLX in at most 16 GB. The model doesn't generate text: the notebook reads its next-token
probabilities for `Easy`, `Medium` and `Hard`, which take 99.8% of the probability mass. Calibration (temperature
and class bias) is fitted on 600 training problems, and the reported variant is chosen on those problems, not on the
test set.

| Variant | Calibration-set macro-F1 | Test accuracy | Test macro-F1 | Test QWK | Test ECE |
|---|---|---|---|---|---|
| 0-shot raw | 0.385 | 0.544 | 0.369 | 0.217 | 0.222 |
| 0-shot calibrated | 0.495 | 0.548 | 0.470 | 0.363 | 0.049 |
| **6-shot raw** (chosen) | **0.626** | 0.589 | **0.586** | **0.563** | 0.142 |
| 6-shot calibrated | 0.583 | 0.637 | 0.572 | 0.531 | 0.047 |
| 6-shot calibrated + Easy/Hard shift | – | 0.648 | 0.617 | 0.577 | – |

Two solved examples per class raise macro-F1 from 0.47 to 0.59. Calibration fixes the confidence (ECE 0.14 → 0.05)
but, because it optimises log-loss, trades Hard predictions for Medium. An extra Easy/Hard shift, tuned for
macro-F1 on the calibration problems, wins them back: Hard recall goes from 24% to 46%, macro-F1 to 0.617.

**Title only.** The same prompts, but with only the problem title:

| Input | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| Llama 0-shot, full statement | 0.548 | 0.470 | 0.363 |
| Llama 0-shot, **title only** | 0.582 | 0.493 | 0.396 |
| Llama 6-shot, full statement | 0.637 | 0.572 | 0.531 |
| Llama 6-shot, **title only** | 0.616 | 0.539 | 0.462 |
| TF-IDF + logistic regression, title only | 0.401 | 0.380 | 0.144 |

A title carries little information (a TF-IDF model on titles gets 0.38), yet Llama does about as well with the
title as with the full statement: it remembers these problems and their labels from its training data.

**Unseen problems.** `data/fetch_new_problems.py` fetched the 876 problems with IDs 3000–4059 (184 Easy / 433
Medium / 259 Hard), all published after Llama 3.1's training cutoff (December 2023). Prompts, calibration and shift
are unchanged.

| Input | Old test macro-F1 | New problems macro-F1 | Change | New QWK |
|---|---|---|---|---|
| TF-IDF + logistic regression, full statement (control) | 0.515 | 0.554 | +0.04 | 0.451 |
| Llama 0-shot calibrated, full statement | 0.470 | 0.533 | +0.06 | 0.437 |
| Llama 6-shot calibrated, full statement | 0.572 | **0.633** | +0.06 | 0.587 |
| Llama 6-shot calibrated + Easy/Hard shift | 0.617 | 0.622 | +0.00 | **0.593** |
| Llama 0-shot calibrated, **title only** | 0.493 | 0.429 | **−0.06** | 0.292 |
| Llama 6-shot calibrated, **title only** | 0.539 | 0.454 | **−0.09** | 0.334 |

- The new problems are slightly easier for every model: the TF-IDF control gains 0.04.
- Title only is the one input that drops, since the title of an unseen problem can't trigger a memory.
- The full statement gains as much as the control, so its score doesn't come from memory. Llama's lead over the
  control grows from 0.06 to 0.08.
- With the shift, Hard recall is 69% (179 / 259), at the cost of 163 of 433 Medium problems predicted as Hard. The
  new set has more Hard problems (30% vs 22%) than the problems the shift was tuned on.

![Llama on new problems](confusion_matrices/llama3.1-8b_new_problems.png)

## Final words

- difficulty labels are partly subjective, and a statement says only so much: text-only models stop around 0.50
  macro-F1
- Llama 3.1 8B does clearly better, also on problems it has never seen
- fun project :-)
