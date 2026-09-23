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

## Experiment tracking (MLflow)

Every notebook logs its runs through `tracking.py` into a local MLflow store in `mlflow/` (no server, no account,
no license). Experiments are per model family, runs share one naming scheme: `<family>-<model>[-<variant>]-s<seed>`.

| Notebook | Experiment / runs | What is logged |
|---|---|---|
| `sklearn_pipeline.ipynb` | `.../sklearn`: `sklearn-<model>-tuned-s42` ×5, `sklearn-comparison-s42` | search space, best setting, CV + test scores, confusion matrix, every tried setting as a table |
| `rnn.ipynb` | `.../rnn`: `rnn-bi<lstm\|gru>-tuned-s42` | every tried setting, CV re-check of the top 3, per-epoch curves, test scores, confusion matrix |
| `laya_experiments.ipynb` | `.../laya`: `laya-<checkpoint>-s42` | the full results table, calibration metrics, all figures |
| `llama_few_shot.ipynb` | `.../llama`: `llama-llama3.1-8b-<0\|6>shot-s42` | raw + calibrated test scores, log-loss / ECE, confusion matrices, prompt settings |

Every run also stores all of `config.py` as parameters (`settings.*`) and the git commit, and runs with
`SAMPLE_LIMIT` set are tagged `dry_run`, so you can filter them out.

Browse the runs with `docker compose up -d` → http://localhost:5050 (or, without Docker,
`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`). `TRACKING_ENABLED=false` turns logging off; without
mlflow installed everything still runs, just without logging.

## Results

Best parameters are written in `slides.pdf`.

**Shallow learning models** (`sklearn_pipeline.ipynb`): perceptron, linear SVM, RBF SVM, MLP classifier and
MLP regressor. Every model gets the same random-search budget (TF-IDF n-grams, lowercasing, feature reduction,
class-weight strength and its own knobs), tuned by cross-validation on the training part only, and is evaluated
on the test set once. The notebook also shows which choices mattered and analyses the best model's mistakes.
Always predicting Medium scores 51.1% accuracy / 22.5 macro-F1 on the test set.

| model | CV macro-F1 | test accuracy | test macro-F1 | test QWK |
|---|---|---|---|---|
| (run `sklearn_pipeline.ipynb`; results land in `results/sklearn_tuned_results.csv` and MLflow) | | | | |

The earlier numbers here (~58-61%) were measured on a downsampled, balanced set that kept the *first* 514 problems of each
class in LeetCode-ID order (all Hard problems, but Medium only from the oldest ~940). Problem age then leaked the label,
which inflated the scores, and 57% of the data was thrown away.

**RNN** (`rnn.ipynb`): bidirectional LSTM/GRU trained from scratch on statement + constraints, with numbers turned
into order-of-magnitude tokens (`10^5` → `<1e5>`). 20 random settings, top 3 re-checked with 3-fold CV, macro-F1
as the score; needs a re-run.

**In-context learning classification** (`llama_few_shot.ipynb`)

Llama 3.1 8B Instruct, 4-bit, run locally with MLX on Apple Silicon (≤16 GB memory). No text generation: the model
reads a Llama 3 chat prompt (rules, optionally 2 solved examples per class from the training part) and we take its
next-token probabilities for `Easy` / `Medium` / `Hard`. Zero-shot and 6-shot, each raw and calibrated
(temperature + class bias fitted on 600 training problems); evaluated on the shared test set; needs a run.

The original 2024 version (Llama 2 13B via LangChain, ~40% accuracy on all problems) had a prompt bug: the same
Easy example was shown three times, labelled easy, medium and hard. Its 40% is below always guessing Medium (51%).

## Final words

- generally a hard task, difficulty of task may not seem objective
- hyperparameters were not so much optimized
  - they were only optimized for dense classifiers, but those does not have enough training data
- not enough training data for models to be able to generalize well
- fun project :-)
