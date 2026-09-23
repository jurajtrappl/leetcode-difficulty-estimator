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

- Perceptron, Linear SVM, MLP classifier, MLP with BERT embeddings as features

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
embedding-projector files → `bert_embeddings/`, caches and tuner trials → `results/`.

## Experiment tracking (MLflow)

Every notebook logs its runs through `tracking.py` into a local MLflow store in `mlflow/` (no server, no account,
no license). Experiments are per model family, runs share one naming scheme: `<family>-<model>[-<variant>]-s<seed>`.

| Notebook | Experiment / runs | What is logged |
|---|---|---|
| `sklearn_pipeline.ipynb` | `.../sklearn`: `sklearn-<model>-tuned-s42` ×5, `sklearn-comparison-s42` | search space, best setting, CV + test scores, confusion matrix, every tried setting as a table |
| `bert_embeddings_mlp.ipynb` | `.../bert`: `bert-<experiment>-s42` | best hyperparameters, per-epoch loss/metrics, test scores, confusion matrix |
| `rnn.ipynb` | `.../rnn`: `rnn-bi<lstm\|gru>-tuned-s42` | every tried setting, CV re-check of the top 3, per-epoch curves, test scores, confusion matrix |
| `laya_experiments.ipynb` | `.../laya`: `laya-<checkpoint>-s42` | the full results table, calibration metrics, all figures |
| `llama2-few-shot-leetcode.ipynb` | `.../llama`: `llama-<model>-few-shot-s42` | accuracy / macro-F1 / QWK, confusion matrix |

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

**Classifier on top of contextualized BERT embeddings**:

- `bert-base-uncased`
  - no fine-tuning (not enough training examples)
  - mean pooling, final feature size is _(768,)_
- Keras 3 (TensorFlow 2.21); originally TensorFlow 2.12
  - HyperBand hyperparameter optimization from KerasTuner

Embeddings are computed with the PyTorch `BertModel`; the classifier is Keras 3 (TensorFlow 2.21), trained with class weights and tuned on validation balanced accuracy.

| model | test accuracy |
|---|---|
| 1. layer embeddings | to re-run (old: 51.7, downsampled) |
| 2. layer embeddings | to re-run (old: 49.1, downsampled) |

**RNN** (`rnn.ipynb`): bidirectional LSTM/GRU trained from scratch on statement + constraints, with numbers turned
into order-of-magnitude tokens (`10^5` → `<1e5>`). 20 random settings, top 3 re-checked with 3-fold CV, macro-F1
as the score; needs a re-run.

**In-context learning classification**

Using Llama-13b-chat from HF. Selected one representative from each difficulty (tried to take a problem with ~25% acceptance rate) and created a few-shot learning prompt. Built with LangChain.

```py
prompt = PromptTemplate.from_template(
    """
    <s>[INST] <<SYS>>
    Task: Given a programming problem description, predict its difficulty.
    The difficulty can be one of easy, medium and hard.
    
    Example:
    Given a programming problem description: {programming_problem_example_1}, the difficulty is:
    easy

    Example:
    Given a programming problem description: {programming_problem_example_1}, the difficulty is:
    medium
    
    Example:
    Given a programming problem description: {programming_problem_example_1}, the difficulty is:
    hard

    <<SYS>>
    Now, given a programming problem description: {programming_problem}, the difficulty is:
    [/INST]
    """
)
```

3 training examples, 2360 testing examples - ~40% accuracy.

## Final words

- generally a hard task, difficulty of task may not seem objective
- hyperparameters were not so much optimized
  - they were only optimized for dense classifiers, but those does not have enough training data
- not enough training data for models to be able to generalize well
- fun project :-)
