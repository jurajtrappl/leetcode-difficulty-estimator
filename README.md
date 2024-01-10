# leetcode-difficulty-estimator

Semestral project for Neural Networks course at MFF, Charles University.

Authors: Mihal Filip, Trappl Juraj 2024.

The work is summarized in `slides.pdf`.

## Task

Given a text description of a programming problem, predict its difficulty - Easy/Medium/Hard. We tried both clasiffication and regression approaches.

## Data

Our dataset consists of 2366 free programming problems from the [LeetCode](https://leetcode.com/). We queried the LeetCode GraphQL API to get the data.
Class imbalance present (~50% medium problems, the rest are ~equal proportions of Easy and Hard problems).

## Models

### Classification

- Perceptron, Linear SVM, MLP classifier, MLP with BERT embeddings as features

### Regression

- MLP regressor

## Results

Best parameters are written in `slides.pdf`.

**Shallow learning models**:
Obtained from stratified 5-fold CV that averages f1 scores, tf-idf features:

| model | average f1 macro | average f1 micro |
|---|---|---|
| Perceptron | 58.26 | 58.56 |
| SVM linear kernel | 60.94 | 61.15 |
| MLP classifier | 59.42 | 59.61 |

**Classifier on top of contextualized BERT embeddings**:

- `bert-base-uncased`
  - no fine-tuning (not enough training examples)
  - mean pooling, final feature size is _(768,)_
- Tensorflow 2.12
  - HyperBand hyperparameter optimization from KerasTuner

| model | test accuracy |
|---|---|
| 1. layer embeddings | 51.7 |
| 2. layer embeddings | 49.1 |

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
