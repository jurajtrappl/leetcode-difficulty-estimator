# leetcode-difficulty-estimator

Semestral project for Neural Networks course at MFF, Charles University.

Authors: Mihal Filip, Trappl Juraj 2024.

The work is summarized in `slides.pdf`.

## Task

Given a text description of a programming problem, predict its difficulty - Easy/Medium/Hard. We tried both clasiffication and regression approaches.

## Data

Our dataset consists of 2366 free programming problems from the [LeetCode](https://leetcode.com/). We queried the LeetCode GraphQL API to get the data.

## Models

### Classification

- Perceptron, Linear SVM, MLP classifier, MLP with BERT embeddings as features

### Regression

- MLP regressor
