#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt

import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.svm
import csv
import sklearn.model_selection

from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--c_n", default=5, type=int, help="Character n-grams")
parser.add_argument("--c_l", default=False, action="store_true", help="Character lowercase")
parser.add_argument("--c_tf", default="binary", type=str, help="Character TF type")
parser.add_argument("--c_mf", default=None, type=int, help="Character max features")
parser.add_argument("--c_wb", default=False, action="store_true", help="Character wb")
parser.add_argument("--model", default="lsvm", type=str, help="Model type")
parser.add_argument("--w_n", default=3, type=int, help="Word n-grams")
parser.add_argument("--w_l", default=False, action="store_true", help="Word lowercase")
parser.add_argument("--w_tf", default="log", type=str, help="Word TF type")
parser.add_argument("--w_mf", default=None, type=int, help="Word max features")
parser.add_argument("--hidden_layer", default=16, type=int, help="Hidden layer size")
parser.add_argument("--alpha", default=0.0005, type=float, help="Alpha for L2 regularization")
parser.add_argument("--activation", default="relu", type=str, help="Activation function")
def convert_dfficulty_class(name):
    if name == "Easy":
        return 0
    if name == "Medium":
        return 1
    if name == "Hard":
        return 2


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:

    np.random.seed(args.seed)
    

    # 1. Load the dataset.
    with open('leetcode_problems_dataset.json', 'r') as f:
        problems = json.load(f)
        
    # 2. Remove HTML tags.
    problems_without_html = []
    for problem_name, problem_data in tqdm(problems.items()):
        if not problem_data["content"]:
            continue
        
        problems_without_html.append((BeautifulSoup(problem_data["content"], "html.parser").get_text(), problem_data["difficulty"]))
        
    # 3. Create X, y
    X, y = [], []
    difficulties_int = { "Easy": 0, "Medium": 1, "Hard": 2 }
    for problem_description, difficulty in problems_without_html:
        X.append(problem_description)
        y.append(difficulties_int[difficulty])
        
    # 4. Prepare downsampling function.
    def downsample_dataset(features, labels):
        """
        Downsamples the dataset to have an equal distribution of classes.

        Parameters:
        X (list): Feature data.
        y (list): Corresponding labels.

        Returns:
        tuple: Downsampled feature data and labels.
        """
        paired_data = list(zip(features, labels))
        class_distribution = Counter(labels)

        min_samples = min(class_distribution.values())

        downsampled_data = []
        class_counts = {cls: 0 for cls in class_distribution.keys()}

        for data, label in paired_data:
            if class_counts[label] < min_samples:
                downsampled_data.append((data, label))
                class_counts[label] += 1

        features_downsampled, labels_downsampled = zip(*downsampled_data)

        return list(features_downsampled), list(labels_downsampled)

    # 5. Downsample dataset.
    X, y = downsample_dataset(X, y)

    # 6. (Optional)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=args.seed, stratify=y_val)

    X_train, X_val, X_test, y_train, y_val, y_test = np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)

    # combine train and val
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    model = sklearn.pipeline.Pipeline([
        ("feature_extraction",
            sklearn.pipeline.FeatureUnion(([
                ("word_level", sklearn.feature_extraction.text.TfidfVectorizer(
                    lowercase=args.w_l, analyzer="word", ngram_range=(1, args.w_n),
                    binary=args.w_tf == "binary", sublinear_tf=args.w_tf == "log", max_features=args.w_mf)),
            ] if args.w_n else []) + ([
                ("char_level", sklearn.feature_extraction.text.TfidfVectorizer(
                    lowercase=args.c_l, analyzer="char_wb" if args.c_wb else "char", ngram_range=(1, args.c_n),
                    binary=args.c_tf == "binary", sublinear_tf=args.c_tf == "log", max_features=args.c_mf)),
            ] if args.c_n else []))),
        ("estimator", {
            "perceptron": sklearn.linear_model.Perceptron(tol=1e-6, early_stopping=True, validation_fraction=0.1, verbose=1, penalty="l2", random_state=args.seed),
            "mlp_c": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layer, max_iter=100, verbose=1, alpha=args.alpha, early_stopping=True, activation=args.activation),
            "mlp_r": sklearn.neural_network.MLPRegressor(hidden_layer_sizes=args.hidden_layer, max_iter=100, verbose=1, alpha=args.alpha, early_stopping=True, activation=args.activation),
            "svm": sklearn.svm.SVC(verbose=1, random_state=args.seed),
            "lsvm": sklearn.svm.LinearSVC(verbose=1, random_state=args.seed),
        }[args.model]),
    ])

  
    model.fit(X_train, y_train)
    print('Test accuracy:')
    print(model.score(X_test, y_test))

    predictions = model.predict(X_test)

    print('Test F1 score:')
    print(f1_score(y_test, predictions, average='macro'))
    print(f1_score(y_test, predictions, average='micro'))


    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(12, 10))

    class_labels = ['Easy', 'Medium', 'Hard']
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {args.model}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{args.model}.png", dpi=300)
    plt.show()


    if args.model == "mlp_r":
        print('Test RMSE:')
        print(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

        
    if args.model == "lsvm" or args.model == "svm":
        model_name = f"{args.model}_{args.w_n}_{args.c_n}"
    else:
        model_name = f"{args.model}_{args.w_n}_{args.c_n}_{args.hidden_layer}_{args.alpha}_{args.activation}"
    # with lzma.open(f"models/{model_name}.pickle", "wb") as model_file:
    #     pickle.dump(model, model_file)
    print(f"Model saved to models/{model_name}.pickle")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
