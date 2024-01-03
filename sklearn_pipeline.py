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

from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--c_n", default=3, type=int, help="Character n-grams")
parser.add_argument("--c_l", default=False, action="store_true", help="Character lowercase")
parser.add_argument("--c_tf", default="binary", type=str, help="Character TF type")
parser.add_argument("--c_mf", default=None, type=int, help="Character max features")
parser.add_argument("--c_wb", default=False, action="store_true", help="Character wb")
parser.add_argument("--model", default="lsvm", type=str, help="Model type")
parser.add_argument("--w_n", default=2, type=int, help="Word n-grams")
parser.add_argument("--w_l", default=False, action="store_true", help="Word lowercase")
parser.add_argument("--w_tf", default="log", type=str, help="Word TF type")
parser.add_argument("--w_mf", default=None, type=int, help="Word max features")
parser.add_argument("--hidden_layer", default=1024, type=int, help="Hidden layer size")
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
    

    problems_without_html = []
    with open('problems_without_html.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            problems_without_html.append(row)
    
    problems = [problem[0] for problem in problems_without_html]
    difficulty_classes = [convert_dfficulty_class(problem[1]) for problem in problems_without_html]
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(problems, difficulty_classes, random_state=args.seed, stratify=difficulty_classes, test_size=0.2)

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
            "perc": sklearn.linear_model.Perceptron(tol=1e-6, early_stopping=True, validation_fraction=0.1, verbose=1, penalty="l2", random_state=args.seed),
            "mlp_c": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layer, max_iter=7, verbose=1, alpha=args.alpha, early_stopping=True, activation=args.activation),
            "mlp_r": sklearn.neural_network.MLPRegressor(hidden_layer_sizes=args.hidden_layer, max_iter=100, verbose=1, alpha=args.alpha, early_stopping=True, activation=args.activation),
            "svm": sklearn.svm.SVC(verbose=1, random_state=args.seed),
            "lsvm": sklearn.svm.LinearSVC(verbose=1, random_state=args.seed),
        }[args.model]),
    ])

  
    model.fit(X_train, y_train)
    print('Test accuracy:')
    print(model.score(X_test, y_test))

    print('Test F1 score:')
    print(f1_score(y_test, model.predict(X_test), average='macro'))
    print(f1_score(y_test, model.predict(X_test), average='micro'))


    if args.model == "mlp_r":
        print('Test RMSE:')
        print(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

        
    if args.model == "lsvm" or args.model == "svm":
        model_name = f"{args.model}_{args.w_n}_{args.c_n}"
    else:
        model_name = f"{args.model}_{args.w_n}_{args.c_n}_{args.hidden_layer}_{args.alpha}_{args.activation}"
    with lzma.open(f"models/{model_name}.pickle", "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to models/{model_name}.pickle")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
