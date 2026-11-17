"""Command line entry point: ``python -m fae bc ...``.

Only the modelling layer is imported, so this works on a headless Linux
server without PySide6, pyradiomics or an X display.
"""

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from BC.DataContainer.DataContainer import DataContainer
from BC.DataContainer.DataSeparate import DataSeparate
from BC.FeatureAnalysis.Classifier import (
    AE,
    LDA,
    LR,
    SVM,
    AdaBoost,
    DecisionTree,
    GaussianProcess,
    LRLasso,
    NaiveBayes,
    RandomForest,
)
from BC.FeatureAnalysis.CrossValidation import (
    ArbitratyCrossValidation,
    GetMaximumCvParts,
)
from BC.FeatureAnalysis.DataBalance import (
    DownSampling,
    NoneBalance,
    SmoteSampling,
    UpSampling,
)
from BC.FeatureAnalysis.DimensionReduction import (
    DimensionReductionByPCA,
    DimensionReductionByPCC,
)
from BC.FeatureAnalysis.FeatureSelector import (
    FeatureSelectByANOVA,
    FeatureSelectByKruskalWallis,
    FeatureSelectByRFE,
)
from BC.FeatureAnalysis.Normalizer import (
    NormalizerMean,
    NormalizerMinMax,
    NormalizerNone,
    NormalizerZscore,
)
from BC.FeatureAnalysis.Pipelines import PipelinesManager
from BC.HyperParamManager.HyperParamManager import (
    GetClassifierHyperParams,
    RandomSeed,
)
from BC.Utility.Constants import (
    ACC,
    AUC,
    BALANCE_DOWN_SAMPLING,
    BALANCE_SMOTE,
    BALANCE_UP_SAMPLING,
    CLASSIFIER_AB,
    CLASSIFIER_AE,
    CLASSIFIER_DT,
    CLASSIFIER_GP,
    CLASSIFIER_LR,
    CLASSIFIER_LRLasso,
    CLASSIFIER_RF,
    CLASSIFIER_SVM,
    MCC,
    SEN,
    SPE,
)
from Utility.PathUtils import get_resource_path

from fae import __version__

BALANCE_CHOICES = ("none", "up", "down", "smote")
NORMALIZER_CHOICES = ("none", "zscore", "minmax", "mean")
REDUCER_CHOICES = ("none", "pcc", "pca")
SELECTOR_CHOICES = ("none", "anova", "kw", "rfe")
CLASSIFIER_CHOICES = ("svm", "lr", "lasso", "lda", "rf", "dt", "ab", "nb", "gp", "ae")


class KeepAll:
    """Pass-through processor used when no reduction/selection is requested."""

    def __init__(self, name="None"):
        self._name = name

    def GetName(self):
        return self._name

    def Run(self, container, *args, **kwargs):
        return container

    def Transform(self, container, *args, **kwargs):
        return container

    def SetSelectedFeatureNumber(self, number):
        pass


def _load_random_seeds():
    config_path = get_resource_path("BC", "HyperParameters", "RandomSeed.json")
    return RandomSeed(config_path).random_seed


def build_balance(name, seeds):
    if name == "none":
        return NoneBalance()
    if name == "up":
        return UpSampling(random_state=seeds[BALANCE_UP_SAMPLING])
    if name == "down":
        return DownSampling(random_state=seeds[BALANCE_DOWN_SAMPLING])
    if name == "smote":
        return SmoteSampling(random_state=seeds[BALANCE_SMOTE])
    raise ValueError("Unknown balance method: {}".format(name))


def build_normalizer(name):
    return {
        "none": lambda: deepcopy(NormalizerNone),
        "zscore": lambda: deepcopy(NormalizerZscore),
        "minmax": lambda: deepcopy(NormalizerMinMax),
        "mean": lambda: deepcopy(NormalizerMean),
    }[name]()


def build_reducer(name):
    return {
        "none": lambda: KeepAll("None"),
        "pcc": lambda: DimensionReductionByPCC(),
        "pca": lambda: DimensionReductionByPCA(),
    }[name]()


def build_selector(name):
    return {
        "none": lambda: KeepAll("None"),
        "anova": lambda: FeatureSelectByANOVA(),
        "kw": lambda: FeatureSelectByKruskalWallis(),
        "rfe": lambda: FeatureSelectByRFE(),
    }[name]()


def build_classifier(name, seeds):
    if name == "svm":
        return SVM()
    if name == "lr":
        return LR(random_state=seeds[CLASSIFIER_LR])
    if name == "lasso":
        return LRLasso(random_state=seeds[CLASSIFIER_LRLasso])
    if name == "lda":
        return LDA()
    if name == "rf":
        return RandomForest(random_state=seeds[CLASSIFIER_RF])
    if name == "dt":
        return DecisionTree(random_state=seeds[CLASSIFIER_DT])
    if name == "ab":
        return AdaBoost(random_state=seeds[CLASSIFIER_AB])
    if name == "nb":
        return NaiveBayes()
    if name == "gp":
        return GaussianProcess(random_state=seeds[CLASSIFIER_GP])
    if name == "ae":
        return AE(random_state=seeds[CLASSIFIER_AE])
    raise ValueError("Unknown classifier: {}".format(name))


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def _ratio(value):
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return ratio


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m fae",
        description="FeAture Explorer headless command line.",
    )
    parser.add_argument("--version", action="version", version="FAE %(prog)s")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bc = subparsers.add_parser(
        "bc",
        help="run the binary classification pipeline on a feature CSV",
        description="Run the BC pipeline: split, normalize, reduce, select, "
        "classify and cross-validate on a numeric feature matrix.",
    )
    bc.add_argument("--data", required=True, help="feature CSV with a label column")
    split = bc.add_mutually_exclusive_group()
    split.add_argument(
        "--test-ratio",
        type=_ratio,
        default=0.3,
        help="test set ratio when splitting randomly (default: 0.3)",
    )
    split.add_argument(
        "--test-data",
        help="CSV with the cases that should be used as the test set",
    )
    bc.add_argument("--seed", type=int, default=2026, help="random seed for the split")
    bc.add_argument("--balance", choices=BALANCE_CHOICES, default="none")
    bc.add_argument(
        "--normalizer",
        choices=NORMALIZER_CHOICES,
        nargs="+",
        default=["zscore"],
        metavar="NAME",
        help="one or more normalizers (default: zscore)",
    )
    bc.add_argument(
        "--reducer",
        choices=REDUCER_CHOICES,
        nargs="+",
        default=["none"],
        metavar="NAME",
        help="one or more dimension reducers (default: none)",
    )
    bc.add_argument(
        "--selector",
        choices=SELECTOR_CHOICES,
        nargs="+",
        default=["none"],
        metavar="NAME",
        help="one or more feature selectors (default: none)",
    )
    bc.add_argument(
        "--feature-numbers",
        type=_positive_int,
        nargs="+",
        default=[1],
        help="feature counts to evaluate, e.g. --feature-numbers 5 10 20",
    )
    bc.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        nargs="+",
        default=["lr"],
        metavar="NAME",
        help="one or more classifiers (default: lr)",
    )
    bc.add_argument("--cv", type=_positive_int, default=5, help="cross-validation folds")
    bc.add_argument(
        "--hyper-param-dir",
        help="classifier hyper-parameter folder (default: BC/HyperParameters/Classifier)",
    )
    bc.add_argument(
        "--no-hyper-param",
        action="store_true",
        help="skip the nested hyper-parameter search",
    )
    bc.add_argument(
        "--output",
        help="result folder (default: fae_runs/<data name>_<timestamp>)",
    )
    bc.add_argument("--quiet", action="store_true", help="hide progress lines")
    bc.add_argument(
        "--top",
        type=_positive_int,
        help="print only the first N rows of the summary (summary.csv keeps all)",
    )
    bc.add_argument(
        "--debug", action="store_true", help="print the full traceback on error"
    )

    info = subparsers.add_parser(
        "info", help="show versions and the available pipeline components"
    )
    info.set_defaults()

    return parser


def _default_output(data_path):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "fae_runs" / "{}_{}".format(data_path.stem, stamp)


def _unique(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _write_summary(output):
    """Join CV and test metrics of every pipeline into one sorted table."""
    cv_path = output / "cv_val_results.csv"
    test_path = output / "test_results.csv"
    if not (cv_path.is_file() and test_path.is_file()):
        return None

    metrics = [AUC, ACC, SEN, SPE, MCC]
    cv_frame = pd.read_csv(cv_path, index_col=0)
    test_frame = pd.read_csv(test_path, index_col=0)
    cv_frame = cv_frame[[name for name in metrics if name in cv_frame.columns]]
    test_frame = test_frame[[name for name in metrics if name in test_frame.columns]]
    summary = cv_frame.add_prefix("CV_").join(
        test_frame.add_prefix("TEST_"), how="outer"
    )
    sort_column = "TEST_{}".format(AUC)
    if sort_column in summary.columns:
        summary = summary.sort_values(sort_column, ascending=False)
    summary.index.name = "Model"
    summary.to_csv(output / "summary.csv", encoding="utf-8")
    return summary


def run_bc(args):
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError("feature file not found: {}".format(data_path))

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output(data_path)
    )

    data_container = DataContainer()
    if not data_container.Load(str(data_path)):
        raise ValueError(
            "cannot load {}: expected a CSV with a 'label' column".format(data_path)
        )
    if data_container.IsEmpty():
        raise ValueError("empty feature matrix: {}".format(data_path))
    if not data_container.IsBinaryLabel():
        raise ValueError("the label column must contain exactly two classes")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))

    separator = DataSeparate()
    if args.test_data:
        test_path = Path(args.test_data).expanduser().resolve()
        reference = DataContainer()
        if not reference.Load(str(test_path)):
            raise ValueError("cannot load test data: {}".format(test_path))
        train, test = separator.RunByTestingReference(data_container, reference)
        split_note = "reference test set {}".format(test_path.name)
    else:
        train, test = separator.RunByTestingPercentage(
            data_container, args.test_ratio
        )
        split_note = "ratio {:.2f}".format(args.test_ratio)

    if train.IsEmpty() or test.IsEmpty():
        raise ValueError("the split produced an empty train or test set")
    max_folds = GetMaximumCvParts(train.GetLabel())
    if args.cv > max_folds:
        raise ValueError(
            "--cv {} exceeds the smallest class size ({}) in the training set".format(
                args.cv, max_folds
            )
        )

    seeds = _load_random_seeds()
    hyper_params = (
        {}
        if args.no_hyper_param
        else GetClassifierHyperParams(args.hyper_param_dir)
    )

    normalizers = _unique(args.normalizer)
    reducers = _unique(args.reducer)
    selectors = _unique(args.selector)
    feature_numbers = _unique(args.feature_numbers)
    classifiers = _unique(args.classifier)

    if len(feature_numbers) > 1 and all(name == "none" for name in selectors):
        print(
            "[FAE] --selector none ignores --feature-numbers; using {}".format(
                feature_numbers[0]
            )
        )
        feature_numbers = [feature_numbers[0]]

    grid_size = (
        len(normalizers)
        * len(reducers)
        * len(selectors)
        * len(feature_numbers)
        * len(classifiers)
    )

    manager = PipelinesManager(
        balancer=build_balance(args.balance, seeds),
        normalizer_list=[build_normalizer(name) for name in normalizers],
        dimension_reduction_list=[build_reducer(name) for name in reducers],
        feature_selector_list=[build_selector(name) for name in selectors],
        feature_selector_num_list=feature_numbers,
        classifier_list=[build_classifier(name, seeds) for name in classifiers],
        cv=ArbitratyCrossValidation(args.cv),
        hyper_param=hyper_params,
        random_seed=seeds,
    )

    output.mkdir(parents=True, exist_ok=True)
    print(
        "[FAE] {} cases x {} features | train {} / test {} ({})".format(
            data_container.GetArray().shape[0],
            data_container.GetArray().shape[1],
            train.GetArray().shape[0],
            test.GetArray().shape[0],
            split_note,
        )
    )
    print(
        "[FAE] grid: {} normalizer x {} reducer x {} selector x {} feature number x "
        "{} classifier = {} models".format(
            len(normalizers),
            len(reducers),
            len(selectors),
            len(feature_numbers),
            len(classifiers),
            grid_size,
        )
    )

    started = time.time()
    print_every = 1 if grid_size <= 20 else max(1, grid_size // 20)
    for total, done in manager.Run(train, test, store_folder=str(output)):
        if not args.quiet and (done == total or done % print_every == 0):
            print("[FAE] pipeline {}/{}".format(done, total), flush=True)

    elapsed = time.time() - started
    print("[FAE] finished in {:.1f}s".format(elapsed))

    summary = _write_summary(output)
    if summary is not None and not summary.empty:
        shown = summary if args.top is None else summary.head(args.top)
        print("\n[SUMMARY] sorted by test AUC")
        print(shown.reset_index().to_string(index=False))
        if args.top is not None and len(summary) > len(shown):
            print(
                "[FAE] showing top {} of {} models; see summary.csv for the rest".format(
                    len(shown), len(summary)
                )
            )

    config = {
        "fae_version": __version__,
        "command": "bc",
        "data": str(data_path),
        "test_data": str(Path(args.test_data).expanduser().resolve())
        if args.test_data
        else None,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "balance": args.balance,
        "normalizers": normalizers,
        "reducers": reducers,
        "selectors": selectors,
        "feature_numbers": feature_numbers,
        "classifiers": classifiers,
        "grid_size": grid_size,
        "cv": args.cv,
        "hyper_param_dir": str(Path(args.hyper_param_dir).expanduser().resolve())
        if args.hyper_param_dir
        else None,
        "no_hyper_param": args.no_hyper_param,
        "train_cases": train.GetArray().shape[0],
        "test_cases": test.GetArray().shape[0],
        "features": data_container.GetArray().shape[1],
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(output / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)

    print("[FAE] results written to {}".format(output))
    return 0


def run_info():
    from importlib import metadata

    print("FAE        : {}".format(__version__))
    print("Python     : {}".format(sys.version.split()[0]))
    print("Platform   : {}".format(sys.platform))
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "shap",
        "lifelines",
        "scikit-survival",
        "matplotlib",
    ):
        try:
            print("{:<11}: {}".format(package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            print("{:<11}: not installed".format(package))

    print("\nbalance    : {}".format(", ".join(BALANCE_CHOICES)))
    print("normalizer : {}".format(", ".join(NORMALIZER_CHOICES)))
    print("reducer    : {}".format(", ".join(REDUCER_CHOICES)))
    print("selector   : {}".format(", ".join(SELECTOR_CHOICES)))
    print("classifier : {}".format(", ".join(CLASSIFIER_CHOICES)))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "bc":
            return run_bc(args)
        if args.command == "info":
            return run_info()
        parser.error("unknown command: {}".format(args.command))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary
        if getattr(args, "debug", False):
            raise
        print("error: {}".format(error), file=sys.stderr)
        return 1
    return 0
