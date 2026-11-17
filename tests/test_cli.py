import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fae.cli import main


def write_feature_csv(path: Path, case_count: int = 60, feature_count: int = 6) -> Path:
    rng = np.random.default_rng(11)
    labels = np.tile(np.array([0, 1]), case_count // 2)
    features = rng.normal(size=(case_count, feature_count))
    features[:, 0] += labels * 2.5
    frame = pd.DataFrame(
        features,
        index=["case_{:03d}".format(index) for index in range(case_count)],
        columns=["feature_{}".format(index) for index in range(feature_count)],
    )
    frame.insert(0, "label", labels)
    frame.to_csv(path, encoding="utf-8")
    return path


def test_info_command(capsys):
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "FAE" in output
    assert "classifier" in output


def test_bc_runs_end_to_end(tmp_path):
    data = write_feature_csv(tmp_path / "features.csv")
    output = tmp_path / "results"

    code = main(
        [
            "bc",
            "--data",
            str(data),
            "--output",
            str(output),
            "--cv",
            "3",
            "--seed",
            "7",
            "--feature-numbers",
            "1",
            "--classifier",
            "lr",
            "--no-hyper-param",
        ]
    )

    assert code == 0
    for name in (
        "cv_val_results.csv",
        "test_results.csv",
        "train_results.csv",
        "pipeline_info.csv",
        "run_config.json",
    ):
        assert (output / name).is_file(), name

    cv_results = pd.read_csv(output / "cv_val_results.csv", encoding="utf-8")
    assert 0.0 <= cv_results["AUC"].iloc[0] <= 1.0

    with open(output / "run_config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    assert config["seed"] == 7
    assert config["cv"] == 3
    assert config["classifiers"] == ["lr"]
    assert config["grid_size"] == 1
    assert config["train_cases"] + config["test_cases"] == 60

    summary = pd.read_csv(output / "summary.csv", encoding="utf-8")
    assert len(summary) == 1
    assert {"Model", "CV_AUC", "TEST_AUC"} <= set(summary.columns)


def test_bc_enumerates_grid(tmp_path):
    data = write_feature_csv(tmp_path / "features.csv", case_count=80)
    output = tmp_path / "grid"

    code = main(
        [
            "bc",
            "--data",
            str(data),
            "--output",
            str(output),
            "--cv",
            "3",
            "--seed",
            "7",
            "--feature-numbers",
            "1",
            "--normalizer",
            "zscore",
            "minmax",
            "--classifier",
            "lr",
            "nb",
            "--no-hyper-param",
            "--quiet",
        ]
    )

    assert code == 0
    summary = pd.read_csv(output / "summary.csv", encoding="utf-8")
    assert len(summary) == 4
    assert summary["Model"].is_unique

    with open(output / "run_config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    assert config["normalizers"] == ["zscore", "minmax"]
    assert config["classifiers"] == ["lr", "nb"]
    assert config["grid_size"] == 4


def test_bc_rejects_missing_file(tmp_path):
    code = main(["bc", "--data", str(tmp_path / "absent.csv"), "--output", str(tmp_path)])
    assert code == 1


def test_bc_rejects_unknown_classifier(tmp_path):
    data = write_feature_csv(tmp_path / "features.csv")
    with pytest.raises(SystemExit) as error:
        main(["bc", "--data", str(data), "--classifier", "nope"])
    assert error.value.code == 2
