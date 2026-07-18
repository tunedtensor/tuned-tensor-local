# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "joblib>=1.4,<2",
#   "numpy>=2,<3",
#   "scikit-learn>=1.7,<2",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUNNER_NAME = "numeric_logistic_regression"
RUNNER_VERSION = 1
PARAMETER_KEYS = {"c", "class_weight", "max_iter", "random_seed"}


def display_value(value: str, maximum_length: int = 96) -> str:
    if len(value) > maximum_length:
        value = value[: maximum_length - 3] + "..."
    return json.dumps(value, ensure_ascii=True)


def require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    description: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{description} is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{description} has unknown fields: {', '.join(unknown)}")


def require_integer(
    value: Any,
    description: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{description} must be between {minimum} and {maximum}")
    return value


def parse_parameters(value: Any) -> dict[str, Any]:
    parameters = require_object(value, "trial parameters")
    require_exact_keys(parameters, PARAMETER_KEYS, "trial parameters")
    c_value = parameters["c"]
    if (
        isinstance(c_value, bool)
        or not isinstance(c_value, (int, float))
        or not math.isfinite(c_value)
        or c_value < 1e-6
        or c_value > 1e6
    ):
        raise ValueError("trial parameter c must be a finite number between 1e-6 and 1e6")
    class_weight = parameters["class_weight"]
    if class_weight not in ("none", "balanced"):
        raise ValueError('trial parameter class_weight must be "none" or "balanced"')
    return {
        "c": float(c_value),
        "class_weight": class_weight,
        "max_iter": require_integer(
            parameters["max_iter"],
            "trial parameter max_iter",
            1,
            10_000,
        ),
        "random_seed": require_integer(
            parameters["random_seed"],
            "trial parameter random_seed",
            0,
            2**32 - 1,
        ),
    }


def parse_request(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        payload = require_object(json.loads(path.read_text(encoding="utf-8")), "trial input")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid trial input JSON at {path}") from error
    require_exact_keys(
        payload,
        {"protocol_version", "trial", "task", "datasets"},
        "trial input",
    )
    if (
        isinstance(payload["protocol_version"], bool)
        or payload["protocol_version"] != 1
    ):
        raise ValueError("trial input protocol_version must be 1")

    trial = require_object(payload["trial"], "trial input trial")
    require_exact_keys(trial, {"id", "name", "parameters"}, "trial input trial")
    for field in ("id", "name"):
        if not isinstance(trial[field], str) or not trial[field]:
            raise ValueError(f"trial input trial.{field} must be a non-empty string")
    parameters = parse_parameters(trial["parameters"])

    task = require_object(payload["task"], "trial input task")
    require_exact_keys(
        task,
        {
            "type",
            "id_column",
            "input_columns",
            "target_column",
            "target_values",
            "prediction",
        },
        "trial input task",
    )
    if task["type"] != "binary_classification":
        raise ValueError("built-in numeric logistic regression requires binary_classification")
    for field in ("id_column", "target_column"):
        if not isinstance(task[field], str) or not task[field]:
            raise ValueError(f"trial input task.{field} must be a non-empty string")
    input_columns = task["input_columns"]
    if (
        not isinstance(input_columns, list)
        or not input_columns
        or any(not isinstance(column, str) or not column for column in input_columns)
        or len(set(input_columns)) != len(input_columns)
    ):
        raise ValueError("trial input task.input_columns must contain unique non-empty strings")
    target_values = require_object(
        task["target_values"],
        "trial input task.target_values",
    )
    if (
        set(target_values) != {"negative", "positive"}
        or isinstance(target_values.get("negative"), bool)
        or isinstance(target_values.get("positive"), bool)
        or target_values != {"negative": 0, "positive": 1}
    ):
        raise ValueError("trial input task.target_values must encode negative=0 and positive=1")
    if task["prediction"] != {
        "field": "probability",
        "meaning": "probability_of_positive_target",
    }:
        raise ValueError("trial input task.prediction has unsupported semantics")

    datasets = require_object(payload["datasets"], "trial input datasets")
    require_exact_keys(
        datasets,
        {"training_csv", "validation_csv"},
        "trial input datasets",
    )
    if any(not isinstance(path_value, str) or not path_value for path_value in datasets.values()):
        raise ValueError("trial dataset paths must be non-empty strings")
    return task, parameters, datasets


def read_projected_csv(
    path: Path,
    expected_columns: list[str],
    split: str,
) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != expected_columns:
                raise ValueError(
                    f"{split} CSV columns must be {expected_columns}, found {reader.fieldnames}"
                )
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read {split} CSV at {path}") from error
    if not rows:
        raise ValueError(f"{split} CSV must contain at least one row")
    for row_number, row in enumerate(rows, start=1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{split} CSV row {row_number} does not match its header")
    return rows


def validate_ids(rows: list[dict[str, str]], id_column: str, split: str) -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        identifier = row[id_column]
        if not identifier:
            raise ValueError(f"{split} CSV row {row_number} has an empty ID")
        if identifier in seen:
            raise ValueError(
                f"{split} CSV has duplicate ID {display_value(identifier)}"
            )
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers


def numeric_matrix(
    rows: list[dict[str, str]],
    input_columns: list[str],
    id_column: str,
    split: str,
) -> tuple[np.ndarray, dict[str, int]]:
    values = np.empty((len(rows), len(input_columns)), dtype=np.float64)
    missing_counts = {column: 0 for column in input_columns}
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(input_columns):
            raw_value = row[column]
            if not raw_value.strip():
                values[row_index, column_index] = np.nan
                missing_counts[column] += 1
                continue
            try:
                number = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{split} ID {display_value(row[id_column])} "
                    f"column {display_value(column)} has nonnumeric value "
                    f"{display_value(raw_value)}"
                ) from error
            if not math.isfinite(number):
                raise ValueError(
                    f"{split} ID {display_value(row[id_column])} "
                    f"column {display_value(column)} "
                    "must be finite or blank"
                )
            values[row_index, column_index] = number
    return values, missing_counts


def training_targets(
    rows: list[dict[str, str]],
    target_column: str,
) -> tuple[np.ndarray, dict[str, int]]:
    targets: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        raw_target = row[target_column]
        if raw_target not in ("0", "1"):
            raise ValueError(
                f"training CSV row {row_number} target must be encoded as 0 or 1"
            )
        targets.append(int(raw_target))
    counts = {"negative": targets.count(0), "positive": targets.count(1)}
    if counts["negative"] == 0 or counts["positive"] == 0:
        raise ValueError("training CSV must contain both target classes")
    return np.asarray(targets, dtype=np.int64), counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(value, destination, indent=2, allow_nan=False)
            destination.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_model_atomic(path: Path, model: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(input_path: Path, output_path: Path, artifact_directory: Path) -> None:
    task, parameters, datasets = parse_request(input_path)
    input_columns = task["input_columns"]
    id_column = task["id_column"]
    target_column = task["target_column"]
    training = read_projected_csv(
        Path(datasets["training_csv"]),
        [id_column, *input_columns, target_column],
        "training",
    )
    validation = read_projected_csv(
        Path(datasets["validation_csv"]),
        [id_column, *input_columns],
        "validation",
    )
    validate_ids(training, id_column, "training")
    validation_ids = validate_ids(validation, id_column, "validation")
    training_values, training_missing = numeric_matrix(
        training,
        input_columns,
        id_column,
        "training",
    )
    validation_values, _ = numeric_matrix(
        validation,
        input_columns,
        id_column,
        "validation",
    )
    all_missing = [
        column
        for column, missing_count in training_missing.items()
        if missing_count == len(training)
    ]
    if all_missing:
        raise ValueError(
            "training numeric features are entirely missing: "
            + ", ".join(display_value(column) for column in all_missing)
        )
    targets, class_counts = training_targets(training, target_column)

    random.seed(parameters["random_seed"])
    np.random.seed(parameters["random_seed"])
    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=parameters["c"],
                class_weight=(
                    None
                    if parameters["class_weight"] == "none"
                    else parameters["class_weight"]
                ),
                max_iter=parameters["max_iter"],
                penalty="l2",
                random_state=parameters["random_seed"],
                solver="liblinear",
                tol=1e-4,
            ),
        ),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(training_values, targets)
        except ConvergenceWarning as error:
            raise ValueError(
                "logistic regression did not converge; increase max_iter or adjust c"
            ) from error

    classifier = pipeline.named_steps["classifier"]
    positive_matches = np.flatnonzero(classifier.classes_ == 1)
    if positive_matches.size != 1:
        raise ValueError("fitted classifier does not contain positive class 1")
    probabilities = pipeline.predict_proba(validation_values)[:, int(positive_matches[0])]
    if (
        probabilities.shape != (len(validation),)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("classifier produced invalid positive-class probabilities")

    artifact_directory.mkdir(parents=True, exist_ok=True)
    model_path = artifact_directory / "model.joblib"
    write_model_atomic(model_path, pipeline)
    indicator = pipeline.named_steps["imputer"].indicator_
    manifest = {
        "schema_version": 1,
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION},
        "parameters": parameters,
        "pipeline": {
            "imputer": {
                "strategy": "median",
                "add_indicator": True,
            },
            "scaler": "standard",
            "classifier": {
                "name": "logistic_regression",
                "solver": "liblinear",
                "penalty": "l2",
                "tolerance": 1e-4,
            },
        },
        "input_columns": input_columns,
        "training": {
            "row_count": len(training),
            "class_counts": class_counts,
            "missing_counts": training_missing,
            "transformed_feature_count": len(input_columns) + len(indicator.features_),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "model": {
            "path": "model.joblib",
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }
    write_json_atomic(artifact_directory / "runner-manifest.json", manifest)
    write_json_atomic(output_path, {
        "protocol_version": 1,
        "predictions": [
            {"id": identifier, "probability": float(probability)}
            for identifier, probability in zip(
                validation_ids,
                probabilities,
                strict=True,
            )
        ],
    })
    print(
        f"Fitted {RUNNER_NAME} on {len(training)} rows and "
        f"predicted {len(validation)} validation rows.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TT Local's bundled numeric logistic-regression Study trial.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir", required=True)
    arguments = parser.parse_args()
    try:
        run(
            Path(arguments.input),
            Path(arguments.output),
            Path(arguments.artifact_dir),
        )
    except Exception as error:
        print(f"{RUNNER_NAME}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
