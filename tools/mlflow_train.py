import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve

MLFLOW_TRACKING_URI = "sqlite:///mlflow/data/mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["USER"] = "Nokish"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def parse_version(path_models: Path, model_prefix: str) -> str:
    path_models.mkdir(parents=True, exist_ok=True)
    existing_models = [
        f.name
        for f in path_models.glob(f"{model_prefix}_v*.onnx")
        if f.is_file() and "_v" in f.name
    ]
    if not existing_models:
        return "v1"
    versions = [
        int(f.split("_v")[1].split(".onnx")[0])
        for f in existing_models
        if f.split("_v")[1].split(".onnx")[0].isdigit()
    ]
    return f"v{max(versions) + 1}" if versions else "v1"


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    prefix: str = "val",
) -> dict[str, float]:
    metrics = {
        f"{prefix}/accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}/precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}/recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}/f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    if y_proba is not None:
        try:
            prob_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics[f"{prefix}/roc_auc"] = roc_auc_score(y_true, prob_scores)
        except Exception:
            pass

    return metrics


def export_to_onnx(
    model: Any,
    X_sample: pd.DataFrame,
    output_path: str,
    model_name: str = "",
    lc_flag=True,
):

    if "catboost" in model_name.lower():
        mlflow.catboost.log_model(cb_model=model, artifact_path="model")
        if lc_flag:
            model.save_model(output_path, format="onnx")
        return

    if hasattr(model, "predict") and not isinstance(model, torch.nn.Module):
        try:
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        except Exception:
            pass

    if hasattr(model, "save_model"):
        model.save_model(output_path, format="onnx")
        return

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def log_feature_importances(
    model: Any, feature_names: list, model_dir: Path, version: str
):
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is not None and len(importances) == len(feature_names):
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values(by="importance", ascending=False)

        fi_path = model_dir / f"feature_importance_{version}.csv"
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), artifact_path="feature_importance")


def train_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    experiment_name: str = "smartphone-addiction-classification",
    lc_flag=True,
):
    model_dir = Path("./models_mlflow") / model_name
    version = parse_version(model_dir, model_name)
    run_name = f"{model_name}_{version}"

    mlflow.autolog(disable=True)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))

        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            model.fit(X_train, y_train)

        datasets = [("train", X_train, y_train), ("val", X_val, y_val)]

        if lc_flag:
            train_sizes, train_scores, val_scores = learning_curve(
                model,
                X_train,
                y_train,
                cv=3,
                scoring="f1_weighted",
                train_sizes=np.linspace(0.2, 1.0, 5),
                # n_jobs=-1,
            )

            for size, t_score, v_score in zip(
                train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)
            ):
                mlflow.log_metrics(
                    {
                        "curve/train_f1": t_score,
                        "curve/val_f1": v_score,
                    },
                    step=int(size),
                )

        for prefix, X, y in datasets:
            y_pred = model.predict(X)
            y_proba = (
                model.predict_proba(X) if hasattr(model, "predict_proba") else None
            )
            metrics = calculate_binary_metrics(y, y_pred, y_proba, prefix=prefix)
            mlflow.log_metrics(metrics)

        feature_names = (
            list(X_train.columns)
            if isinstance(X_train, pd.DataFrame)
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )
        log_feature_importances(model, feature_names, model_dir, version)

        onnx_file_path = model_dir / f"{model_name}_{version}.onnx"
        export_to_onnx(
            model, X_train, str(onnx_file_path), model_name=model_name, lc_flag=lc_flag
        )

        if onnx_file_path.exists():
            onnx_proto = onnx.load(str(onnx_file_path))
            mlflow.onnx.log_model(onnx_model=onnx_proto, artifact_path="onnx_model")


def train_nn_model(
    model_name: str,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    epochs: int,
    learning_rate: float = 0.001,
    experiment_name: str = "smartphone-addiction-classification",
):
    mlflow.set_experiment(experiment_name)

    model_dir = Path("./models_mlflow") / model_name
    version = parse_version(model_dir, model_name)

    run_name = f"{model_name}_{version}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)

        result = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            log_to_mlflow=True,
        )

        model.eval()
        onnx_file_path = model_dir / f"{model_name}_{version}.onnx"

        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

        X_sample, _ = next(iter(train_loader))

        torch.onnx.export(
            model,
            X_sample,
            str(onnx_file_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            dynamo=False,
        )

        if onnx_file_path.exists():
            try:
                onnx_proto = onnx.load(str(onnx_file_path))
                mlflow.onnx.log_model(onnx_model=onnx_proto, artifact_path="onnx_model")
            except Exception:
                mlflow.log_artifact(str(onnx_file_path), artifact_path="onnx_model")

    return result
