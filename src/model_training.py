# model_training.py
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split dataset into training and testing sets.
    """
    if X.ndim == 1 or isinstance(X, pd.Series):
        X = X.to_frame()  # ensure X is 2D

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# =====================================================
# 2️⃣ Train a Single Model with Optional Hyperparameter Tuning
# =====================================================
def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any] = None,
    search_type: str = "grid",
    cv: int = 5,
    random_state: int = 42
):
    """
    Train a model with optional hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    Returns best model and best parameters.
    """
    # Select model & default hyperparameters
    if model_name == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        default_param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=random_state)
        default_param_grid = {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]}
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
        default_param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state)
        default_param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if param_grid is None:
        param_grid = default_param_grid

    # Choose search type
    if search_type == "grid":
        search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring="roc_auc")
    elif search_type == "random":
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring="roc_auc", n_iter=10, random_state=random_state)
    else:
        raise ValueError(f"search_type must be 'grid' or 'random', got {search_type}")

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# =====================================================
# 3️⃣ Evaluate Model
# =====================================================
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    return metrics


# =====================================================
# 4️⃣ MLflow Logging
# =====================================================
def log_experiment(
    model_name: str,
    model,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    experiment_name: str = "credit_risk_model"
):
    """
    Log model parameters, metrics, and the trained model to MLflow.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"✅ Logged {model_name} to MLflow")


# =====================================================
# 5️⃣ End-to-End Training & Tracking
# =====================================================
def train_and_track(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: List[str] = ["logistic", "random_forest"],
    search_type: str = "grid"
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models, evaluate, and log experiments in MLflow.
    Returns dictionary of results.
    """
    results = {}
    for model_name in models:
        print(f"Training {model_name}...")
        model, best_params = train_model(model_name, X_train, y_train, search_type=search_type)
        metrics = evaluate_model(model, X_test, y_test)
        log_experiment(model_name, model, best_params, metrics)
        results[model_name] = {"model": model, "params": best_params, "metrics": metrics}
        print(f"{model_name} metrics:", metrics)
    return results
