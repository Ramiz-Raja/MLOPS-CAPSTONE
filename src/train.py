import os
import joblib
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from model import load_and_preprocess
import pandas as pd

def main():
    # Initialize W&B
    wandb.init(
        project="titanic",  # Your W&B project name
        config={
            "n_estimators": 100,
            "max_depth": 7,
            "test_size": 0.2,
            "random_state": 42
        }
    )

    cfg = wandb.config

    # Load data
    df = pd.read_csv("data/raw/titanic.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess("data/raw/titanic.csv")

    # Model training
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Optional: if your model supports predict_proba
    if hasattr(model, "predict_proba"):
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
    else:
        train_loss = test_loss = None

    # Log metrics to W&B
    wandb.log({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss
    })

    # ✅ Fix confusion matrix index issue
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_reset = pd.Series(y_pred_test).reset_index(drop=True)

    # Log confusion matrix (safe version)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_reset,
            preds=y_pred_reset,
            class_names=["Not Survived", "Survived"]
        )
    })

    # Save model + log artifact
    model_path = "rf_titanic.joblib"
    joblib.dump(model, model_path)
    artifact = wandb.Artifact("titanic-rf", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    print(f"✅ Training done | Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
