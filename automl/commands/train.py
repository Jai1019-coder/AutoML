import typer
import pandas as pd
import joblib
from typing import Literal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

app = typer.Typer(help="Train a model (regression or classification)")

@app.command("run")
def run(
    data_path: str = typer.Option(..., help="Path to CSV dataset"),
    target: str    = typer.Option(..., help="Name of target column"),
    task: Literal["regression", "classification"] = typer.Option(
        ..., "--task", "-t",
        help="Task type: 'regression' for LinearRegression, 'classification' for LogisticRegression"
    ),
    test_size: float = typer.Option(0.2, help="Fraction for test split"),
    model_out: str   = typer.Option("model.pkl", help="Where to save the trained model"),
):
    """
    Train either a LinearRegression or LogisticRegression on <data_path>.
    """
    # 1 Load and split
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 2 Choose model
    if task == "regression":
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=1000)

    # 3 Train
    model.fit(X_train, y_train)

    # 4 Persist both model + test‑set
    payload = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "task": task
    }
    joblib.dump(payload, model_out)

    typer.secho(f"Trained {task.title()} model and saved to {model_out}", fg=typer.colors.GREEN)
