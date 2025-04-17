import typer
import joblib
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score
)

app = typer.Typer(help="Evaluate a saved model")

@app.command("run")
def run(
    model_path: str = typer.Option(..., help="Path to saved model (.pkl)")
):
    """
    Load a model + test data from <model_path> and print metrics.
    """
    data = joblib.load(model_path)
    model  = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    task   = data.get("task", "regression")

    preds = model.predict(X_test)

    if task == "regression":
        mse = mean_squared_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        typer.secho(f"📊 MSE: {mse:.4f}", fg=typer.colors.CYAN)
        typer.secho(f"📊 R²:  {r2:.4f}", fg=typer.colors.CYAN)
    else:
        # Ensure binary classes
        if preds.ndim == 1 and preds.dtype != int:
            y_pred = (preds > 0.5).astype(int)
        else:
            y_pred = preds
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        typer.secho(f"📊 Accuracy:  {acc:.4f}",  fg=typer.colors.MAGENTA)
        typer.secho(f"📊 Precision: {prec:.4f}", fg=typer.colors.MAGENTA)
        typer.secho(f"📊 Recall:    {rec:.4f}", fg=typer.colors.MAGENTA)
