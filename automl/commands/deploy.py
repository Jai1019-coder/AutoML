import os
import typer
import joblib

app = typer.Typer(help="Deploy model to Groq Cloud")

@app.command("run")
def run(
    model_path: str = typer.Option(..., help="Path to saved model (.pkl)"),
    name:       str = typer.Option("automl‑model", help="Deployment name"),
):
    """
    Deploy the pickle to Groq Cloud via its API.
    """
    # read the pickle bytes
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    # Option A: Official groq client
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.models.deployments.create(
            name=name,
            model_binary=model_bytes,
            runtime="sklearn"
        )
        typer.secho(f"🚀 Deployed via groq-client: {resp}", fg=typer.colors.GREEN)
        return
    except ImportError:
        typer.secho("⚠️ `groq` client not installed, falling back to HTTP", fg=typer.colors.YELLOW)

    # Option B: Raw HTTP
    import requests
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise typer.Exit("❌ Please set your GROQ_API_KEY environment variable")

    url = "https://api.groqcloud.com/v1/deploy"  # replace with the real endpoint
    headers = {"Authorization": f"Bearer {api_key}"}
    files   = {"model": ("model.pkl", model_bytes)}
    data    = {"name": name, "runtime": "sklearn"}

    resp = requests.post(url, headers=headers, data=data, files=files)
    if resp.ok:
        typer.secho(f"🚀 Deployment succeeded: {resp.json()}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"❌ Deployment failed [{resp.status_code}]: {resp.text}", fg=typer.colors.RED)
