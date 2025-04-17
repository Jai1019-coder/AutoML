import typer

from automl.commands import train, evaluate, deploy

app = typer.Typer(help="AutoML Pipeline Generator")

# register sub‑commands
app.add_typer(train.app,    name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(deploy.app,   name="deploy")

def version_callback(value: bool):
    if value:
        typer.echo("automl version 0.1.0")
        raise typer.Exit()

app.callback()(  # top‑level options
    typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
        help="Show the application version and exit."
    )
)

if __name__ == "__main__":
    app()
