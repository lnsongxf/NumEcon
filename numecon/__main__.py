import click
from .NotebooksRun import run as _run


@click.group()
def cli():
    pass


@cli.group("Notebook")
def notebook():
    """Run numecons notebooks locally and with mybinder.org"""
    pass


@notebook.command("MyBinder")
@click.option(
    "-i",
    "--ipynb",
    type=click.Path(),
    help="Path to ipython notebook within the Notebook directory",
)
def mybinder(ipynb=None):
    """Run numecons notebook on mybinder.org.
    
    Your default browser will try to open a new tab or a new window. 
    """
    _run.open_binder(ipynb=ipynb)


@notebook.command("Run")
@click.option(
    "-i",
    "--ipynb",
    type=click.Path(),
    help="Path to ipython notebook within the Notebook directory",
)
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=True),
    help="Will copy all numecons notebooks to this path, and open a jupyter notebook at the location",
)
def run(ipynb=None, path=None):
    """Run jupyter notebook locally.
    
    Your default browser will try to open a new tab or a new window."""
    _run.jupyter_notebook(path, ipynb)


@notebook.command("List")
def list_notebooks():
    """Lists all available notebooks."""
    click.echo("The following notebooks are available:")
    for i in _run.notebooks_list():
        click.echo(i)


if __name__ == "__main__":
    cli()
