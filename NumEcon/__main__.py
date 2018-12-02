import click
from .NotebooksRun import run as _run


@click.group()
def cli():
    pass


@cli.group("Notebook")
def notebook():
    pass


@notebook.command("MyBinder")
@click.option("-i", "--ipynb", type=click.Path())
def mybinder(ipynb=None):
    _run.open_binder(ipynb=ipynb)


@notebook.command("Run")
@click.option("-i", "--ipynb", type=click.Path())
@click.option("-p", "--path", type=click.Path(exists=True))
def run(ipynb=None, path=None):
    _run.jupyter_notebook(path, ipynb)


# @cli.command()
# @click.option("-b", "--binder", is_flag=True)
# @click.option("-p", "--path", type=click.Path(exists=True))
# @click.option("-t", "--temp", is_flag=True)
# @click.option("-f", "--file", type=click.Path(exists=True))
# @click.pass_context
# def notebook(ctx, binder, path, temp):
#     if not binder or path or temp:
#         click.echo(ctx.get_help())

#     if binder:
#         run.open_binder()

#     if path:
#         run.jupyter_notebook(path)

#     if temp:
#         run.jupyter_notebook()


if __name__ == "__main__":
    cli()
