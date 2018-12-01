import click
from .NotebooksRun import run


@click.group()
def cli():
    pass


@cli.command()
@click.option("-b", "--binder", is_flag=True)
@click.option("-p", "--path", type=click.Path(exists=True))
@click.option("-t", "--temp", is_flag=True)
@click.pass_context
def notebook(ctx, binder, path, temp):
    if not binder or path or temp:
        click.echo(ctx.get_help())
        
    if binder:
        run.open_binder()

    if path:
        run.jupyter_notebook(path)

    if temp:
        run.jupyter_notebook()


if __name__ == "__main__":
    cli()
