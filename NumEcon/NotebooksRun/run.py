from distutils.dir_util import copy_tree
from pkg_resources import resource_filename
from shlex import split
from subprocess import run
from tempfile import TemporaryDirectory
import os

import webbrowser


def open_binder(path="NumEcon/Notebooks"):
    """ """
    url = "https://mybinder.org/v2/gh/NumEconCopenhagen/NumEcon/master"
    webbrowser.open_new_tab(f"{url}?filepath={path}")


def jupyter_notebook(path=None):
    """ """
    notebook_dir = resource_filename(__package__.split(".")[0], "Notebooks")
    if path:
        copy_tree(notebook_dir, path)
        run_notebook(path)
    else:
        print("No path supplied. Notebooks will be deleted on exit.")
        with TemporaryDirectory() as td:
            copy_tree(notebook_dir, td)
            run_notebook(td)


def run_notebook(path=None):
    """ """
    try:
        if path:
            run_cmd(f"python -mnotebook --notebook-dir={path}")
        else:
            run_cmd(f"python -mnotebook")
    except KeyboardInterrupt:
        pass


def run_cmd(cmd):
    """ """
    return run(split(cmd))
