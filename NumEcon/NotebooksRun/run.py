"""
Run NumEcon's notebooks locally and remotely.
"""
from distutils.dir_util import copy_tree
from pkg_resources import resource_filename
from shlex import split
from subprocess import run
from tempfile import TemporaryDirectory
import os

import webbrowser


def open_binder(path="NumEcon/Notebooks", ipynb=None):
    """Open NumEcons on mybinder.org using default browser.
    
    args:
        path(:obj:`str`): path to where the notebook should be opened.
        ipynb(:obj:`str`): if specified mybinder will open the specified notebook.
            Should be relative to `path`
    """
    url = f"https://mybinder.org/v2/gh/NumEconCopenhagen/NumEcon/master?filepath={path}"
    if ipynb:
        url = os.path.join(url, ipynb)
    webbrowser.open_new_tab(url)


def jupyter_notebook(path=None, ipynb=None):
    """ """
    notebook_dir = resource_filename(__package__.split(".")[0], "Notebooks")
    if ipynb:
        with TemporaryDirectory() as td:
            copy_tree(notebook_dir, td)
            run_notebook(os.path.join(td, ipynb))
    else:
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
            run_cmd(f"python -mnotebook {path}")
        else:
            run_cmd(f"python -mnotebook")
    except KeyboardInterrupt:
        pass


def run_cmd(cmd):
    """Run system command.
    
    args:
        cmd(:obj:`str`): Runs as system command using subprocess.
    """
    return run(split(cmd))


def notebooks_list(path=None):
    if not path:
        path = resource_filename(__package__.split(".")[0], "Notebooks")
    return os.listdir(path)
