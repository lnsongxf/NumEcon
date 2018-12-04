#!/usr/bin/env python
import distutils
import os
import setuptools
import shlex
import shutil
import subprocess
import threading
import time

from watchdog import observers
from watchdog import events


class BlackCommand(distutils.cmd.Command):
    """A custom command to format python code"""

    description = "run Black on Python source files"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        command = f"python -mblack setup.py {self.distribution.get_name()} tests"

        self.announce(f"Running command: {command}", level=distutils.log.INFO)
        subprocess.check_call(shlex.split(command))


class DocsCommand(distutils.cmd.Command):
    """A custom command to format python code"""

    description = "generate documentation"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        command = "make html"

        if os.path.exists("docs/_build") and os.path.isdir("docs/_build"):
            self.announce(f"Removing build directory", level=distutils.log.INFO)
            shutil.rmtree("docs/_build")

        self.announce(f"Running command: {command}", level=distutils.log.INFO)
        if os.name == "nt":
            subprocess.check_call(shlex.split(command), cwd="docs", shell=True)
        else:
            subprocess.check_call(shlex.split(command), cwd="docs")


class PipWatch(events.PatternMatchingEventHandler):
    def on_any_event(self, event):
        subprocess.check_call(shlex.split("pip install ."))


class WatchCommand(distutils.cmd.Command):
    """A custom command to format python code"""

    description = "watch install"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        event_handler = PipWatch("*.py")
        observer = observers.Observer()
        observer.schedule(event_handler, self.distribution.get_name(), recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            observer.stop()


class JupyterCommand(distutils.cmd.Command):
    description = "Jupyter"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        threading.Thread(
            target=lambda: subprocess.run(
                shlex.split("jupyter notebook NumEcon/Notebooks")
            )
        ).start()
        threading.Thread(
            target=lambda: subprocess.run(shlex.split("python setup.py watch"))
        ).start()


setuptools.setup(
    name="NumEcon",
    version="0.1.0",
    author="Jakob Jul Elben, Jeppe Druedahl",
    packages=setuptools.find_packages(),
    package_data={"NumEcon": ["Notebooks/*"]},
    entry_points={"console_scripts": ["NumEcon=NumEcon.__main__:cli"]},
    cmdclass={
        "black": BlackCommand,
        "docs": DocsCommand,
        "jupyter": JupyterCommand,
        "watch": WatchCommand,
    },
)
