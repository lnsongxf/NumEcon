#!/usr/bin/env python
import distutils
import os
import setuptools
import shlex
import shutil
import subprocess


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
        command = "python -mblack NumEcon tests"

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
        subprocess.check_call(shlex.split(command), cwd="docs")


setuptools.setup(
    name="NumEcon",
    version="0.1.0",
    author="Jakob Jul Elben, Jeppe Druedahl",
    packages=setuptools.find_packages(),
    package_data={"NumEcon": ["Notebooks/*"]},
    entry_points={"console_scripts": ["NumEcon=NumEcon.__main__:cli"]},
    cmdclass={"black": BlackCommand, "docs": DocsCommand},
)
