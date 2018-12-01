#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="NumEcon",
    version="0.1.0",
    author="Jakob Jul Elben, Jeppe Druedahl",
    packages=find_packages(),
    package_data={
        'NumEcon': ['Notebooks/*']
    },
    entry_points={
        'console_scripts': [
            'NumEcon=NumEcon.__main__:cli'
        ]
    }
)