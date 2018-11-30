#!/usr/bin/env python
from setuptools import find_packages, setup

install_requires_list = []
setup_requires_list = ["pytest-runner"]
test_requires_list = ["pytest"]

setup(
    name="NumEcon",
    version="0.1.0",
    author="Jakob Jul Elben, Jeppe Druedahl",
    packages=find_packages(),
    install_requires=install_requires_list,
    setup_requires=setup_requires_list,
    test_requires=test_requires_list,
)