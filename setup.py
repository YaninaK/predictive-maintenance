#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="predictive-maintenance",
    version="1.0",
    description="LSTM for predictive maintenance",
    author="Yanina Kutovaya",
    author_email="kutovaiayp@yandex.ru",
    url="https://github.com/YaninaK/predictive-maintenance.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)