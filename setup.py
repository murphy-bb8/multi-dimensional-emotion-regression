"""
Multi-dimensional Emotion Regression with Deep Learning

This package provides implementations for multi-dimensional emotion prediction
using advanced deep learning techniques including BERT-based models and
Mixture of Experts architectures.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-dimensional-emotion-regression",
    version="1.0.0",
    author="Wenxuan Wang, Huilin Zuo",
    author_email="maytheforce806@gmail.com",
    description="Multi-dimensional emotion regression using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/murphy-bb8/multi-dimensional-emotion-regression",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-train=src.training.train:main",
            "emotion-evaluate=src.evaluation.evaluate:main",
            "emotion-preprocess=src.data.preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "deep learning",
        "emotion recognition",
        "natural language processing",
        "bert",
        "regression",
        "machine learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/murphy-bb8/multi-dimensional-emotion-regression/issues",
        "Source": "https://github.com/murphy-bb8/multi-dimensional-emotion-regression",
        "Documentation": "https://multi-dimensional-emotion-regression.readthedocs.io/",
    },
)
