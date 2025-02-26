"""Setup script for the IFEval package."""

from setuptools import setup, find_packages

setup(
    name="ifeval",
    version="0.1.0",
    description="Instruction Following Evaluation Framework",
    author="Kolibrify Team",
    author_email="example@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langdetect>=1.0.9",
        "nltk>=3.6.0",
        "immutabledict>=2.0.0",
        "absl-py>=0.12.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)