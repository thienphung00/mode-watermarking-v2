#!/usr/bin/env python3
"""Setup script for mode-watermarking package."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mode-watermarking",
    version="0.1.0",
    author="Mode Watermarking Team",
    author_email="contact@modewatermarking.com",
    description="Advanced watermarking system for diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mode-watermarking/mode-watermarking",
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
