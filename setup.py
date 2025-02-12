import os
from setuptools import setup, find_packages

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="maxwent",
    version="0.3.0",
    description="Maximum Weight Entropy",
    author="Antoine de Mathelin",
    author_email="antoine.demat@gmail.com",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        'tf': ['tensorflow>=2.16'],
        'torch': ['torch>=1.0'],
    },
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
