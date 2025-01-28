from setuptools import setup, find_packages

setup(
    name="maxwent",
    version="0.2.0",
    description="Maximum Weight Entropy",
    author="Antoine de Mathelin",
    author_email="antoine.demat@gmail.com",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        'tf': ['tensorflow>=2.16'],
        'torch': ['torch>=1.0'],
    },
)