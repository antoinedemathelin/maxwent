# Maximum Weight Entropy

[![PyPI version](https://badge.fury.io/py/maxwent.svg)](https://pypi.org/project/maxwent)
[![Build Status](https://github.com/antoinedemathelin/maxwent/actions/workflows/run-test.yml/badge.svg)](https://github.com/antoinedemathelin/maxwent/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10|%203.11-blue)](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10|%203.11-blue)

Maximum Weight Entropy

---

Maximum Weight Entropy is an open source library providing ... for Tensorflow and Pytorch

<table>
  <tr valign="top">
    <td width="50%" >
        <a href="https://adapt-python.github.io/adapt/examples/Sample_bias_example.html">
            <br>
            <b>Regression Example</b>
            <br>
            <br>
            <img src="">
        </a>
    </td>
    <td width="50%">
        <a href="https://adapt-python.github.io/adapt/examples/Flowers_example.html">
            <br>
            <b>Classification Example</b>
            <br>
            <br>
            <img src="">
        </a>
    </td>
</table>

## Installation and Usage

This package is available on [Pypi](https://pypi.org/project/maxwent) and can be installed with the following command line: 

```
pip install maxwent
```

You will need either Tensorflow or Pytorch to be installed. If both packages are installed, the Tensorflow framework of maxwent will be used by default.

To change the framework, please use the `set_framework` function:

- Pytorch framework
```python
import maxwent
maxwent.set_framework("torch")
```

- Tensorflow framework
```python
import maxwent
maxwent.set_framework("tf")
```

## Quick Start

...
