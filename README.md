# MaxWEnt

[![PyPI version](https://badge.fury.io/py/maxwent.svg)](https://pypi.org/project/maxwent)
[![Build Status](https://github.com/antoinedemathelin/maxwent/actions/workflows/run-test.yml/badge.svg)](https://github.com/antoinedemathelin/maxwent/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10|%203.11-blue)](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10|%203.11-blue)

**Max**imum **W**eight **Ent**ropy

This repository offers a packaged implementation of MaxWEnt, based on the original method described in [Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization (JMLR)](https://www.jmlr.org/papers/v26/23-1359.html)

---

MaxWEnt is a method designed for epistemic uncertainty quantification in deep learning. It applies the Maximum Entropy principle to stochastic neural networks. The goal of MaxWEnt is to determine the weight distribution with the highest entropy that remains consistent with the training observations. By doing so, the method enhances the epistemic uncertainty, which generally leads to improved detection of out-of-distribution samples.

<table>
  <tr valign="top">
    <td width="50%" >
        <a href="https://antoinedemathelin.github.io/maxwent/1D_Regression_Example.html">
            <br>
            <b>Regression Example</b>
            <br>
            <br>
            <img src="https://github.com/antoinedemathelin/maxwent/blob/42cee2020a52850666b6abb521c30f5ef1d3ce9e/docs/imgs/regression1d.png">
    </a>
    </td>
    <td width="50%">
        <a href="https://antoinedemathelin.github.io/maxwent/2D_Classification_Example.html">
            <br>
            <b>Classification Example</b>
            <br>
            <br>
            <img src="https://github.com/antoinedemathelin/maxwent/blob/a016e60b39d753459871f9bdc5fc8b7973ac2f2d/docs/imgs/classification2d.png">
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

## Reference

If you use this repository in your research, please cite our work using the following reference:

```
@article{JMLR:v26:23-1359,
  author  = {Antoine de Mathelin and Fran{\c{c}}ois Deheeger and Mathilde Mougeot and Nicolas Vayatis},
  title   = {Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  volume  = {26},
  number  = {4},
  pages   = {1--68},
  url     = {http://jmlr.org/papers/v26/23-1359.html}
}
```
