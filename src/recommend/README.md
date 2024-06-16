# `recommend`
[![JFrog](https://img.shields.io/badge/JFrog-Artifact%20(QB/Mckinsey%20only)-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.recommend)

## Installation
```shell
pip install oai.recommend
```
See [package installation guide](../../../README.md) for more details

## Tutorials
Please see tutorial notebooks where we demonstrate functions usage on sample datasets:
+ [![](https://img.shields.io/badge/TUTORIAL-recommend-orange?logo=Jupyter&style=flat)](notebooks/recommend.ipynb) to learn more about basic package workflow
+ [![](https://img.shields.io/badge/TUTORIAL-optimization_explainer-orange?logo=Jupyter&style=flat)](notebooks/optimization_explainer.ipynb) to learn how to provide a visual explanation for the black-box optimization
+ [![](https://img.shields.io/badge/TUTORIAL-export-orange?logo=Jupyter&style=flat)](notebooks/export.ipynb) to learn how to export `recommend.optimize` results (`Solutions`)
+ [![](https://img.shields.io/badge/TUTORIAL-common_constraints-orange?logo=Jupyter&style=flat)](notebooks/common_constraints.ipynb) to learn about out-of-the-box constraints that are frequently used with OAI
+ [![](https://img.shields.io/badge/TUTORIAL-implementation_tracking-orange?logo=Jupyter&style=flat)](notebooks/implementation_tracking.ipynb) to learn how to asses the implementation status of recommendations
+ [![](https://img.shields.io/badge/TUTORIAL-reports.counterfactual_analysis-orange?logo=Jupyter&style=flat)](notebooks/counterfactual_analysis.ipynb) to learn about a counterfactual analysis report
+ [![](https://img.shields.io/badge/TUTORIAL-Uplift_calculation-orange?logo=Jupyter&style=flat)](notebooks/uplift_calculation.ipynb) to learn how to calculate uplift for impact estimation

## Overview
`recommend` package contains functions to run optimization. This package is dependent on `optimizer` package so if you are not familiar, please refer to documentation on [`optimizer`](../../../optimizer/docs/source/01_get_started/01_optimizer_installation_guide.md).

## Usage
This package is kedro independent, meaning every function or class in this package can 
be imported and used for their own purposes.

It requires components from [`optimizer`](../../../optimizer/docs/source/01_get_started/01_optimizer_installation_guide.md) package, such as: 
- [`solver`](../../../optimizer/docs/source/04_user_guide/03_solver.md)
- [`repair`](../../../optimizer/docs/source/04_user_guide/02_repair.ipynb)
- [`penalty`](../../../optimizer/docs/source/04_user_guide/01_penalty.ipynb)
- [`stopper`](../../../optimizer/docs/source/04_user_guide/05_stopper.md)

Once you get familiar with components in `optimizer` package, please see:
- [API section](../../../../../docs/build/apidoc/recommend/modules.rst) to learn more about package structure and descriptions of the functions
<br><br>
