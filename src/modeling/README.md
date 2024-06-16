# `modeling`
[![JFrog](https://img.shields.io/badge/JFrog-Artifact%20(QB/Mckinsey%20only)-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.modeling)

## Installation
```shell
pip install oai.modeling
```
See [package installation guide](../../../README.md) for more details

## Tutorials
Please see tutorial notebooks where we demonstrate functions usage on sample datasets:
+ [![model_base](https://img.shields.io/badge/TUTORIAL-ModelBase-orange?logo=Jupyter&style=flat)](notebooks/model_base.ipynb) – demonstrates how model classes work and interact with each others
+ [![splitter_base](https://img.shields.io/badge/TUTORIAL-SplitterBase-orange?logo=Jupyter&style=flat)](notebooks/splitter_base.ipynb) – demonstrates how splitter classes work 
+ [![modeling](https://img.shields.io/badge/TUTORIAL-modeling-orange?logo=Jupyter&style=flat)](notebooks/modeling.ipynb) – demonstrates how modeling problem can be solved with the functionality provided in the package on sample dataset
+ [![model_performance_report](https://img.shields.io/badge/TUTORIAL-Model_Performance_Report-orange?logo=Jupyter&style=flat)](notebooks/performance_report.ipynb) — creating a model performance report
+ [![sensitivity analyzer](https://img.shields.io/badge/TUTORIAL-Sensitivity_Analyzer-orange?logo=Jupyter&style=flat)](notebooks/sensitivity_analyzer.ipynb) – demonstrates how to launch, use, and customise the Sensitivity Analyzer application
+ [![modeling_gaussian_process](https://img.shields.io/badge/TUTORIAL-modelingGP-orange?logo=Jupyter&style=flat)](notebooks/modeling_gaussian_process.ipynb) – demonstrates how to use Gaussian Process on a sample dataset by creating `GaussianProcessModelFactory` with user defined kernels
+ [![experiment_tracking](https://img.shields.io/badge/TUTORIAL-Experiment_Tracking-orange?logo=Markdown&style=flat)](notebooks/experiment_tracking.md) – illustrates how to integrate `modeling` with experiment tracking tools, such as `mlflow` and `kedro-viz`
+ [![generalized_additive_model](https://img.shields.io/badge/TUTORIAL-GAM-orange?logo=Jupyter&style=flat)](notebooks/generalized_additive_model.ipynb) – demonstrates how to use Generalized Additive Model on a sample dataset
+ [![baselining](https://img.shields.io/badge/TUTORIAL-Baselining-orange?logo=Jupyter&style=flat)](notebooks/baselining.ipynb) – demonstrates how to create a baseline for impact estimation

## Overview

[UML cheat sheet](http://uml-diagrams.org).

Before going into the details familiarize yourself with package structure below.
![package diagram](./notebooks/_images/_modeling.png)

Please see the [API section](../../../../../docs/build/apidoc/modeling/modules.rst) to learn more about package structure and descriptions of the functions and classes

#### `modeling.api`
Provides protocols that `modeling` utilizes internally
![package diagram](./notebooks/_images/_api.png)

#### `modeling.models`
Provides functionality to initialize, train, tune and evaluate regression models with unified API
![model base diagram](./notebooks/_images/_ModelBase.png)

#### `modeling.splitters`
Provides splitting classes with the unified API
![splitter_diagram](./notebooks/_images/_SplitterBase.png)

## Usage
This package is kedro independent, meaning every function or class in this package can 
be imported and used for their own purposes.

It assumes already preprocessed input dataset. Some functionalities utilize `TagDict` with tag-specific settings. 
We provide `sample_model_input_data.csv` and `sample_tag_dict.csv` as references to run tutorial notebooks.

## Additional information
Please see:
* [Plotting for model performance user guide](../../../reporting/src/reporting/notebooks/charts/model_performance.ipynb) from `reporting` package that create a series of useful charts that allow to check and report model performance after fitting a model.
