# `reporting`

[![JFrog](https://img.shields.io/badge/JFrog-Artifact%20(QB/Mckinsey%20only)-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.reporting)

## Installation

```shell
pip install oai.reporting
```
See [package installation guide](../../../README.md) for more details

## Overview

This package contains following subpackages (see the package diagram below to understand
the dependencies between them):
* `rendering` - converts report structures into shareable html files
  * a report structure is a (nested) dictionary containing charts
  * the dictionary keys are the titles (and subtitles) of the sections of the report
  * a report structure contains multiple analyses about a topic
  * `interactive` – submodule that provides widgets to ease charts wrangling in jupyter
* `charts` – provides several charts
  * charts are figures, tables or text that present information
  * a dictionary of (a combination of) figures, tables and/or text that together present
    a piece of information is also considered a chart
* `kedro_utils` - provides custom dataset implementations to save reports using the `kedro` framework
* `api` – provides users with an external API
* `datasets` – provides mock data for showcasing functionality

![](../../docs/diagrams/reporting_package_overview.png)


## Usage 

These usage guides are divided into two categories:
- tutorials - providing an entry point for new users and describing the basic functionality
- how-to guides - presenting the details of the functionality.  
  The how-to guides are aimed at users who have a basic understanding and want to leverage the full functionality of the `reporting` package in their line of work

### Tutorials
  + [![](https://img.shields.io/badge/TUTORIAL-reports-orange?logo=Jupyter&style=flat)](notebooks/reports_tutorial.ipynb) — composing both custom reports
### How-to guides
  + [![](https://img.shields.io/badge/HOW--TO-charts-orange?logo=Jupyter&style=flat)](notebooks/charts/charts.ipynb) — how to plot various charts available in reporting
  + [![](https://img.shields.io/badge/HOW--TO-charts.batch__analytics-orange?logo=Jupyter&style=flat)](notebooks/charts/batchplot.ipynb) — how to plot use-case-specific charts
  + [![](https://img.shields.io/badge/HOW--TO-rendering--html-orange?logo=Jupyter&style=flat)](notebooks/rendering_html_how_to.ipynb) — how to apply advanced customization to the html reports

## Custom datasets

Please see the [Custom report dataset tutorial](notebooks/CustomDataset_usage.md) to
learn more about using these with kedro.


## Package architecture

Please see the [API section](../../../../../docs/build/apidoc/reporting/modules.rst)
to learn more about package structure and descriptions of the functions and classes.

Before going into the details familiarize yourself with package structure below.

[UML cheat sheet](http://uml-diagrams.org).

### `reporting.charts` subpackage diagram

The `charts` module contains plotting functions that produce
package compatible figures (see `api.types.PlotlyLike`, `api.MatplotlibLike`).

The available charts are organized at the subpackage level
* `feature_overview`
  * timelines and distribution of features
* `modeling`
  * validation strategy
  * model performance
  * model explanation
* `batchplot`
  * comparison of batch profiles
  * comparison of sensor values for different batches
* `primitives`
  * consolidates reusable plots
* `utils`
  * provides reusable charts functions

The package diagram below illustrates the dependencies between these packages, as well
as the types defined within the `charts` module.


![](../../docs/diagrams/reporting_charts_package_overview.png)

[More details about `reporting.charts`](../../../../../docs/build/apidoc/reporting/reporting.charts.rst)


## Troubleshooting

The reporting module sometimes generates charts with wrong resolution and small fonts,
with matplotlib on MacOS.

Below is an example of this issue:
![](../../docs/diagrams/bad_rendering.png)

Current solution involves temporarily changing the `matplotlib` backend to a
standard one (`agg` by default) within the code responsible for generating problematic
figures as shown below.

```python
from reporting.config import with_default_pyplot_backend

@with_default_pyplot_backend  # <<<<<<<< JUST ADD THE DECORATOR
def create_some_plot_or_dict_of_plots(...):
    ...
```
The resulting properly rendered figures would look as below

![](../../docs/diagrams/good_rendering.png)

[Link to GitHub Issue](https://github.com/McK-Private/optimus/issues/3101)

