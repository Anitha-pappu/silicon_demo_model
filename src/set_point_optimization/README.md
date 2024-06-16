# `set_point_optimization`
[![Tutorial notebook](https://img.shields.io/badge/jupyter-tutorial_notebook-orange?style=for-the-badge&logo=jupyter)](notebooks/set_point_optimization.ipynb)

## Overview
Set-point optimization is the most widely used Optimus use case
that helps clients to maximize their plant production
by providing optimal set-points in real-time for machines in their plants.

## Dataflow diagram
![img.png](docs/source/_images/spo_dataflow.svg)


## Installation
To install `set_point_optimization` use case, please follow steps below.

1. Install [Conda](https://www.anaconda.com/products/distribution).<br>

2. Create and activate a new conda environment.

    ```bash
    conda create -n "optimus" python=3.10
    conda activate optimus
    ```

3. Change directories to the root of your use case.

    ```bash
    cd /path/to/set_point_optimization/
    ```

4. Install requirements of the use case.

    ```bash
    pip install -r src/requirements.txt
    ```

5. Follow tutorial notebook `set_point_optimization/notebooks/set_point_optimization.ipynb` to understand end-to-end use case.
<br><br>


## Tutorials
```{eval-rst}

.. toctree::
   :maxdepth: 1

   notebooks/set_point_optimization.ipynb
   ../../../set_point_optimization_kedro/src/set_point_optimization_kedro/README.md
```

## Building Documentation

To generate a documentation for this use case project, please follow the steps below.

1. Install the packages required for building documentation by `sphinx`.

   ```bash
    pip install -r docs/requirements.txt
    ```

2. Run the following command under the root directory of the use case project

   ```bash
   ./docs/build.sh
   ```

3. Once the documentation building process is finished, you can find the documentation under `docs/build/html/` and you can start browsing the document by opening `docs/build/html/index.html` in your browser.


### Update documentation to reflect your adjustment in the usecase
Very likely, you will use this use case as a starting point and build new features or make adaptation upon it. This means that you may want to update the content and strucutre of the documentation accordingly. You can simply modify the following files and rebuild the documentation.

1. `index.rst`: Use this file to update the index structure of your documentation.

2. `build.sh`: Update this file if you want to add or remove any API documentation for the packages.

3. `conf.py`: Update this file if you want to change configuration on how the documentation should be rendered. More details can be found at [Sphinx's documentaion](https://www.sphinx-doc.org/en/master/usage/configuration.html).

4. Update any files that will be included in the documentation according to `index.rst` 