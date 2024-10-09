
# Localization Workshop: Hands-on

## Prerequisites

[Conda](https://docs.conda.io/en/latest/miniconda.html) should be installed on your system.

## Steps to Set Up the Environment

1. Clone this repository

2. Create a Conda environment using the `environment.yaml` file.

   ```bash
   conda env create -f environment.yaml
   ```

   This will install all the necessary dependencies specified in the file.

3. Activate the newly created environment.

   ```bash
   conda activate locenv
   ```

4. Launch Jupyter Notebook.

   ```bash
   jupyter-notebook
   ```

   This will open Jupyter Notebook in your default web browser. Open the notebook *workshop.ipynb*

## Alternative
If you do not have Conda, it should be sufficient to install the following dependencies: matplotlib, numpy, notebook.


