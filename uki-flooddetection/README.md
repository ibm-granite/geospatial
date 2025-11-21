# A Fine-Tuned Geospatial Foundation Model for Flood Detection

## About
Public notebooks and utilities for running inference and fine tuning

## Setup
1. clone the repository: 
```sh
git clone https://github.com/ibm-granite/geospatial.git
```
2. Set up a virtual environment. Python 3.12 is recommended.
3. Activate the virtual environment. 
4. Install the relevant packages. This will depend on where you're running the notebooks.
  - on Google Colab `pip install -e ./geospatial/uki-flooddetection[colab]` 
  - on local machine 
      ```
      cd geospatial/uki-flooddetection
      
      pip install -e .[notebooks]
      ```
      please note: depending on the set-up on your machine, slightly different notations may be needed e.g. `pip install -e ."[notebooks]"`

## Notebooks 
We have two notebooks available. 

1. Check out the [1_getting_started](./notebooks/1_getting_started.ipynb) notebook to run a fine-tuned flood detection model. 
2. For a demonstration of fine tuning for flood detection in a different location, please see the [2_fine_tuning-tuning](./notebooks/2_fine_tuning.ipynb) notebook

## Data
Data to run the examples is archived [here](https://zenodo.org/records/14216851). Data download is handled automatically by the download script.

## IBM Public Repository Disclosure
All content in this repository including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.