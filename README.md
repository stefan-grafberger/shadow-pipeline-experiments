
## DEEM: Shadow Pipeline Experiments

This repository contains some experiments for the DEEM paper we are writing.


## Local setup

Prerequisite: Python 3.9

1. Clone this repository
2. Set up the environment

	`cd shadow_pipeline_experiments` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>
	
3. Install pip dependencies 

    `SETUPTOOLS_USE_DISTUTILS=stdlib SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -e ."[dev]"` <br>
