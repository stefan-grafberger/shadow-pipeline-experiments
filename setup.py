"""
For 'python setup.py develop' and 'python setup.py test'
"""
import os
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)

with open(os.path.join(ROOT, "requirements", "requirements.txt")) as f:
    required = f.read().splitlines()

with open(os.path.join(ROOT, "requirements", "requirements.dev.txt")) as f:
    test_required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="shadow_pipeline_experiments",
    version="0.0.1.dev0",
    description="Towards Interactively Improving ML Data Preparation Code via Shadow Pipelines",
    author='Stefan Grafberger',
    author_email='stefangrafberger@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    tests_require=test_required,
    extras_require={'dev': test_required},
    license='Apache License 2.0',
    python_requires='==3.9.*',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9'
    ]
)
