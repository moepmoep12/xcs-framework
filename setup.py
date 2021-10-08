from setuptools import setup, find_packages

from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

setup(
    name='xcsframework',
    version='0.1',
    description='A framework for the eXtended Classifier System (XCS) in Python',
    author='Andreas Schmidt',
    author_email='moepmoep12@gmail.com',
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    install_requires=["overrides", "numpy"]
)
