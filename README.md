# About

This is a framework called **xcsframework** for the eXtended Classifier System (XCS) written in Python. It includes the basic XCS aswell as a variant capable of real valued input. The XCS is composed of components and is thus highly flexible. The framework offers basic implementations to be used out of the box. 

The XCS implementation is based upon Butz & Wilsons paper "An algorithmic description of XCS" (DOI: [10.1007/s005000100111](http://link.springer.com/10.1007/s005000100111)).

The real valued implementation is based upon Stone & Bull's paper "For Real! XCS with Continuous-Valued Inputs" (DOI: [10.1162/106365603322365315](http://www.mitpressjournals.org/doi/10.1162/106365603322365315) ) & Wilson's paper "Get Real! XCS with Continuous-Valued Inputs" (DOI: [10.1007/3-540-45027-0_11](https://link.springer.com/chapter/10.1007/3-540-45027-0_11)).


For more information about the framework and XCS refer to the [wiki](home).

# Install
The framework can be installed as a library. Note that it requires at least **python version 3.8**.  In order to do so run the following command in the root folder of this repository (where setup.py is located):

`pip install -e .`

# Running the examples
After xcsframework has been installed, one can run the examples found in the examples folder. To run any of the examples, simply execute the following command within the repository root:

`python examples/EXAMPLE_NAME.py`

Where EXAMPLE_NAME is the name of the example/file. Alternatively one can also use absolute paths.

## XOR example
This example demonstrates how a XCS can be used to learn a variant of the XOR function. The file can be found [here](examples/xor.py). The process will be output to the console. The training should be done within 10s and the accuracy should reach 100%. If not, simply run again. 

## Multiplexer example
There are two examples for the multiplexer problem. One uses a [binary representation](examples/multiplexer.py) and the other uses [real values](examples/multiplexer_real.py). At default the 6-bit Multiplexer will be used. The XCS should reach 100% accuracy on the binary representation. The real valued version should obtain an accuary of >= 80%.

This example is also available as a [jupyter notebook](notebooks/multiplexer.ipynb).

## Cartpole example
This example uses the [gym environment](https://gym.openai.com/). In order to run this example you have to first install gym:

`pip install gym[all]`

The '[all]' suffix installs further dependencies and thus allows rendering the game. 

In this example the XCS shall learn to control a pole upon a car. The environment is represented by real values. The XCS struggles to learn this example but the result is much better than random movement.

Upon starting the example the XCS starts training without rendering. The process will be output to the console. After training is finished, the XCS uses the best population to test the gained knowledge base. This test will be rendered.
