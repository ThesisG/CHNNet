# Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence


This `README.md` file contains the instructions for implementing the experiments accompanying the paper.


## Dataset

- **Boston Housing:** This dataset is automatically imported from TensorFlow while executing the code.
- **MNIST:** This dataset is automatically imported from TensorFlow while executing the code.
- **Fashion MNIST:** This dataset is automatically imported from TensorFlow while executing the code.
- **EMNIST:** This dataset is automatically imported from TensorFlow Datasets while executing the code.


## Environment Setup
All the experiments accompanying the paper have been conducted using TensorFlow 2.10. A detailed guideline for installing TensorFlow with pip can be found on their official [website](https://www.tensorflow.org/install/pip).


## Requirements

The experiments are carried out in a Python 3.8 environment. The following additional packages are required to run the tests:
- tensorflow-datasets (version 1.2.0)
- matplotlib (version 3.7.1)
- pandas (version 1.5.3)
- scikit-learn (version 1.2.1)


The dependencies can be installed manually or by using the following command:
```
pip install -r ./requirements.txt
```
It is recommended to use a virtual environment for installing all the modules.

### Potential Errors
While using tensorflow-dataset, you can encounter the following error:
```
importError: cannot import 'builder' from google.protobuf.internal
```
To fix this error, you can install protobuf version 3.20 using the following command:
```
pip install protobuf==3.20
```

## Code Explanation

### CHN Layer


The CHN layer is coded in the `CHNLayer.py` file using `Layer` superclass of Keras. The variables named `kernel_Input_Units` and `kernel_Hidden_Units` in `build` function represent the two sets of weights mentioned in the paper.


The `call` method defines the forward pass of the layer and can handle different types of inputs. The backpropagation of the layer is handled by tensorflow itself. TensorFlow's automatic differentiation mechanism has been used to calculate the gradients during backpropagation.


### Test Files

The py files named {dataset_name}Test.py holds the codes for the tests on that respective dataset. By executing the codes in these files, the test results for the respective dataset can be generated.

### Parameters

The following adjustable parameters allow for customization of the model's architecture, training duration, and optimization strategy based on the specific task and dataset:


1. `epochs`: represents the number of epochs for training.
2. `batchSize`: represents the size of each batch for training.
3. `CHN_hn`: represents the number of hidden neurons in the n<sup>th</sup> hidden layer of the CHNLayer.
4. `MLP_hn`: represents the number of hidden neurons in the n<sup>th</sup> hidden layer of the Dense layer.
5. `loss`: determines the objective function used to measure the model's performance and guide its learning during training.
6. `optimizer`: determines the algorithm used to optimize the neural network model during training.


### Results

- When the training is complete, the model summary and statistical test results are displayed on the terminal.
- The loss graphs for each seed are displayed in separate windows afterward.