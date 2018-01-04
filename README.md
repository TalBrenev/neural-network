# Neural Network
This is a python implementation of a simple feedforward neural network, along with a few example scripts which use the network.

## Usage
To use the neural network class, first import everything from `neural.py`:
```python
from neural import *
```
You can now create an instance of the `Network` class.
The constructor takes two parameters:
- `num_layers`: An integer representing the number of layers in the neural network
- `neurons_in_layer`: A list containing the number of neurons in each layer, in order

For example:
```python
network = Network(3, [64, 30, 8])
```
The above line of code will create a neural network with 3 layers, containing a layer of 64 input neurons, followed by a hidden layer of 30 neurons, followed by a layer of 8 output neurons.

Note that `num_layers` must be greater than or equal to 2, and the number of elements in `neurons_in_layer` must be equal to `num_layers`. The number of neurons in each layer must be greater than or equal to 1.

### Feeding Forward
To calculate the output of the network when it is given a certain set of inputs, use the `FeedForward` method.
This method takes a single parameter, `inputs`, which is a list of floats.
The number of elements in `inputs` must be equal to the number of input neurons in the network.
The method returns a list of floats representing the output of the network.
For example, if `network` is a neural network with 5 input neurons, we could use the `FeedForward` method as follows:
```python
result = network.FeedForward([0.5, 0.3, 1.0, 0.2, 0.1])
```

### Training

You can train the neural network using the `Train` method. This method takes three parameters:
- `inputs_set`: A list of lists. The inner lists consist of floats, representing a single set of inputs to the neural network
- `expected_set`: A list of lists. The inner lists consist of floats, representing a single set of expected outputs from the neural network
- `learning_rate`: A float which dictates how fast the network should learn

The number of elements in `inputs_set` and `expected_set` must be equal. The learning rate must be a positive number.
Each of the inner lists in `inputs_set` must have a number of elements equal to the number of input neurons in the network. Similarly, each of the inner lists in `expected_set` must have a number of elements equal to the number of output neurons in the network.

Usage of the `Train` method is shown in the example below:
```python
network = Network(3, [2, 10, 2])
network.Train([[0.1, 0.2], [0.5, 0.3]], [[0.5, 0.8], [0.6, 0.7]], 3)
```

Alternatively, you can train the neural network using data in a text file, with the `TrainFromFile` method.
The method takes two parameters: `file_name`, which is a path to the training file, and `learning_rate`, which was described above.
The file must be formatted as follows:
- The first line contains the number of training sets, T
- The next 2T lines alternate between:
  1. A line of space-separated floats representing a set of inputs
  2. A line of space-separated floats representing a set of expected outputs

Below is an example of a training file.
The data in this training file is exactly the same as the data passed to the `Train` method in the example above.
```
2
0.1 0.2
0.5 0.8
0.5 0.3
0.6 0.7
```

### Saving and Loading

The `Network` class has methods for saving/loading instances of the class into a text file.
This is shown in the below example:
```python
# Save the network to the file path 'my_network.nn'
network.SaveNetwork('my_network.nn')
# Load the network at the file path 'my_network.nn'
loaded_network = Network.LoadNetwork('my_network.nn')
```

## Examples

### Even or Odd?
`odd_even.py` shows how to create and train a neural network which checks whether a number is even or odd.
This script creates a network with 16 input neurons and 1 output neuron.
The inputs represent a 16-bit number. The output of the network should be 1 if the number is even, or 0 if the number is odd.
The script trains the network using the first 1000 natural numbers.
It then asks the user to input numbers between 0 and 65535, and uses the trained network to determine whether each inputted number is even or odd.

### Identifying Digits
`digits.nn` contains data for a neural network which was trained using the [MNIST database of handwritten digits.](http://yann.lecun.com/exdb/mnist/) `digits.py` loads this network, and asks the user for file names of images with a resolution of 28x28. The script then uses the neural network to identify which digit is drawn in the image. The network can identify the correct digit with an accuracy of ~92%. Note that this script requires [Pillow](https://python-pillow.org/) to run.
