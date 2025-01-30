---
title: "What are MATLAB's feedforwardnet default hyperparameters?"
date: "2025-01-30"
id: "what-are-matlabs-feedforwardnet-default-hyperparameters"
---
MATLAB’s `feedforwardnet` function, used to construct a multilayer perceptron neural network, initializes its structure with specific default hyperparameters that significantly influence training behavior and model performance. These default values, while functional for initial experimentation, often require adjustment to achieve optimal results for specific datasets and problem types. Understanding these defaults is crucial before attempting fine-tuning.

The key default architecture of a `feedforwardnet` when called without explicit layer definitions is a single hidden layer containing 10 neurons, connected to the input and output layers according to the provided data dimensionality. Specifically, without specifying `hiddenSizes`, the network defaults to this single hidden layer configuration. Beyond layer count and neuron number, several other crucial aspects are set automatically. The transfer (activation) function for the hidden layer neurons defaults to the hyperbolic tangent sigmoid function, typically represented as `tansig` in MATLAB. The output layer's transfer function defaults to the linear function, `purelin`, when classification isn't explicitly specified. Training behavior is further governed by the chosen training algorithm, defaulted to Levenberg-Marquardt backpropagation, accessible via `trainlm`, a gradient-based method designed for efficient error minimization. Finally, performance evaluation is measured using mean squared error (`mse`) by default. These configurations form the foundation for any `feedforwardnet` object generated using the basic syntax. I have observed that ignoring these details during initial development leads to inefficient debugging and a frustrating workflow when developing custom neural models. Let’s examine code examples to show these concepts practically.

```matlab
% Example 1: Creating a default feedforwardnet and examining its properties

net = feedforwardnet; % Creates a network using default parameters

disp('Hidden Layer Size:');
disp(net.layers{1}.size); % Displays the hidden layer size (10)

disp('Hidden Layer Transfer Function:');
disp(net.layers{1}.transferFcn); % Displays the hidden layer transfer function ('tansig')

disp('Output Layer Transfer Function:');
disp(net.layers{2}.transferFcn); % Displays the output layer transfer function ('purelin')

disp('Training Function:');
disp(net.trainFcn); % Displays the training function ('trainlm')

disp('Performance Function:');
disp(net.performFcn); % Displays the performance function ('mse')

```

This first example directly instantiates a `feedforwardnet` with no explicit parameters. Through querying the object’s properties, it confirms the default values I previously described. `net.layers{1}.size` provides the count of neurons in the first hidden layer, `net.layers{1}.transferFcn` and `net.layers{2}.transferFcn` show the transfer functions for hidden and output layers respectively. `net.trainFcn` displays the learning algorithm, and `net.performFcn` indicates the performance evaluation metric. I’ve often used this method for verifying defaults when debugging issues stemming from unexpectedly configured nets. The output of this block clearly displays the hidden layer size, hidden and output layer transfer functions, training algorithm, and performance function.

```matlab
% Example 2: Training with default settings

% Sample Input and Target Data (Simple XOR example)
inputs = [0 0 1 1; 0 1 0 1];
targets = [0 1 1 0];

% Create default feedforwardnet
net = feedforwardnet;

% Train the network
net = train(net, inputs, targets);

% Simulate the network
outputs = net(inputs);

disp('Network Outputs:');
disp(outputs);

% Calculate Mean Squared Error
mse_value = mse(targets, outputs);
disp('Mean Squared Error:');
disp(mse_value);

```
This second example moves to practical training. The example constructs a simple XOR dataset and utilizes the previously constructed default `feedforwardnet` network. I find XOR a robust benchmark for testing network functionality given it's a non-linear function. The `train` function is used to conduct the learning process, using the default Levenberg-Marquardt algorithm and mean squared error performance function. The resulting output, although simple, demonstrates the core behavior when these defaults are employed, even without further adjustment. This scenario highlights how the network's structure influences learning when only data is supplied.

```matlab
% Example 3: Customizing with non-default parameters

% Sample Input and Target Data (Same as Example 2)
inputs = [0 0 1 1; 0 1 0 1];
targets = [0 1 1 0];

% Create a network with 2 hidden layers (5 neurons each),
% sigmoid transfer function, and gradient descent training
net = feedforwardnet([5 5], 'trainscg');
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
% Train the network
net = train(net,inputs,targets);

% Simulate the network
outputs = net(inputs);

disp('Network Outputs:');
disp(outputs);

% Calculate Mean Squared Error
mse_value = mse(targets, outputs);
disp('Mean Squared Error:');
disp(mse_value);

```

In this third example, I showcase how to override the default hyperparameters of `feedforwardnet`. The syntax `feedforwardnet([5 5], 'trainscg')` creates a network with *two* hidden layers, each containing five neurons, and sets the training function to scale conjugate gradient, specified by `trainscg`. Furthermore, the activation functions of layers are changed to logistic sigmoid by `logsig`, thus providing a clear demonstration of adjusting the default configuration. This customization illustrates the control I’ve needed during more complex projects; a single set of defaults doesn’t always address all cases. The resulting model, after training, is expected to have a different behavior profile compared to the default parameterization.

When beginning with a new modeling problem in MATLAB, it's crucial to be familiar with the default hyperparameter settings because they impact both the network's learning capacity and the training process, as evidenced in the provided examples. Understanding the default of a single hidden layer with 10 neurons, `tansig` transfer, `purelin` on the output, `trainlm`, and `mse` allows me to start my custom tuning more effectively.

Regarding further learning, I recommend consulting the official MATLAB documentation on the neural network toolbox; specifically, look for sections detailing the `feedforwardnet` function. Textbooks on neural network design, particularly those focusing on multilayer perceptron networks, provide foundational understanding of the theory behind these elements. Furthermore, exploring examples provided within MATLAB’s neural network toolbox tutorials is very useful for seeing how these structures are applied to a variety of problems.  It’s a good practice to experiment with changing the hyperparameters on small problems to get an intuition for their effects, similar to the examples I provided. While software documentation is key to specific function usage, a combination of conceptual understanding and empirical experimentation is necessary for robust neural network design within any environment. By carefully considering the impact of each element – layer size, transfer function, learning algorithm, and performance metric – I’ve consistently achieved better model performance.
