---
title: "How can multilayer neural networks be used to produce output in MATLAB?"
date: "2024-12-23"
id: "how-can-multilayer-neural-networks-be-used-to-produce-output-in-matlab"
---

Alright, let’s talk multilayer neural networks in MATLAB. I've spent a good chunk of my career implementing these, from simple classifications to more complex time-series predictions. It's a powerful tool, but getting the output right requires understanding a few core concepts, and the specifics of MATLAB's neural network toolbox.

So, how do we get output? Essentially, it boils down to constructing the network, training it with suitable data, and then feeding it new input to obtain the corresponding output. Let's break down each of these steps, including some code to illustrate the process.

First, creating a multilayer network in MATLAB involves using the `feedforwardnet` or `patternnet` functions. `feedforwardnet` gives you a basic feedforward architecture suitable for regression or function approximation, while `patternnet` is optimized for classification tasks. The key here is defining the number of neurons in each hidden layer. For instance, if you wanted a simple two-layer network, the command might look something like this:

```matlab
hiddenLayerSize = [10 5]; % 10 neurons in the first layer, 5 in the second.
net = feedforwardnet(hiddenLayerSize);
```

Now, `net` is an object representing your network structure. However, it’s untrained. To train the network, we need input (`X`) and target output (`T`) datasets. Suppose we're aiming for a function approximation, and we've generated some synthetic training data:

```matlab
X = linspace(-10, 10, 500);
T = sin(X) + 0.2 * randn(1, 500); % Introduce some noise
```

Training the network is done with the `train` function:

```matlab
net = train(net, X, T);
```

MATLAB uses backpropagation by default, a fairly robust algorithm. You can tune various aspects of the training process using the `net.trainParam` structure, such as the learning rate (`net.trainParam.lr`) or the number of training epochs (`net.trainParam.epochs`). The 'train' function applies these algorithms and adjusts the network's internal weights and biases based on the discrepancy between output and target values. The goal, in essence, is to minimize this discrepancy across the training set.

After training, we can then test or use our neural network using the `sim` function, or directly by using parenthesis indexing `net(X_new)`. We provide new inputs (`X_new`), and the network will produce the corresponding output. For example:

```matlab
X_new = linspace(-12, 12, 200);
Y_new = net(X_new); % Using the parentheses method

plot(X, T, 'o'); % Training points
hold on
plot(X_new, Y_new); % Network output
hold off
legend('Training data', 'Network output');
```

This will visualize the network's approximation. You'll see the trained network tries to fit the underlying sine wave function using only the provided training data.

Let's move on to a slightly more complex scenario, say a classification task using `patternnet`. Assume we're trying to distinguish between two clusters of points. We can generate some artificial data:

```matlab
rng(0); % For reproducibility

cluster1 = 0.5 * randn(2, 200) + repmat([-2; -2], 1, 200);
cluster2 = 0.5 * randn(2, 200) + repmat([2; 2], 1, 200);

X = [cluster1 cluster2];
T = [zeros(1, 200) ones(1, 200)]; % Targets: 0 for cluster 1, 1 for cluster 2.

% Transpose X to fit MATLAB's conventions for input data
X = X';
T = categorical(T);  % make it categorical as it's for pattern classification
```

Here, `X` contains the points and `T` contains the category labels. The target here is categorical since we are doing a classification problem.

Now, we can create and train our pattern network. Since this is a simple classification problem we will choose a single layer network:

```matlab
net_class = patternnet(10); % 10 neurons in the hidden layer

% Set up the training parameters
net_class.trainParam.showWindow = false;
net_class = train(net_class, X', T); % Transpose X to match training parameters requirements
```
After training, for using the network for future predications, we may apply this network to test data and examine the predicted labels:
```matlab
test_cluster1 = 0.5 * randn(2, 50) + repmat([-2; -2], 1, 50);
test_cluster2 = 0.5 * randn(2, 50) + repmat([2; 2], 1, 50);
X_test = [test_cluster1 test_cluster2];
X_test = X_test';

Y_test_class = net_class(X_test'); % Transpose X to match prediction parameters

% Transform the categorical prediction back to numerical
[~, predictedClasses] = max(Y_test_class);
predictedClasses = predictedClasses - 1;

figure;
gscatter(X_test(:, 1), X_test(:, 2), predictedClasses);
xlabel('X1');
ylabel('X2');
title('Classification Result');

```
This code will produce a figure that visually shows the classification done by the network for previously unseen data. The use of categorical output and converting back to numerical classes highlights how pattern recognition and classification tasks are approached.

There’s no magic bullet, though. When your network performs poorly, it usually comes down to a few things. First, the architecture, how many layers and neurons there are, is critical. Experiment with different sizes. Also, consider the activation functions; by default, they’re usually `tansig` (hyperbolic tangent sigmoid) for hidden layers and `purelin` for the output layer, but other options like relu are available. Proper data preprocessing, such as normalization or standardization is vital; feeding unscaled data into a neural network can cause it to struggle. Furthermore, always be mindful of overfitting, where the network memorizes the training data instead of learning the underlying patterns, and address it with techniques like regularization or early stopping during training. The `net.trainParam` structure allows such adjustments.

For a deeper dive into neural networks in general, I would suggest checking out "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. It’s a comprehensive book covering the theoretical foundations and practical applications. For MATLAB specifics, the documentation itself is a valuable resource. The help pages for functions like `feedforwardnet`, `patternnet`, and `train` are surprisingly well written and provide detailed explanations and example usage cases. Also, "Neural Networks: A Comprehensive Foundation" by Simon Haykin offers extensive information on various neural network architectures and their underlying math.

In my experience, the key is iterative experimentation and careful examination of the training data and output. There is never a one-size-fits all model or approach; each problem is unique. By systematically testing and modifying parameters, you can create a neural network that fits your specific needs within MATLAB.
