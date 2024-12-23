---
title: "How are scores and losses calculated using rubix/ml's multilayer perceptron?"
date: "2024-12-23"
id: "how-are-scores-and-losses-calculated-using-rubixmls-multilayer-perceptron"
---

Let’s tackle this. It’s a common area of confusion, and I've seen it crop up quite a bit in projects over the years, especially when teams are moving beyond simpler linear models. Understanding how rubix/ml computes scores and losses in its multilayer perceptron (MLP) is fundamental to debugging and optimizing these networks, so let's break it down systematically.

First, it's critical to remember that a multilayer perceptron, at its core, is a function approximator. It attempts to learn a complex mapping between input features and output targets. This learning process is driven by the concept of a *loss function*, which quantifies the discrepancy between the network's predictions and the actual ground-truth values. The goal of training, quite simply, is to minimize this loss. This minimization happens through iterative adjustments of the network's internal weights and biases. This is usually done with gradient descent or its variants.

The scores, on the other hand, are typically the raw outputs of the final layer before any post-processing steps (like converting the output to probabilities) might be applied. The nature of these scores depends on the type of problem. For example, in a binary classification scenario, the final layer often has a single output node, which can be interpreted as a logit, which can be converted to the probability of the instance belonging to the positive class. In a multi-class classification, you will have a node for each class.

Now, let’s dive into specifics concerning `rubix/ml`. It's important to note that this library is designed to be quite flexible, so the exact formulas might slightly vary depending on the configuration and the chosen loss function. However, the core concepts remain consistent.

Generally, during the feedforward pass, each neuron's output within a layer can be computed using the following formula:

`output_j = activation_function(sum(weight_ij * input_i) + bias_j)`

Here:
*   `output_j` is the output of the *j*th neuron in the layer.
*   `weight_ij` represents the weight of the connection between the *i*th neuron in the previous layer and the *j*th neuron in the current layer.
*   `input_i` is the output from the *i*th neuron in the previous layer.
*   `bias_j` is the bias term for the *j*th neuron.
*   `activation_function` is a non-linear function like sigmoid, tanh, or relu applied to the weighted sum and bias.

The scores, as previously stated, are simply the outputs of the final layer of the network, *before* the application of any post-processing step (such as softmax or sigmoid) or loss function. So, in this case, the 'output_j' would be your score.

The loss is then calculated based on these scores and the true values.

Let's illustrate with some examples. First, a simple binary classification using binary cross-entropy loss:

```php
<?php

use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\LossFunctions\BinaryCrossEntropy;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Random\Random;

// Sample data (just for illustration)
$data = [[0.1, 0.2], [0.5, 0.6], [0.8, 0.9], [0.2, 0.4]];
$labels = [0, 1, 1, 0];

$dataset = new Labeled($data, $labels);

$layers = [
    new Dense(2),
    new Activation(new Sigmoid()),
    new Dense(1),
    new Activation(new Sigmoid()) // Output activation, score is pre-sigmoid
];

$loss = new BinaryCrossEntropy();
$optimizer = new Adam();

$network = new Network($layers, $loss, $optimizer, 1);

// Pre-training score computation
$scores = $network->predict($dataset);
print_r("Scores before training: ");
print_r($scores->toarray());

// Training
$network->train($dataset);

// Post-training score computation
$scores = $network->predict($dataset);
print_r("\nScores after training: ");
print_r($scores->toarray());
$loss_val = $network->loss($dataset);
print_r("\nLoss after training:");
print_r($loss_val);

?>

```

In the example above, note that before the sigmoid activation, the output of the final dense layer would be considered the score. The sigmoid output is then compared to labels during loss calculations. The Binary cross entropy measures how well the network’s predicted probability aligns with the true label. This specific implementation of `rubix/ml` does not show the raw scores (pre-sigmoid) after every iteration and so we cannot explicitly see these without a bit of a modification. The `loss()` function returns the *average* loss over the dataset.

Next, let's consider a multi-class classification with categorical cross-entropy loss. Here the scores are the raw logit output of the final layer before the softmax function:

```php
<?php

use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use Rubix\ML\NeuralNet\LossFunctions\CategoricalCrossEntropy;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Random\Random;

// Sample data (just for illustration)
$data = [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.4, 0.6]];
$labels = [0, 1, 2, 0];

$dataset = new Labeled($data, $labels);

$layers = [
    new Dense(5),
    new Activation(new ReLU()),
    new Dense(3), // Three output classes
    new Activation(new Softmax()) // Output Activation, score is pre-softmax
];


$loss = new CategoricalCrossEntropy();
$optimizer = new Adam();


$network = new Network($layers, $loss, $optimizer, 1);

// Pre-training score computation
$scores = $network->predict($dataset);
print_r("Scores before training: ");
print_r($scores->toarray());


// Training
$network->train($dataset);

// Post-training score computation
$scores = $network->predict($dataset);
print_r("\nScores after training: ");
print_r($scores->toarray());
$loss_val = $network->loss($dataset);
print_r("\nLoss after training:");
print_r($loss_val);


?>
```

Here, the `Softmax` activation converts the scores from the dense layer into probabilities, which will sum to 1. The categorical cross-entropy then computes the loss based on the network's predicted probabilities and the actual one-hot encoded labels.

Finally, consider a regression task using mean squared error:

```php
<?php

use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\LossFunctions\MeanSquaredError;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Random\Random;

// Sample regression data (just for illustration)
$data = [[0.1, 0.2], [0.5, 0.6], [0.8, 0.9], [0.2, 0.4]];
$labels = [1.5, 2.6, 3.7, 1.8];

$dataset = new Labeled($data, $labels);

$layers = [
    new Dense(5),
    new Activation(new ReLU()),
    new Dense(1) // One output node for regression
];


$loss = new MeanSquaredError();
$optimizer = new Adam();


$network = new Network($layers, $loss, $optimizer, 1);


// Pre-training score computation
$scores = $network->predict($dataset);
print_r("Scores before training: ");
print_r($scores->toarray());

// Training
$network->train($dataset);

// Post-training score computation
$scores = $network->predict($dataset);
print_r("\nScores after training: ");
print_r($scores->toarray());
$loss_val = $network->loss($dataset);
print_r("\nLoss after training:");
print_r($loss_val);


?>
```

In regression, the final dense layer produces a scalar output directly; there's no activation function to transform this raw score. The `MeanSquaredError` loss calculates the average of the squares of the difference between the predicted and actual values.

For those seeking further depth, I strongly recommend exploring *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; it offers a very detailed exposition of these topics. Also, the original paper on backpropagation and gradient descent would be extremely valuable. It's often overlooked but a classic for a reason: *Learning representations by back-propagating errors*, D.E. Rumelhart, G.E. Hinton, and R.J. Williams. In addition to that, you might find the documentation for `rubix/ml` itself useful, since it contains the specifics on their implementation details, alongside examples for each use case.

The core idea, no matter the specifics, is that during forward pass, raw predictions (scores) are output from the final layer and a loss function quantifies the prediction error. The entire training process focuses on iterative minimization of this loss. Understanding this mechanism is key to effectively troubleshooting, debugging and optimization of your network.
