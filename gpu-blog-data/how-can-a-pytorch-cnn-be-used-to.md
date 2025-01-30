---
title: "How can a PyTorch CNN be used to play Connect Four?"
date: "2025-01-30"
id: "how-can-a-pytorch-cnn-be-used-to"
---
Convolutional Neural Networks (CNNs), initially designed for image recognition, can be effectively adapted to represent and process game board states, such as those in Connect Four, enabling an AI agent to make strategic decisions. I've personally utilized this approach in developing game-playing AI for various board games. The key is to frame the board as a spatial input, which aligns well with the strengths of convolutional operations.

A Connect Four board, typically 6 rows by 7 columns, can be directly represented as a tensor. We encode each cell not as a pixel, but as a feature representing its state. For instance, an empty cell might be represented by 0, a cell occupied by the AI by 1, and a cell occupied by the opponent by -1. This creates a 6x7 matrix that serves as the input to the CNN. The core idea isn't image recognition, but recognizing patterns of player moves, which translate to specific arrangements in the matrix. We are effectively asking the network to "see" opportunities or threats on the board.

The CNN's architecture then becomes critical. Instead of classifying images, we need the network to evaluate the board and suggest a column to play. This is achieved through several convolutional layers, interspersed with activation functions (e.g., ReLU) and potentially pooling layers, although their impact can vary in this context. The convolutional layers are where spatial patterns within the board, such as sequences of three of one player's pieces, are detected. These convolutional filters, learned during the training process, become pattern detectors specific to the game. After the convolutional layers, we typically have fully connected layers that reduce the output to a one-dimensional vector that represents scores assigned to each possible column. The column with the highest score is the move the agent would make. This setup treats move selection as a classification task over the set of columns.

Training a CNN for Connect Four usually involves one of several methods. Supervised learning, where an agent learns from example game play by expert human or AI, is one method. The training data would consist of pairs of board states and the corresponding optimal column choice for those states. Alternatively, reinforcement learning can be implemented. A popular example involves training using Monte Carlo Tree Search (MCTS) enhanced with a value network. In this approach the CNN is trained to predict the value of given board states, thereby guiding the MCTS algorithm to explore more promising parts of the game tree. The value network is essentially a CNN that outputs a single value, estimating who has the advantage. Regardless of method, gradient descent is used to optimize the network's weights.

Here are a few code examples illustrating key concepts:

**Example 1: Defining the CNN architecture:**

```python
import torch
import torch.nn as nn

class ConnectFourCNN(nn.Module):
    def __init__(self):
        super(ConnectFourCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Input is 1 channel (board state), 32 filters
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filters
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 7, 128) # Flattened input from convolutions, hidden layer of 128
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 7)  # Output: Scores for each column

    def forward(self, x):
        x = x.unsqueeze(1) # Adding channel dimension for input
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Usage example
model = ConnectFourCNN()
dummy_input = torch.randn(1, 6, 7) # Batch size 1, 6x7 board
output = model(dummy_input)
print(output.shape) # Output shape: torch.Size([1, 7]) which is the score for each of the seven columns
```

This example shows the fundamental structure of a CNN for Connect Four. The input has a single channel because it is not a color image, and is passed through convolutional layers that extract features. After flattening, the result is fed through fully connected layers that result in one score per possible move column. The comment at the bottom shows how to get the scores output by the network and what shape those scores take. The network structure is relatively simple to start with and can be expanded based on testing and iterative development.

**Example 2: Converting a game board to a tensor:**

```python
import torch
import numpy as np

def board_to_tensor(board):
    """Converts a Connect Four board to a PyTorch tensor.
        Assumes board is a list of lists, with 0 for empty, 1 for player, and -1 for opponent"""
    board_array = np.array(board, dtype=np.float32)
    tensor = torch.from_numpy(board_array)
    return tensor

# Example usage:
board_state = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, -1, 1, 0, 0, 0],
    [0, 0, -1, -1, 1, 0, 0],
    [0, 0, 1, -1, -1, 0, 0]
    ]

tensor = board_to_tensor(board_state)
print(tensor.shape) # Output: torch.Size([6, 7])
print(tensor) # Output shows the converted tensor with float dtype
```

Here, the `board_to_tensor` function provides a critical step in preprocessing game states. It receives the raw state represented as a list of lists and converts it into the necessary PyTorch tensor with floating-point data type, which the CNN can then process. The example shows both the tensor's shape and contents, reinforcing the conversion. This stage is crucial as the CNN requires numerical input that represents the board state.

**Example 3: Selecting the best move from the CNN's output:**

```python
import torch

def select_move(cnn_output):
    """Selects the column with the highest score from CNN output"""
    scores = cnn_output.squeeze() # Remove batch dimension
    best_column = torch.argmax(scores)
    return best_column.item()

# Assuming 'output' is the CNN result from example 1
# Using example 1 output in this example
# Generating dummy output scores
cnn_output = torch.tensor([[0.2, 0.5, -0.1, 0.8, 0.1, -0.3, 0.4]])

best_move_column = select_move(cnn_output)
print(f"Best column to play: {best_move_column}")
```

This example displays how to interpret the output of the network and select a column. After receiving the network output, `select_move` function squeezes the batch dimension (if it exists) and uses the `argmax` function to find the column with the highest associated score. The column index is extracted and returned as an integer. Using the dummy values, it identifies the 4th column (index 3) as the optimal choice to take. The decision-making process is therefore integrated with the outputs of the CNN.

In summary, utilizing CNNs for Connect Four involves representing the board as a tensor, training a network to either classify best move columns directly or estimate board values, and then using that network to guide move selection.

For further learning, I recommend exploring materials on:
1.  **Deep learning with PyTorch:** Focus on tutorials for CNN implementations.
2.  **Reinforcement Learning**: Especially those emphasizing Deep Q-Learning or policy gradient methods.
3.  **Game AI algorithms:** Look at resources discussing Monte Carlo Tree Search.
4.  **Practical implementation examples:** Search for implementations of CNNs for other games, then generalize that knowledge to Connect Four.
