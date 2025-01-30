---
title: "How can neural network output be constrained to adhere to chess move legality?"
date: "2025-01-30"
id: "how-can-neural-network-output-be-constrained-to"
---
The core challenge in training a neural network to play chess lies not solely in optimizing for strategic strength, but also in enforcing the rigid rules governing valid piece movements. Directly predicting legal moves within the network's output space is complex and often leads to violations, even with sophisticated training. I've found that treating the network as a *move proposer* rather than a *move generator* significantly improves the adherence to chess legality. The network predicts a probability distribution over a vast, unconstrained output space, and a separate filtering step ensures that the move eventually selected from this distribution is, in fact, legal according to the rules of chess.

My experience has shown that this methodology requires two critical components: a robust move generation engine and a strategic neural network. The engine, external to the network, must be capable of enumerating *all* legal moves for any given chessboard state, with complete fidelity. This acts as the “rule-checker,” effectively constraining the network’s somewhat arbitrary suggestions. The network itself doesn’t need to understand the rules of chess; it merely learns which of the potential moves (supplied by the engine) are most desirable.

The process begins with the engine providing a list of all legal moves for the current board position. Each legal move is represented as a pair of coordinates (origin and destination). These legal moves are used to build a move mask. The network outputs a raw probability distribution over all possible move coordinates (an output space generally much larger than the number of legal moves). The move mask converts this raw output to represent *only* the likelihoods of legal moves. Then, a sampling procedure selects one move based on the masked probability distribution. Finally, this sampled move is executed, advancing the game state, and the process repeats.

Here’s an illustration of how this might function in Python using a simplified representation of a chess board and move generation:

```python
import numpy as np

# Dummy legal move engine. In reality, a more robust library like python-chess would be used.
def generate_legal_moves(board):
    """Generates sample legal moves for a board position. Returns a list of tuples (origin, destination)
       where origin and destination are coordinates on an 8x8 board represented by row and column integers."""

    legal_moves = []
    for row in range(8):
        for col in range(8):
             if (row + col) % 2 == 0: # arbitrary example: moves only on even sum row/col positions
                  for row2 in range(8):
                    for col2 in range(8):
                        if abs(row-row2)<=1 and abs(col-col2)<=1 and (row != row2 or col != col2):
                          legal_moves.append(((row, col), (row2,col2)))
    return legal_moves


# Mock neural network output (logits). In practice, this is an output layer with appropriate activation
def mock_network_output(num_possible_moves):
  return np.random.rand(num_possible_moves)

def coordinates_to_index(row, col):
  return row * 8 + col


def index_to_coordinates(index):
  return index // 8, index % 8


def apply_move_mask(logits, legal_moves):
  """Applies a mask to the neural network output based on legal moves.
     Returns a probability distribution over the legal moves."""
  num_possible_moves = 64*64
  move_mask = np.full(num_possible_moves, -np.inf) # mask of -infinity
  for (origin, destination) in legal_moves:
    origin_index = coordinates_to_index(origin[0], origin[1])
    destination_index = coordinates_to_index(destination[0], destination[1])
    move_index = origin_index * 64 + destination_index # create index over entire possible origin and destination space
    move_mask[move_index] = logits[move_index]
  
  masked_probabilities = softmax(move_mask)
  return masked_probabilities

def softmax(logits):
  exp_values = np.exp(logits - np.max(logits)) # numerically stable softmax
  return exp_values/np.sum(exp_values)



def sample_move(masked_probabilities, legal_moves):
    """Samples a move from the masked probability distribution.
    Returns the selected move (origin, destination) and its index in the list of legal moves"""

    num_possible_moves = 64*64
    probabilities = masked_probabilities.reshape((64,64))
    selected_index = np.random.choice(num_possible_moves, p=masked_probabilities)

    origin_index= selected_index // 64
    destination_index = selected_index % 64

    origin = index_to_coordinates(origin_index)
    destination = index_to_coordinates(destination_index)
    
    selected_move = (origin,destination)

    #find the index of this selected move in the original list
    legal_move_index = -1
    for index, move in enumerate(legal_moves):
        if move == selected_move:
            legal_move_index = index
            break

    return selected_move, legal_move_index


# Simulate one move
board = np.zeros((8,8))  # Dummy board representation
legal_moves = generate_legal_moves(board)

num_possible_moves = 64*64
logits = mock_network_output(num_possible_moves)
masked_probabilities = apply_move_mask(logits, legal_moves)

selected_move, legal_move_index = sample_move(masked_probabilities, legal_moves)

print(f"Legal move selected: {selected_move}, index in legal moves list: {legal_move_index} ")

```
In this example, `generate_legal_moves` produces a list of legal moves for the board.  The `mock_network_output` generates a random probability distribution which represents network output before masking. `apply_move_mask` filters that output and transforms it into a new probability distribution over *only* legal moves.  The `sample_move` function then selects one of those moves. This ensures the network only returns valid moves. Note how a large output space (64*64) is used, to avoid the requirement that the neural network "knows" or enforces chess rules.

For more complicated games, I’ve found it useful to represent the moves via a different method. Consider the following:
```python
def generate_legal_moves_complex(board):
    """Generates sample legal moves for a board position. 
       Returns a list of dictionaries containing a "from" and "to" key, 
       each with row and col integer coordinates."""

    legal_moves = []
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                for row2 in range(8):
                    for col2 in range(8):
                         if abs(row-row2)<=1 and abs(col-col2)<=1 and (row != row2 or col != col2):
                            legal_moves.append({"from": (row, col), "to": (row2, col2)})
    return legal_moves

def apply_move_mask_complex(logits, legal_moves):
    """Applies a mask to the neural network output based on legal moves.
       The neural network output will consist of a probability for all possible locations
       for both origin and destination of the pieces.
       Returns a probability distribution over legal moves."""

    num_possible_moves = 64
    move_mask = np.full(num_possible_moves * num_possible_moves , -np.inf) # -inf for masking

    #loop through legal moves, and generate an index for the move in the probability space
    for move in legal_moves:
         from_row, from_col = move["from"]
         to_row, to_col = move["to"]
         
         from_index = coordinates_to_index(from_row,from_col)
         to_index = coordinates_to_index(to_row, to_col)
         move_index = from_index * 64 + to_index
         move_mask[move_index] = logits[move_index]

    masked_probabilities = softmax(move_mask)
    return masked_probabilities


def sample_move_complex(masked_probabilities, legal_moves):
    """Samples a move from the masked probability distribution, which is indexed by two locations.
       Returns a selected move dictionary (from, to)"""

    num_possible_moves = 64
    selected_index = np.random.choice(num_possible_moves * num_possible_moves, p=masked_probabilities)

    from_index = selected_index // 64
    to_index = selected_index % 64

    from_coords = index_to_coordinates(from_index)
    to_coords = index_to_coordinates(to_index)
    
    selected_move = {"from":from_coords, "to":to_coords}

     #find the index of this selected move in the original list
    legal_move_index = -1
    for index, move in enumerate(legal_moves):
        if move == selected_move:
            legal_move_index = index
            break
    
    return selected_move, legal_move_index


# Simulate one move using this new, complex representation
board = np.zeros((8,8))
legal_moves = generate_legal_moves_complex(board)
num_possible_moves = 64*64
logits = mock_network_output(num_possible_moves)

masked_probabilities = apply_move_mask_complex(logits, legal_moves)
selected_move, legal_move_index = sample_move_complex(masked_probabilities, legal_moves)
print(f"Legal move selected: {selected_move}, index in legal moves list: {legal_move_index}")

```
This illustrates how the same move masking technique is used with an alternative dictionary based representation of moves. It highlights the flexibility in choosing how to represent a move, and how to ensure consistency between the move mask, and move list.

Lastly, to incorporate the current player I have had to expand this even further. Note that the current player has impact on move legality:
```python
def generate_legal_moves_player(board, current_player):
    """Generates sample legal moves for a board position. Includes consideration of the current player."""

    legal_moves = []
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                for row2 in range(8):
                    for col2 in range(8):
                        if abs(row-row2)<=1 and abs(col-col2)<=1 and (row != row2 or col != col2):
                            # arbitrary condition making some moves only valid for player 1 and others for player 2
                            if current_player == 1 and row % 2 ==0 and col % 2 ==0:
                                 legal_moves.append({"from": (row, col), "to": (row2, col2)})
                            elif current_player == 2 and row % 2 == 1 and col % 2 == 1:
                                 legal_moves.append({"from": (row, col), "to": (row2, col2)})
    return legal_moves


def apply_move_mask_player(logits, legal_moves, current_player):
    """Applies a mask to the neural network output based on legal moves, considering the current player
    Returns a probability distribution over legal moves."""
    num_possible_moves = 64
    move_mask = np.full(num_possible_moves * num_possible_moves , -np.inf)

    for move in legal_moves:
          from_row, from_col = move["from"]
          to_row, to_col = move["to"]

          from_index = coordinates_to_index(from_row,from_col)
          to_index = coordinates_to_index(to_row, to_col)
          move_index = from_index * 64 + to_index
          move_mask[move_index] = logits[move_index]

    masked_probabilities = softmax(move_mask)
    return masked_probabilities

def sample_move_player(masked_probabilities, legal_moves):
   """Samples a move from the masked probability distribution, indexed by location.
     Returns the selected move (from, to) and the index of the move in the legal moves list.
   """
   num_possible_moves = 64
   selected_index = np.random.choice(num_possible_moves * num_possible_moves, p=masked_probabilities)

   from_index = selected_index // 64
   to_index = selected_index % 64

   from_coords = index_to_coordinates(from_index)
   to_coords = index_to_coordinates(to_index)

   selected_move = {"from":from_coords, "to":to_coords}

   legal_move_index = -1
   for index, move in enumerate(legal_moves):
       if move == selected_move:
           legal_move_index = index
           break
   return selected_move, legal_move_index



# Simulate one move including current player
board = np.zeros((8,8))
current_player = 1
legal_moves = generate_legal_moves_player(board, current_player)
num_possible_moves = 64*64
logits = mock_network_output(num_possible_moves)
masked_probabilities = apply_move_mask_player(logits, legal_moves, current_player)
selected_move, legal_move_index = sample_move_player(masked_probabilities, legal_moves)

print(f"Legal move selected (player {current_player}): {selected_move}, index: {legal_move_index}")

```

Here, `generate_legal_moves_player` takes the current player as a parameter and generates different legal moves depending on the player, while `apply_move_mask_player` and `sample_move_player` are modified to work with this new input. The masking technique remains the same, but the list of legal moves on which the masking is applied is modified per player.

Implementing a system like this requires careful consideration of performance bottlenecks. The move generation engine and the network forward pass should be as efficient as possible to allow for fast inference. I recommend examining techniques such as efficient indexing of move representations.

For further exploration, resources such as the "Chess Programming Wiki" provide a wealth of information regarding chess engine implementation details. Books dedicated to game playing AI also offer insights into various approaches of implementing move selection from neural networks. Furthermore, implementations of popular open-source chess engines (such as Stockfish) are an invaluable resource for examining real-world practices. Lastly, the documentation for machine learning libraries (such as TensorFlow and PyTorch) is crucial for building the neural network and applying the masking and move sampling logic.
