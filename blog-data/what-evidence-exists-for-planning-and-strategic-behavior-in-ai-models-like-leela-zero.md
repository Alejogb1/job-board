---
title: "What evidence exists for planning and strategic behavior in AI models like Leela Zero?"
date: "2024-12-11"
id: "what-evidence-exists-for-planning-and-strategic-behavior-in-ai-models-like-leela-zero"
---

Okay so you wanna know about planning and strategy in AIs like Leela Zero huh  That's a cool question  It's kinda tricky because we don't really have a solid definition of planning and strategy in the human sense  I mean we can see AIs achieving amazing results like Leela Zero dominating Go but is it actually *planning* or is it just really really good pattern recognition and search  It's a bit of a philosophical debate you know

One thing we can say is these AIs aren't explicitly programmed with strategic plans  They learn through self-play  Tons and tons of self-play generating a massive amount of data  Think of it like this a kid learning to play chess they don't start out with a grandmaster's strategy guide they just play a lot and get better  Leela Zero is kinda similar except it's playing millions of games and learning from every single one  The "strategy" emerges from all this data

So the evidence for planning well it's indirect  We see the results the stunningly effective game play  They don't just make random moves they build up advantages they adapt to different opponents  They even exhibit something that looks a lot like long-term planning like sacrificing pieces for positional advantage  It's not like they have a little roadmap in their code saying "step 1 control the center step 2 develop your pieces" but the emergent behavior looks remarkably like that

Think of it like a really complex ant colony  Individual ants don't have a master plan for building a nest but through local interactions and simple rules they create this complex and surprisingly effective structure  The AI's "strategy" emerges from the interaction of its many components and the vast amount of training data its seen  It's more like a swarm intelligence thing than explicit planning in the way humans might think of it

One thing that supports the idea of emergent strategy is the way these AIs handle uncertainty  They don't just calculate the best move based on the current board position  They consider all possible future scenarios and adjust their strategy based on the probabilities of different outcomes  This is kinda reminiscent of what humans do when we plan ahead we don't have perfect foresight so we try to anticipate what might happen and adjust our plans accordingly  Check out some papers on Monte Carlo Tree Search MCTS that's a crucial algorithm in AIs like AlphaGo and Leela Zero it deals directly with handling uncertainty

Now lets get into some code  I'll try to make it simple  This is really just pseudocode to get the general idea across  Real implementations are way more complex



```python
# Simple example of a decision tree for a game
# Not a true MCTS but illustrates some key concepts
def choose_move(board, depth):
  if depth == 0 or game_over(board):
    return evaluate_board(board)  #Heuristic evaluation

  best_move = None
  best_score = -float('inf')

  for move in possible_moves(board):
    new_board = make_move(board, move)
    score = choose_move(new_board, depth - 1)
    if score > best_score:
      best_score = score
      best_move = move

  return best_move

```

This is a super simplified decision tree  It just explores a few moves ahead evaluating the board situation at each step  Real AI implementations are far more sophisticated they use MCTS to explore a much wider range of possibilities  Instead of a simple evaluation they use neural networks to estimate the value of different board positions

Here's another snippet illustrating a different aspect  This one touches on how neural networks might be used to evaluate positions



```python
#Illustrative example of a neural network evaluation
import numpy as np

#Simplified neural network representation
class NeuralNetwork:
  def __init__(self):
    self.weights = np.random.rand(100,1) #random weights

  def predict(self, board_features):
    #Simplified prediction - real networks are way more complex
    return np.dot(board_features, self.weights)

# Example usage
nn = NeuralNetwork()
board_features = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1]) # Example board features
evaluation = nn.predict(board_features)
print(evaluation) #The evaluation given by the neural network
```

This is a super super simplified neural network  In real life these things have millions of parameters and are trained on huge datasets  But the basic idea is to use the network to estimate the win probability or some other value associated with a given board state


And finally a bit of code hinting at how self-play works

```python
#Conceptual self-play loop
def self_play(model):
    board = initial_board()
    while not game_over(board):
        move = model.choose_move(board)
        board = make_move(board,move)

    return game_result(board) # 1 for win -1 for loss etc


#Training loop
for i in range(10000):
  result = self_play(model)
  model.update(result) #update model parameters using backpropagation
```

Again this is a simplified representation  In real systems the model update part is incredibly complex involving techniques like reinforcement learning  But the core concept is the same  The AI plays against itself repeatedly learning and improving its strategy with every game


So to wrap it all up  The evidence for planning and strategic behavior in AIs like Leela Zero is largely indirect  We don't see explicit plans but the emergent behavior is strikingly similar to what we see in human strategic thinking  MCTS and neural networks are key components that help to create this emergent strategy  It's not exactly like human planning but it's impressive nonetheless  If you want to dig deeper I'd recommend looking at some papers on MCTS reinforcement learning and game AI  There are some great books on AI too but honestly  lots of the cutting-edge stuff is in recent papers
