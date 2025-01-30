---
title: "How can a Markov chain be reconstructed from a figure in R?"
date: "2025-01-30"
id: "how-can-a-markov-chain-be-reconstructed-from"
---
The reconstruction of a Markov chain from a visual representation, specifically a directed graph figure generated in R, necessitates a careful interpretation of the graph's structure and its underlying numerical relationships. I've encountered this challenge frequently during model validation and process visualization, where a graph, often the product of someone else’s work, serves as the only available representation of an underlying Markov process. The core task lies in extracting the transition probabilities – the numeric probabilities of moving from one state to another – from the visual arrangement of nodes and edges.

The process hinges on the observation that a directed graph of a Markov chain inherently displays the *states* as nodes (typically visually represented as circles or boxes) and the transitions between them as directed edges (arrows). The weight associated with each transition, or the probability, is implicit in the structure of the graph. A key constraint, crucial for validation, is that for any given state, the sum of probabilities emanating from that state across all possible outgoing transitions must equal one. This characteristic forms the fundamental mathematical definition of a Markov chain, ensuring that probability mass is always conserved in state transitions.

The primary difficulty in reconstructing a chain stems from the fact that probabilities are rarely directly written onto the graph itself, and there is seldom any explicit numerical data accompanying it, especially for a figure embedded within an external report. In most scenarios, the probabilities are implicitly encoded either through edge thickness or through the graph layout itself, as is the practice in some visualization libraries. Therefore, the reconstruction process involves identifying these implicit encodings and extracting them through systematic inspection and calculation. The procedure effectively comprises three stages: state identification, transition mapping, and probability estimation.

First, the states must be identified. This involves discerning which nodes are separate and valid state representations within the system. Often, especially in complicated visualisations, there are additional nodes which do not have a functional purpose within the Markov chain. I often find this is the case with figures designed for visual appeal over technical clarity. In essence, if a node does not have outgoing edges or incoming edges that contribute to the flow of probability, it can likely be disregarded. The states that do have connected edges form the basis of our chain. We can assign arbitrary integer identifiers to these states; the actual values are not critical, just that we create a mapping to them from the visual layout of the graph.

Next, the graph is mapped into a set of transition pairs. Each edge leading out of a state and into another corresponds to a transition from the first to the second. We record all such directed connections, forming an exhaustive list. For each state, we also need to identify all of its outgoing edges.

Finally, comes probability estimation. Here is where the true technical challenge is faced. There are two common ways probabilities can be encoded in the figure. First, edge thickness is sometimes used, whereby the width of a given arrow representing a transition is linearly scaled to the probability (thicker arrow = higher probability). In these cases we need to use our subjective estimation to measure the width and create some numerical representation. Second, relative placement is sometimes used, especially in force-directed graphs: transitions with more probability will attract other nodes more strongly, giving a sense of proportion. We need to identify from the overall graph design how probability is encoded, and then reverse engineer a numerical mapping. This may require multiple manual attempts and iterations.

Here are three examples based on some figures I’ve worked with:

**Example 1: Simple State Chain (Edge Thickness Encoding)**

Assume we have a graph with three states (A, B, C), where edges connect A->B, B->C, and C->A. Let’s say the edge A->B is thick, B->C is medium thickness, and C->A is thin. After assigning integers to the states (A=1, B=2, C=3), we record the transitions: (1,2), (2,3), and (3,1). Estimating the thicknesses by visual examination, we might say the thicknesses correspond to 50%, 30%, and 20% respectively. We create a transition matrix, scaling the probability of state transitions appropriately. The output should sum to one for each source state, therefore, we have

```R
transition_matrix <- matrix(0, nrow = 3, ncol = 3)  # Initialize matrix

# Populate the matrix with the transitions and their estimates.
# Note, for these illustrative purposes we're assuming our visual estimates are
# reasonable approximations for relative thickness of edge (ie: 50% is the largest,
# 20% is the smallest)
transition_matrix[1, 2] <- 0.5
transition_matrix[2, 3] <- 0.3
transition_matrix[3, 1] <- 0.2

#We need to normalise for outgoing edges, to ensure each sum to 1:

rowSums <- rowSums(transition_matrix)

for(i in 1:nrow(transition_matrix)){
    if(rowSums[i] > 0){
        transition_matrix[i,] <- transition_matrix[i,] / rowSums[i]
    }
}

print(transition_matrix)
```

The matrix `transition_matrix` now contains the estimated transition probabilities, with un-connected states having a probability of zero. Row i represents starting in state i, while the columns represent ending in state x.

**Example 2: Self Loops and Multiple Transitions (Edge Thickness Encoding)**

Suppose the graph has two states (X, Y), with edges X->X (a self loop), X->Y, Y->X, and Y->Y (self loop). Assume X->X has medium thickness, X->Y is thin, Y->X is thick, and Y->Y is medium thickness. State labels (X=1, Y=2). Transitions: (1,1), (1,2), (2,1), (2,2). We observe our thicknesses, scaling appropriately:

```R
transition_matrix <- matrix(0, nrow = 2, ncol = 2)  # Initialize matrix

# Populate the matrix with the transitions and their estimates.
# Again, our visual estimates are used.
transition_matrix[1, 1] <- 0.3
transition_matrix[1, 2] <- 0.1
transition_matrix[2, 1] <- 0.5
transition_matrix[2, 2] <- 0.4

#Normalise
rowSums <- rowSums(transition_matrix)

for(i in 1:nrow(transition_matrix)){
    if(rowSums[i] > 0){
        transition_matrix[i,] <- transition_matrix[i,] / rowSums[i]
    }
}

print(transition_matrix)
```

Here, the key thing is properly accounting for the self-loops, as they are just as valid a transition as any other and must be incorporated into our state normalisation logic.

**Example 3: Force Directed Graph (Layout Position Encoding)**

This example presents a harder challenge. We have a graph with three states (P, Q, R). The force-directed graph positions P somewhat close to Q, R is distant from both P and Q. Assume the visual placement suggests the transition from P->Q is strong, P->R weak, Q->P strong, Q->R weak, and transitions back to the same state (e.g. P->P) are not explicitly visualized, implying zero transition probability. State Labels (P=1, Q=2, R=3), Transitions (1,2), (1,3), (2,1), (2,3). Note the absense of all self loops.

```R
transition_matrix <- matrix(0, nrow = 3, ncol = 3)  # Initialize matrix

# The challenge here is to map the relative distances between nodes into
# probabilities. This is subjective. Lets assume distance is inversely proportional
# to probability. P and Q are close, so transition (1,2) is high. R is very far
# from P, so transition (1,3) is low.

transition_matrix[1, 2] <- 0.6
transition_matrix[1, 3] <- 0.1
transition_matrix[2, 1] <- 0.5
transition_matrix[2, 3] <- 0.2

#Normalise
rowSums <- rowSums(transition_matrix)

for(i in 1:nrow(transition_matrix)){
    if(rowSums[i] > 0){
        transition_matrix[i,] <- transition_matrix[i,] / rowSums[i]
    }
}

print(transition_matrix)
```

Here, the probabilities are significantly less precise because there is no direct mapping to numerical values. This example shows the inherent ambiguity in relying on graph layouts as a probability encoding scheme.

For more detail on graph manipulation in R, resources focusing on network analysis packages such as igraph and networkD3 are recommended. Further study into the theoretical framework of discrete-time Markov chains provides essential context for validating the reconstructed models. In cases where precise numerical values are critical, efforts should be made to obtain the original data sources instead of relying on visual representation. A deep understanding of these concepts and their application has been invaluable in my career when having to decipher other people's modelling outputs, and I have often had to reconstruct many such models from the graphical representation alone.
