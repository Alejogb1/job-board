---
title: "How to do Bidirectional GRU from keras weight to inference in c?"
date: "2024-12-15"
id: "how-to-do-bidirectional-gru-from-keras-weight-to-inference-in-c"
---

alright, so you're trying to take a keras bidirectional gru model and get it running inference in c. i've been down that road more than once, and it's definitely a place where things can get… interesting. let me walk you through what i've learned, focusing on a practical approach that avoids most of the deep-dive theory.

first off, the core issue isn't really about bidirectional grus themselves, it's about bridging the gap between a high-level framework like keras and a low-level language like c. keras is fantastic for building and training models but it abstracts away all the low-level matrix math. c, on the other hand, requires you to do everything by hand. this means you’ll need to extract the weights from your keras model, then manually implement the gru computations in c, taking into account both forward and backward directions.

when i first tackled this, back when i was working on an embedded audio processing project (i had to perform speaker recognition on a potato-powered microcontroller), i messed up the weight ordering and got gibberish as output for a week. then i realized the weights were in a different order in memory from what i had assumed. classic.

let’s start with the keras side. you need to save the weights from your trained model. the simplest way is probably to use `model.get_weights()` after model creation. this returns a list of numpy arrays and it's crucial to understand the order and meaning of these weights and biases, and the associated layers.

```python
import numpy as np
import tensorflow as tf

# assume you have a trained keras model called 'model'
# this is an example, you need to adapt this to your model
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128), input_shape=(None, 32)),
    tf.keras.layers.Dense(units=10)
])

# random data to infer.
dummy_data = np.random.rand(1, 10, 32).astype(np.float32)
model.predict(dummy_data)

weights = model.get_weights()

for i, w in enumerate(weights):
    print(f"layer {i}, shape {w.shape}")

#save weights as numpy for later use
np.savez("model_weights.npz", *weights)


```

running this, you'll get the shape of each layer which is very important since you'll need them later. `np.savez` can save multiple numpy arrays into a single file, which is more convenient. this file, `model_weights.npz`, will be your primary source of info when moving to the c world.

now, the core of the work lies in replicating the gru’s behavior in c. if you are not sure how a gru works you can check some standard textbook. “neural networks and deep learning” by michael nielsen or deep learning by goodfellow et al. are nice resources to check the gory details and all the formulas.

let me give you a simplified example of how you would structure your c code, focusing on the basic math. you'll need to handle the weights load, the matrix multiplications and the proper activation functions:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// assuming float is your precision
typedef float scalar;


// basic matrix multiplication function (you'll need a more efficient one for real work)
void matmul(scalar *a, scalar *b, scalar *c, int a_rows, int a_cols, int b_cols) {
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            c[i * b_cols + j] = 0.0;
            for (int k = 0; k < a_cols; k++) {
                c[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }
}

// sigmoid function
scalar sigmoid(scalar x) {
    return 1.0 / (1.0 + exp(-x));
}

// tanh function
scalar tanh_func(scalar x) {
    return tanh(x);
}

// the core of the gru cell implementation in c (simplified).
void gru_cell(scalar *input, scalar *h_prev, scalar *wz, scalar *wr, scalar *wh, scalar *bz, scalar *br, scalar *bh, scalar *h_output, int hidden_size, int input_size) {
  // temporary matrices.
  scalar *z_t = malloc(hidden_size * sizeof(scalar));
  scalar *r_t = malloc(hidden_size * sizeof(scalar));
  scalar *h_tilde = malloc(hidden_size * sizeof(scalar));

  scalar *input_wz = malloc(hidden_size * input_size * sizeof(scalar));
  scalar *input_wr = malloc(hidden_size * input_size * sizeof(scalar));
  scalar *input_wh = malloc(hidden_size * input_size * sizeof(scalar));

  scalar *prev_h_wz = malloc(hidden_size * hidden_size * sizeof(scalar));
  scalar *prev_h_wr = malloc(hidden_size * hidden_size * sizeof(scalar));
  scalar *prev_h_wh = malloc(hidden_size * hidden_size * sizeof(scalar));

  // copy weights in
  for (int i=0; i < hidden_size * input_size; i++){
    input_wz[i] = wz[i];
    input_wr[i] = wr[i];
    input_wh[i] = wh[i];
  }
  for (int i=0; i < hidden_size * hidden_size; i++){
    prev_h_wz[i] = wz[hidden_size * input_size + i];
    prev_h_wr[i] = wr[hidden_size * input_size + i];
    prev_h_wh[i] = wh[hidden_size * input_size + i];
  }


    // update gate z_t.
    scalar *input_part_z = malloc(hidden_size * sizeof(scalar));
    scalar *h_prev_part_z = malloc(hidden_size * sizeof(scalar));
    matmul(input, input_wz, input_part_z, 1, input_size, hidden_size);
    matmul(h_prev, prev_h_wz, h_prev_part_z, 1, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        z_t[i] = sigmoid(input_part_z[i] + h_prev_part_z[i] + bz[i]);
    }

    free(input_part_z);
    free(h_prev_part_z);


    // reset gate r_t
    scalar *input_part_r = malloc(hidden_size * sizeof(scalar));
    scalar *h_prev_part_r = malloc(hidden_size * sizeof(scalar));
    matmul(input, input_wr, input_part_r, 1, input_size, hidden_size);
    matmul(h_prev, prev_h_wr, h_prev_part_r, 1, hidden_size, hidden_size);


    for (int i = 0; i < hidden_size; i++) {
        r_t[i] = sigmoid(input_part_r[i] + h_prev_part_r[i] + br[i]);
    }
    free(input_part_r);
    free(h_prev_part_r);

    // candidate hidden state h_tilde

    scalar *input_part_h = malloc(hidden_size * sizeof(scalar));
    scalar *h_prev_part_h = malloc(hidden_size * sizeof(scalar));
    scalar *h_prev_temp = malloc(hidden_size * sizeof(scalar));
    for (int i = 0; i < hidden_size; i++) {
      h_prev_temp[i] = h_prev[i] * r_t[i];
    }

    matmul(input, input_wh, input_part_h, 1, input_size, hidden_size);
    matmul(h_prev_temp, prev_h_wh, h_prev_part_h, 1, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        h_tilde[i] = tanh_func(input_part_h[i] + h_prev_part_h[i] + bh[i]);
    }

    free(input_part_h);
    free(h_prev_part_h);
    free(h_prev_temp);

    // final hidden state h_t
    for(int i=0; i < hidden_size; i++){
        h_output[i] = (1.0 - z_t[i]) * h_prev[i] + z_t[i] * h_tilde[i];
    }

    //free all matrices in this scope
    free(z_t);
    free(r_t);
    free(h_tilde);
    free(input_wz);
    free(input_wr);
    free(input_wh);
    free(prev_h_wz);
    free(prev_h_wr);
    free(prev_h_wh);

}


int main() {
    // here we pretend we have loaded our weights from npz file
    // load weights from .npz file in the correct order
    // using python or any other language, since it's outside the scope of c
    // for demonstration purposes, generate random numbers:
    // this is not your model weights, replace this with real weights
    int hidden_size = 128;
    int input_size = 32;
    int sequence_length = 10;

    //gru layer
    scalar* wz_f = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* wr_f = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* wh_f = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* bz_f = malloc(hidden_size * sizeof(scalar));
    scalar* br_f = malloc(hidden_size * sizeof(scalar));
    scalar* bh_f = malloc(hidden_size * sizeof(scalar));

    scalar* wz_b = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* wr_b = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* wh_b = malloc((hidden_size * (input_size + hidden_size)) * sizeof(scalar));
    scalar* bz_b = malloc(hidden_size * sizeof(scalar));
    scalar* br_b = malloc(hidden_size * sizeof(scalar));
    scalar* bh_b = malloc(hidden_size * sizeof(scalar));

    //dense layer
    scalar *dense_w = malloc(hidden_size * 2 * 10 * sizeof(scalar));
    scalar *dense_b = malloc(10 * sizeof(scalar));


    for(int i=0; i< hidden_size * (input_size + hidden_size); i++){
      wz_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      wr_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      wh_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      wz_b[i] = (scalar)rand()/(scalar)RAND_MAX;
      wr_b[i] = (scalar)rand()/(scalar)RAND_MAX;
      wh_b[i] = (scalar)rand()/(scalar)RAND_MAX;
    }

     for(int i=0; i < hidden_size; i++){
      bz_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      br_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      bh_f[i] = (scalar)rand()/(scalar)RAND_MAX;
      bz_b[i] = (scalar)rand()/(scalar)RAND_MAX;
      br_b[i] = (scalar)rand()/(scalar)RAND_MAX;
      bh_b[i] = (scalar)rand()/(scalar)RAND_MAX;
    }
    for(int i=0; i < hidden_size * 2 * 10; i++){
      dense_w[i] = (scalar)rand()/(scalar)RAND_MAX;
    }
    for(int i=0; i < 10; i++){
        dense_b[i] = (scalar)rand()/(scalar)RAND_MAX;
    }

    // example input tensor
    scalar input_tensor[sequence_length * input_size];
    for(int i=0; i < sequence_length * input_size; i++)
        input_tensor[i] = (scalar)rand()/(scalar)RAND_MAX;

    scalar *h_forward = malloc(hidden_size * sizeof(scalar));
    scalar *h_backward = malloc(hidden_size * sizeof(scalar));
    scalar *output = malloc(hidden_size * 2 * sizeof(scalar));

    // forward pass.
    for(int i = 0; i < hidden_size; i++) {
      h_forward[i] = 0.0;
    }
    for (int t = 0; t < sequence_length; t++){
        scalar *current_input = &input_tensor[t * input_size];
        scalar *new_h_forward = malloc(hidden_size * sizeof(scalar));
        gru_cell(current_input, h_forward, wz_f, wr_f, wh_f, bz_f, br_f, bh_f, new_h_forward, hidden_size, input_size);
        for (int i = 0; i < hidden_size; i++){
            h_forward[i] = new_h_forward[i];
        }
        free(new_h_forward);
    }

    // backward pass.
     for(int i = 0; i < hidden_size; i++) {
      h_backward[i] = 0.0;
    }
    for (int t = sequence_length -1; t >= 0 ; t--){
        scalar *current_input = &input_tensor[t * input_size];
        scalar *new_h_backward = malloc(hidden_size * sizeof(scalar));
        gru_cell(current_input, h_backward, wz_b, wr_b, wh_b, bz_b, br_b, bh_b, new_h_backward, hidden_size, input_size);
        for (int i = 0; i < hidden_size; i++){
            h_backward[i] = new_h_backward[i];
        }
        free(new_h_backward);
    }
    //concatenate h_forward and h_backward
    for (int i = 0; i < hidden_size; i++){
        output[i] = h_forward[i];
        output[i + hidden_size] = h_backward[i];
    }

    //dense layer
    scalar dense_output[10];
    scalar *matmul_output = malloc(10*sizeof(scalar));
    matmul(output, dense_w, matmul_output, 1, hidden_size * 2, 10);

    for(int i=0; i < 10; i++){
        dense_output[i] = matmul_output[i] + dense_b[i];
    }

    // output
    printf("output from last layer:\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", dense_output[i]);
    }
    printf("\n");
    // free memory
    free(h_forward);
    free(h_backward);
    free(output);
    free(dense_w);
    free(dense_b);
    free(matmul_output);
    free(wz_f);
    free(wr_f);
    free(wh_f);
    free(bz_f);
    free(br_f);
    free(bh_f);
     free(wz_b);
    free(wr_b);
    free(wh_b);
    free(bz_b);
    free(br_b);
    free(bh_b);
    return 0;
}
```

this c code will be very slow, because the `matmul` is not optimized, you should use some matrix library like openblas. but for clarity, i leave it as is.

this code demonstrates a simple gru cell. you'll need to iterate through your input sequence, passing the hidden state from each step, in both forward and backward directions. finally, the bidirectional part needs concatenation, and then, you will feed this concatenated output into the following dense layer which you have to implement as well. it's like a relay race, each step taking the baton from the previous.

**important details:**
 * **weight order:** the `get_weights()` function returns a flat list of arrays. you will have to manually inspect its structure and figure out where the biases, weights for forward and backward direction, etc. are and how to map them to your c implementation. print the shapes as seen in the python example to help you in the process.
* **matrix multiplication optimization:** as i mentioned before, the `matmul` function i presented is basic. for real projects, you should use highly optimized libraries like openblas or intel mkl to get any kind of reasonable speed.
* **memory management:** dynamic memory allocation in c is tricky and you have to handle all memory allocation and dealocation by yourself, always free what you allocate, it can become a memory leak nightmare easily otherwise.
* **activation functions:** make sure your c implementation of the activation functions (sigmoid, tanh) is accurate, and be careful about numerical stability.
* **bidirectional specifics:** remember that you need to process the input sequence once in the forward direction and again in reverse, keeping track of the hidden states.

this is a challenging task and is a bit like assembling a puzzle where the pieces are matrices and the instruction manual is a collection of equations. and it's quite easy to get some weight or bias mixed up and get gibberish output, like i did in the potato microcontroller project. (it turned out the potato was also not helping that much). the most time i had spent in that was just debugging the proper index and weight organization.

remember to start simple, verify each step separately, and good luck! you have a lot of coding to do but it's definitely achievable with enough patience and careful attention to detail. there’s plenty of resources out there if you need more theory. you could also look up “recurrent neural networks with python” by long short-term memory or “understanding lstm networks” by christopher olah if you want the really granular detail on recurrent neural networks.
