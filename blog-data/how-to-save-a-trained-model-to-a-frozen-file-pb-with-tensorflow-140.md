---
title: "How to Save a trained model to a frozen file (.pb) with Tensorflow 1.4.0?"
date: "2024-12-15"
id: "how-to-save-a-trained-model-to-a-frozen-file-pb-with-tensorflow-140"
---

alright, so you're trying to save a tensorflow model as a frozen graph in a .pb file, and you're on tensorflow 1.4.0. i’ve been there. it's not exactly straightforward, especially with the older versions. i remember banging my head against the wall for a good couple of days trying to get this working with a custom network back in, oh, probably 2018? i was working on a project that involved image segmentation, and i needed to deploy the model on an embedded device which didn't exactly play nice with the dynamic graph nature of tensorflow at the time.

the key thing to understand is that a frozen graph is a single file that contains your model’s architecture (the computational graph), and the values of all the trained weights. this is different from saving checkpoints. checkpoints only store the weights and not the actual graph structure and you need to have a graph already defined somewhere to restore them, usually in a training script. a frozen graph, on the other hand, is self-contained. this makes it very portable and easy to deploy.

tf 1.4.0, like many previous versions, had this somewhat convoluted process to save to a frozen graph. basically, you first need to save your graph definition as a .meta file and your model weights to checkpoint files, these are usually named like `.data-00000-of-00001`, `.index`, `.meta` files and then extract the necessary ops and weights and create a frozen graph from it.

let’s break it down step by step, and i'll give you some code snippets that should help.

first, let's assume you have a trained model. usually, your model is constructed something like this:

```python
import tensorflow as tf

# assuming you have defined your model architecture before this
# here's a dummy example
graph = tf.Graph()
with graph.as_default():
    input_tensor = tf.placeholder(tf.float32, shape=(None, 784), name='input')
    hidden_layer = tf.layers.dense(input_tensor, 128, activation=tf.nn.relu)
    output_tensor = tf.layers.dense(hidden_layer, 10, name='output')

    # Define the loss and optimizer as you normally would in your model
    labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_tensor)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # initialize variables, etc
    init = tf.global_variables_initializer()

    # you could also specify a saver instance if you want to save checkpoints to restore the model
    saver = tf.train.Saver()

# you have run your training before this with the graph 'graph'
with tf.Session(graph=graph) as sess:
    sess.run(init)
    # you do the training here
    # then
    # suppose you trained and reached your desired result here
    # now save the checkpoint:
    saver.save(sess, "path/to/your/model.ckpt") # save checkpoint files
```

, this is your standard model training process and now you have your model weights saved as checkpoints in the `path/to/your/model.ckpt` location. the goal now is to transform all this information into a single frozen graph .pb file that you can deploy.

here is how you would create the frozen graph for your model, it is basically a function that accepts your model, the path where your checkpoint files are, the name of the final operation (usually the output of the model) and path to save the frozen graph.

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(model_dir, output_node_names, frozen_graph_name):
    """
    Freezes the state of a session into a pruned graph def.
    """

    with tf.Session(graph=tf.Graph()) as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(model_dir + '.meta')

        # Restore the weights
        saver.restore(sess, model_dir)


        # Convert variables to constants and remove not needed operations
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names.split(",")
        )

        # write the frozen graph
        with tf.gfile.GFile(frozen_graph_name, "wb") as f:
                f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        return output_graph_def
```

you use this function like this:

```python
# Assuming you trained your model and saved checkpoint files into 'path/to/your/model.ckpt'
# Define the output node name
output_node = 'output/BiasAdd'

# this will create path/to/your/frozen_model.pb file
freeze_graph('path/to/your/model.ckpt', output_node,'path/to/your/frozen_model.pb')
```

notice that `output/BiasAdd` was the output node for the example neural network in the previous code snippet. you will need to adjust the output node name depending on your model output name. usually if you did not specify an output name or output operation it defaults to `<your layer name>/BiasAdd` or `/Relu` or `/Sigmoid` and so on. this will depend on the activation function of your final layer, in the example it's a linear layer so it will end with `BiasAdd`.  use `tensorboard` to check for the right operation name, it is usually the output of the last layer.

the `freeze_graph` function basically loads the meta graph which contains your graph definition, then it loads the weights from your checkpoint file, and finally it uses the utility `graph_util.convert_variables_to_constants` which creates a frozen graph with variables as constant and removes some unnecesary operations that are not needed to make inferences. finally, it saves it to a pb file.

this process creates a file `path/to/your/frozen_model.pb` which now contains the entire model graph and weights. now, if you have a model that has different outputs or nodes of interest, you need to specify them comma separated like this: `output1/BiasAdd,output2/Relu,output3/Sigmoid`. also, there can be nodes in your graph that are intermediary results, and you might want to output them too, which you can do by including them in that same comma-separated list of node names.

now, when you want to load this frozen graph to make predictions, here's how you'd do it:

```python
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # load frozen graph
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Import graph and set as default
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name='') # this name argument can be any name
    return graph

# load the graph from the frozen file
graph = load_graph('path/to/your/frozen_model.pb')

# find the input and output operations by name
input_tensor_name = 'input:0' # you have to specify the ':0' at the end
output_tensor_name = 'output/BiasAdd:0'

# start a session to execute it
with tf.Session(graph=graph) as sess:
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    # use some dummy data (replace this with your actual input data)
    import numpy as np
    dummy_input = np.random.rand(1,784).astype(np.float32)

    # make a prediction
    predictions = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
    print(predictions)
```

this code loads the frozen graph from the .pb file, gets the input and output tensors, and makes a prediction. remember to replace `"input:0"` and `"output/BiasAdd:0"` with the actual names of your input and output tensors. they can be seen using a program such as `netron` or by loading your graph into tensorboard.

and that's it. it’s a bit of a roundabout process but it works. you'll probably find this process easier in newer versions of tensorflow. in the newer versions it’s much cleaner and you don’t need to extract it, you can simply save it using a tensorflow saver but with the `.pb` extension or using tensorflow `saved_model` module which is easier but more complicated for the older versions. also the function `tf.graph_util.convert_variables_to_constants` has been deprecated and moved to `tf.compat.v1.graph_util.convert_variables_to_constants` in newer versions of tensorflow. but this works for 1.4.0 as requested, so no need to change the code here.

for resources, i would recommend going through the tensorflow documentation for the `graph_util` module, and maybe some older blog posts or tutorials on saving frozen graphs for tensorflow 1.x. there aren't any specific books for older versions. also, i found that the “hands-on machine learning with scikit-learn, keras & tensorflow” by aurélien géron is good for a broader understanding of building machine learning pipelines and model deployment, though it focuses on more modern tensorflow.

lastly, remember that debugging this can be tricky. tensorflow errors are usually not very descriptive. so if it’s not working, double check your output node names. and remember what i always tell my coworkers: "the problem is never the tensorflow code, it's always the user" ( just kidding, sometimes it is the code).
