---
title: "Why is a Keras LSTM performance terrible when using recurrent prediction?"
date: "2024-12-15"
id: "why-is-a-keras-lstm-performance-terrible-when-using-recurrent-prediction"
---

so, you're seeing a keras lstm choke when doing recurrent prediction, eh? i've been there, many times. it's a common pitfall and there are a few usual suspects. let's break this down. first off, when we talk about 'recurrent prediction' with an lstm, we're talking about feeding the output of the lstm back into itself as input for the next time step, and you have a sequence of inputs and outputs that are dependent on each other. this is very different than just doing a one-shot prediction based on an input. it's much more challenging for the model.

one thing i've noticed over my years building these things, is that people new to this field often assume that the model will just *figure it out* magically when you change the architecture from simple prediction to recurrent prediction but this is wrong assumption.

let's get into specific reasons and some code snippets.

*   **vanishing/exploding gradients:** this is a classic neural network problem, but it’s particularly relevant to recurrent networks. when gradients become really small, learning slows down, and when they become very large, it creates instability. during recurrent prediction, you're essentially chaining multiple computations together, so gradients can shrink or explode quickly across the many time steps. this makes the model hard to train. lstms are meant to mitigate it, but they don't eliminate it completely. you must watch out that all the hidden states are ok and well propagated and there are no unexpected behaviour.

    here's how i often catch this in training: look at the loss. if you see it plateauing early, or jumping around wildly, it can be a sign that gradients are not well behaved. if the error is too high, then it means that it's not learning anything. also, plot the gradients of the different layers in your model. keras makes this easier with the `tf.GradientTape`. this helped me find the problem on my first major lstm project building a machine translation model. i was trying to do something similar to sequence to sequence translation using recurrent input in 2017. at the time tensorflow had a very buggy api, and the debugging was a nightmare. i was using a very old nvidia card, and i thought that the problem was the graphics card. until i found out that the problem was the gradient explosion. it turns out that the hidden states initialization was incorrect. this lead to big instabilities in the training.
   
    here's code to monitor gradient norms:

    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Model
    import numpy as np

    # let's generate some dummy data
    timesteps = 10
    input_dim = 5
    units = 32
    batch_size = 64
    num_samples = 1000

    x_train = np.random.rand(num_samples, timesteps, input_dim)
    y_train = np.random.rand(num_samples, timesteps, input_dim)

    inputs = tf.keras.Input(shape=(timesteps, input_dim))
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    outputs = Dense(input_dim)(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, gradients
   
    for epoch in range(10):
        for i in range(0, num_samples, batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            loss, gradients = train_step(batch_x, batch_y)

            gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
            
            print(f"epoch {epoch} batch {i} loss: {loss:.4f}, max grad norm: {max(gradient_norms) if gradient_norms else 'no gradients'}")
    ```

    this script provides a way to see the maximum gradient magnitude in each batch during training. a very high value here can indicate a gradient explosion, while very low values can mean that the learning rate may be too low.

*   **incorrect state management:** the 'memory' of an lstm is stored in its hidden and cell states. when you're doing recurrent prediction, you need to carefully track and pass these states from one prediction step to the next. if you mess this up, the model will lose its context and will not have the past information. this is something that also happened to me building a chatbot using lstm many years ago. it worked fine at small sequence lengths, but when i tried to extend the sequence, it suddenly started to answer with nonsense or repeating the same phrases all the time. the problem was how i was initializing and passing the hidden states.

    here is some code with a method to show a problem example:
   
   ```python
   import tensorflow as tf
   from tensorflow.keras.layers import LSTM, Dense
   from tensorflow.keras.models import Model
   import numpy as np

   timesteps = 10
   input_dim = 5
   units = 32

   # create a very simple model
   inputs = tf.keras.Input(shape=(1, input_dim)) # now input is one step at the time
   lstm_layer = LSTM(units, return_state=True) # return states
   lstm_out, state_h, state_c = lstm_layer(inputs)
   outputs = Dense(input_dim)(lstm_out)
   model = Model(inputs=inputs, outputs=[outputs, state_h, state_c])

   # lets create input for one prediction step
   initial_input = np.random.rand(1, 1, input_dim) # one single step
   # get hidden states from the initial step
   prediction, state_h, state_c = model.predict(initial_input)

   # perform a number of recurrent predictions:
   recurrent_predictions = []
   current_input = initial_input
   num_steps = 15

   for _ in range(num_steps):
       prediction, state_h, state_c = model.predict(current_input)
       recurrent_predictions.append(prediction)
       # here is the problem, passing the last prediction as input
       current_input = prediction.reshape(1, 1, input_dim)
   
   print(f"recurrent prediction using incorrect input: {np.array(recurrent_predictions).shape}")


   # corrected code

   inputs = tf.keras.Input(shape=(1, input_dim)) # now input is one step at the time
   lstm_layer = LSTM(units, return_state=True) # return states
   lstm_out, state_h, state_c = lstm_layer(inputs)
   outputs = Dense(input_dim)(lstm_out)
   model_recurrent = Model(inputs=inputs, outputs=[outputs, state_h, state_c])


   # let's create a state model to do recurrent
   state_input_h = tf.keras.Input(shape=(units,))
   state_input_c = tf.keras.Input(shape=(units,))
   lstm_out_r, state_h_r, state_c_r = lstm_layer(inputs, initial_state=[state_input_h,state_input_c])
   output_r = Dense(input_dim)(lstm_out_r)
   model_recurrent_state = Model(inputs=[inputs, state_input_h, state_input_c], outputs=[output_r, state_h_r, state_c_r])


   # lets create input for one prediction step
   initial_input = np.random.rand(1, 1, input_dim) # one single step
   # get hidden states from the initial step
   prediction, state_h, state_c = model_recurrent.predict(initial_input)


   # correct state passing
   recurrent_predictions = []
   current_input = initial_input
   for _ in range(num_steps):
    prediction, state_h, state_c = model_recurrent_state.predict([current_input, state_h, state_c])
    recurrent_predictions.append(prediction)
    current_input = np.random.rand(1, 1, input_dim) # you can put the prediction or use new input

   print(f"recurrent prediction using correct input and states: {np.array(recurrent_predictions).shape}")

   ```
   as you can see the first section of the code shows an example with incorrect state passing. it uses the last prediction as an input, but this is wrong because it doesn't pass the hidden states. the second example shows the correct way to perform this, using a state passing model that correctly passes the hidden states.
  
*   **overfitting to training sequences:** another common issue, particularly when the training data is small or repetitive, is the model becomes good at memorizing the training sequences. but performs badly on new sequences. this happens during recurrent prediction because the model can pick up patterns specific to the training data that are not applicable to unseen data. this happened to me on a time series prediction project. it would predict the training data perfectly, but the predictions would be totally off when i used new data. at the time i used regularization as a way to fix it. also adding gaussian noise helped a lot.
   
    here's some code illustrating this:

    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import numpy as np

    # let's create some toy data
    timesteps = 10
    input_dim = 5
    units = 32
    batch_size = 32
    num_samples = 500

    x_train = np.random.rand(num_samples, timesteps, input_dim)
    # generating y values based on a simple function for simplicity
    y_train = np.roll(x_train, shift=1, axis=1) # a very easy sequence to predict
    

    # base model with overfitting issue
    inputs = tf.keras.Input(shape=(timesteps, input_dim))
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    outputs = Dense(input_dim)(lstm_out)
    model_overfit = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    model_overfit.compile(optimizer=optimizer, loss=loss_fn)
    model_overfit.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=0)

    # predict some data to show the overfitting in training data
    train_prediction = model_overfit.predict(x_train)
    print(f"overfitting error in train data: {loss_fn(train_prediction, y_train).numpy():.4f}")
    
    # creating new data to show the poor performance
    x_test = np.random.rand(200, timesteps, input_dim)
    y_test = np.roll(x_test, shift=1, axis=1)
    test_prediction = model_overfit.predict(x_test)
    print(f"overfitting error in test data: {loss_fn(test_prediction, y_test).numpy():.4f}")
   

   
   # model using dropout to reduce overfitting
    inputs_reg = tf.keras.Input(shape=(timesteps, input_dim))
    lstm_out_reg = LSTM(units, return_sequences=True)(inputs_reg)
    drop_out_reg = Dropout(0.5)(lstm_out_reg)
    outputs_reg = Dense(input_dim)(drop_out_reg)
    model_dropout = Model(inputs=inputs_reg, outputs=outputs_reg)
   
    model_dropout.compile(optimizer=optimizer, loss=loss_fn)
    model_dropout.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=0)

    # predict some data to show the overfitting
    train_prediction_dropout = model_dropout.predict(x_train)
    print(f"dropout model error in train data: {loss_fn(train_prediction_dropout, y_train).numpy():.4f}")
    
    # creating new data to show the poor performance
    test_prediction_dropout = model_dropout.predict(x_test)
    print(f"dropout model error in test data: {loss_fn(test_prediction_dropout, y_test).numpy():.4f}")

    ```
   this script shows that the first model performs very well in the training data, but very bad on the new data. the second model using dropout in between the lstm and the dense layer has less overfitting. but always remember that overfitting is a very common problem and you must always be wary of it. in addition there are many ways to improve it, like l1 and l2 regularization, dropout and batchnorm.

*   **data preprocessing and feature engineering:** this might sound silly, but i've seen projects where the problems were not in the model itself but the data. sometimes you simply have not prepared the data well enough for the lstm model. i worked once on a sentiment analysis problem, and the model was not working very well until i found that the data was filled with a lot of html tags, and non-ascii characters. cleaning this improved the model performance dramatically. also creating features can help. for instance, using embeddings can help a lot in some cases. but in general, it is really difficult to give a recommendation here. it very much depends on the problem you have at hand.

for more info, check out *deep learning with python* by francois chollet. it has some really good explanations and examples of lstms and time series models. also *hands-on machine learning with scikit-learn, keras & tensorflow* by aurelien geron is a great resource for general machine learning concepts, including recurrent neural networks. they are really good books. avoid online tutorials for the fundamentals as many of them have mistakes.

finally, remember that debugging these models is hard. but with enough experience it can be less painful. also one thing that i've learned with time is that trying to build a simple model first is usually the best strategy. so, always start simple and then increase the complexity. remember that sometimes the best solution for a complex problem is to simplify it. it's kinda like coding, if you want a program that works, first make the first line of code work. and then go from there, one thing at a time.
