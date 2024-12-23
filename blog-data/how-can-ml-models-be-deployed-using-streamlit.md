---
title: "How can ML models be deployed using Streamlit?"
date: "2024-12-23"
id: "how-can-ml-models-be-deployed-using-streamlit"
---

Okay, let's explore deploying machine learning models using Streamlit. I recall a particularly challenging project a few years back, where we had a complex image classification model trained on a massive dataset. We needed a way to make it accessible to non-technical stakeholders for evaluation and testing purposes, and that's where Streamlit really shined. It wasn't about building a complex web application; it was about rapid prototyping and demonstrating value, and Streamlit’s ease of use was a game changer.

The beauty of Streamlit lies in its ability to translate Python code directly into interactive web applications without requiring extensive front-end development knowledge. It lets you focus on the data science and machine learning parts of the equation, not the intricacies of html, css, or javascript. From my experience, when you're in the rapid prototyping or iteration phase, this is absolutely crucial.

At its core, Streamlit operates by re-running your entire Python script whenever a user interacts with a widget. This might sound inefficient, but Streamlit’s caching mechanism and intelligent state management handle this quite effectively for most use cases. The core deployment pattern usually looks like this: you load your trained model, you define your user interface elements with Streamlit's intuitive api, and then you tie the user's input to a prediction function. The results are then displayed back to the user.

Let’s break this down with a few practical examples. Assume we’ve already trained a simple linear regression model and saved it using `pickle`.

**Example 1: Simple Linear Regression Model**

This code illustrates deploying a very basic model, showing how Streamlit can take numerical input and display the prediction output.

```python
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
try:
  with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
except FileNotFoundError:
    st.error("Could not find 'linear_regression_model.pkl'. Please ensure the model file exists.")
    st.stop()

# Title of the app
st.title("Linear Regression Predictor")

# Input for the user
input_feature = st.slider("Select Input Feature", 0.0, 10.0, 5.0)

# Prediction logic
if st.button("Predict"):
    input_array = np.array([[input_feature]])
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Output: {prediction:.2f}")

```

In this example, we load the pickled model, create a slider using `st.slider`, and a button `st.button`. Once the user clicks the predict button, the model predicts based on the slider's value and the output is displayed. This simple example demonstrates the basic flow - input, processing, and output. A key aspect to understand here is the `st.button` - the code inside the `if` block only runs when that button is triggered.

**Example 2: Image Classification**

Now, let's move to a scenario closer to my earlier image classification problem, handling image uploads. For this, assume you have a model trained with something like `tensorflow` or `pytorch`. For demonstration purposes, I'll simulate a dummy model that returns a random class. This keeps the focus on the Streamlit side of things.

```python
import streamlit as st
import numpy as np
from PIL import Image
import io
import random #for the dummy model

# Dummy model (replace with your actual model loading)
def dummy_classify(image):
  labels = ["Cat", "Dog", "Bird", "Fish"] # Just some classes
  return random.choice(labels)

st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Image"):
            #Preprocess the image here if required based on the model type.

            prediction = dummy_classify(image)
            st.success(f"Predicted Class: {prediction}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
```

Here we are using `st.file_uploader` which allows users to upload an image. Once uploaded, it gets displayed, and clicking the “Classify Image” button runs the prediction using our dummy classifier. Replace `dummy_classify` with your actual model loading, preprocessing, and prediction logic. Note how we are handling potential errors during file processing which is a good practice. Also, the use of a `try/except` block is crucial for robust applications. Remember, always sanitize user inputs and handle potential errors gracefully.

**Example 3: Data Exploration with Pandas**

Streamlit isn't just for model deployment; it is also fantastic for basic data exploration. Consider this example where you let users explore a pandas dataframe.

```python
import streamlit as st
import pandas as pd

# Sample dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
}
df = pd.DataFrame(data)

st.title("DataFrame Explorer")

st.write("### Original DataFrame")
st.dataframe(df)

selected_column = st.selectbox("Select a Column", df.columns)
st.write(f"### Displaying column: {selected_column}")
st.write(df[selected_column])

if st.checkbox("Show Summary Statistics"):
    st.write("### Summary Statistics")
    st.write(df.describe())
```

This snippet shows how Streamlit can handle and display pandas dataframes in a user-friendly way. We start by creating a sample dataframe and then we display it. Then, the `st.selectbox` enables selection of a column, and the output reflects the user’s selection. The `st.checkbox` allows toggle of the summary statistics section. This simple example demonstrates how Streamlit can create interactive reports and visualizations.

**Key Considerations for Production**

While Streamlit is fantastic for rapid development, some aspects should be considered for production scenarios. Firstly, if your model is computationally intensive, running inference on the same thread that handles the web requests might cause the application to become unresponsive. You may consider using asynchronous task queues like Celery or Redis Queue to offload the heavy lifting. Secondly, for scaling Streamlit in production, a good practice would be to use a load balancer in front of multiple instances of your Streamlit application, using a framework like Docker and Kubernetes for containerization and deployment. Also, security considerations should never be ignored when exposing your application to external users. Ensure your deployment environment is properly secured.

**Further Learning**

For a deeper understanding of machine learning deployment, I strongly suggest these resources. For general ml model development, "Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow" by Aurélien Géron is a fantastic resource. For best practices in deploying and managing ML models, I would highly suggest “Machine Learning Engineering” by Andriy Burkov. Finally, for more advanced Streamlit concepts and usage patterns, consulting the official Streamlit documentation is invaluable; specifically, look into sections covering caching and state management for optimization of your applications. These resources helped me a lot on my journey and will definitely guide you along the way too.

In closing, Streamlit provides a very accessible and efficient way to turn your python machine learning projects into interactive demos. However, for production systems, you’ll need to augment this with additional architectural decisions to ensure your application is scalable, performant and secure. This is something I have learnt the hard way, but the ability to quickly demonstrate the value of a trained model to stakeholders cannot be underestimated.
