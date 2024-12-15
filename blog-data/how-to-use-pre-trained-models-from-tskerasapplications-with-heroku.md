---
title: "How to Use pre-trained models from `ts.keras.applications` with Heroku?"
date: "2024-12-15"
id: "how-to-use-pre-trained-models-from-tskerasapplications-with-heroku"
---

alright, so you're looking at using keras pre-trained models from `ts.keras.applications` within a heroku deployment. this is a pretty common scenario, and i've banged my head against it a few times myself. it mostly boils down to making sure your dependencies are managed properly and that the model loading process is smooth in a cloud environment. i’ll walk through my past hiccups with this and how i typically handle it now.

the biggest initial snag i hit, and many others too i'd bet, was assuming that heroku's build process would automatically install everything perfectly. i mean, it *does* try, but when you're pulling in hefty things like tensorflow, keras, and potentially some pre-processing libraries, things can go sideways. the first time i tried it, i had a model that was running locally like butter, a resnet50 if i recall correctly. the thing was classifying images of cats and dogs, you know, the typical introductory deep learning project. i deployed it thinking, 'easy peasy' and then i got that beautiful heroku application error screen. turned out, i was missing a specific version of tensorflow in my `requirements.txt`. a lesson well learned.

so, step one is to get your requirements spot on. now you probably have a `requirements.txt` file, if not please create one. make sure that you specify the tensorflow version that you used during your development and training. it is very critical. i'd advise you to also include keras and any other relevant packages. this is not the time to skimp on details. for instance, you'd have something that looks similar to:

```text
tensorflow==2.10.0
keras==2.10.0
numpy==1.23.5
scikit-image==0.19.3
pillow==9.4.0
```

notice how i'm pinning the versions? this is very important and essential. relying on the latest version can lead to unexpected issues if heroku's environment has a different default. it is basically chaos management. the number of hours you save from dependency headaches by pinning versions is remarkable.

now, the `ts` prefix in `ts.keras.applications` implies that you are working with tensorflow and keras models. that's good. this part should be straightforward because the models are already there; you don't need to load weights or do extensive model building. the real trick is to make sure heroku can *find* and *load* them at runtime. here is the minimal amount of code for just loading the model and giving an example using an image:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def load_and_predict(image_url):
    try:
        # Load the pretrained model
        model = keras.applications.resnet50.ResNet50(weights='imagenet')

        # Load and preprocess the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        img_array = np.array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)


        # Make the prediction
        predictions = model.predict(img_array)
        decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=3)[0]
        return decoded_predictions

    except Exception as e:
        return str(e) # return the error string

if __name__ == '__main__':
    url = "https://i.imgur.com/gJ22l0g.jpeg" # some random image url
    results = load_and_predict(url)
    print(results)
```

this shows the core idea. note that `weights='imagenet'` tells keras to download the pre-trained weights if not already available. now for heroku, this download during runtime might create problems, especially with slow network connections or resource limitations. to overcome this, one strategy i have used is to pre-download the weights and include them in my application's directory structure, usually in a folder named something like `model_weights`. if you do that, just make sure you change your code to load from the local path instead of `imagenet`. i have a folder called `model_weights` and inside i have two files: `resnet50_weights_tf_dim_ordering_tf_kernels.h5` and `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`.

i would do that in the development phase so that when deploying, it will already have the files:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def load_and_predict_local(image_url):
    try:
        # Load the pretrained model
        weights_path = 'model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        model = keras.applications.resnet50.ResNet50(weights=weights_path)
        
        # Load and preprocess the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        img_array = np.array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)


        # Make the prediction
        predictions = model.predict(img_array)
        decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=3)[0]
        return decoded_predictions

    except Exception as e:
        return str(e) # return the error string

if __name__ == '__main__':
    url = "https://i.imgur.com/gJ22l0g.jpeg"
    results = load_and_predict_local(url)
    print(results)
```

notice i changed the weights attribute to point to the location of the weights. when you deploy to heroku you need to make sure those weights are in the git repository and heroku deployment will automatically put them in the right path. also, if you have some weird caching issue, you can add code to check if the files exist.

now, if you're planning to do serious work with this i'd strongly suggest diving into resources that go into the nitty-gritty of tensorflow deployment. look for publications by martin abadi on the inner workings of tensorflow, or maybe the keras documentation, those sources usually provide a deeper understanding and are more valuable than blog posts. they usually tackle deployment from different angles and are very helpful. they’re dry, but they do the job.

one last thing i've seen a few times is the problem of memory. pre-trained models, especially the larger ones like vgg16 or inceptionv3, can consume a fair amount of ram. you may not face this right away but if your heroku instance is small, you might run into memory errors, particularly when multiple concurrent requests hit your application. to handle this, consider optimizing your code to run inference more efficiently and maybe look into asynchronous handling using tools like aiohttp if you have many requests, though that would make things more complex. alternatively, use a bigger heroku instance with a higher memory limit. you can also explore quantization or model pruning to reduce model size and memory consumption. there is an old saying that "an ounce of prevention is worth a pound of cure" and this saying always comes to my mind when working with memory and heroku.

here is a full flask example which uses the local weights to perform classification:

```python
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

def load_and_predict_local(image_url):
    try:
        # Load the pretrained model
        weights_path = 'model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        model = keras.applications.resnet50.ResNet50(weights=weights_path)

        # Load and preprocess the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        img_array = np.array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        predictions = model.predict(img_array)
        decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=3)[0]
        return decoded_predictions

    except Exception as e:
        return str(e) # return the error string

@app.route('/predict', methods=['POST'])
def predict_image():
    data = request.get_json()
    if not data or 'image_url' not in data:
       return jsonify({'error': 'missing image url'}), 400

    image_url = data['image_url']
    results = load_and_predict_local(image_url)
    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

this would create an endpoint that is listening at `/predict` and would take in a json input `{'image_url': your_image_url}` and would return the predictions of the resnet50 model.

in summary, the key takeaways for heroku deployment with `ts.keras.applications` models are: pin your dependencies, consider pre-downloading the model weights, and watch out for memory consumption. it’s more about good deployment practices and less about doing anything drastically different to what you would do locally. once you have all that sorted, deploying models with keras on heroku is relatively smooth. and well, after several deployments, it does become a smooth process.
