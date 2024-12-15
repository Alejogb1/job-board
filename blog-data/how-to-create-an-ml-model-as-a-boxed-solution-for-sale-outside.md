---
title: "How to create an ML model as a boxed solution for sale outside?"
date: "2024-12-15"
id: "how-to-create-an-ml-model-as-a-boxed-solution-for-sale-outside"
---

alright, let's talk about packaging an ml model for sale. i've been down this road a few times, and it's rarely as straightforward as it seems at first glance. you have this awesome model, trained and performing well on your local machine or cloud setup and now comes the hard part: trying to make it usable by others, without giving away the secret sauce. it’s not just about dumping the model weights and a python script online, it involves some serious thought about the entire process. think of it as going from a prototype to a product.

first off, forget about expecting your customers to be python wizards or to have some specific jupyter environment. they probably want something that “just works”. this means we've got to think about packaging and ease of deployment. for me, the most common problems are dependency hell, incompatible python versions, and varying hardware, especially if you’re using cuda for gpu acceleration.

i learned this the hard way back in the days when i was building a text classification model to analyse support tickets, for a small company. i thought the model was all the juice, i wrote a notebook, all worked great, then i sent the notebook and the trained model file to the customer, they spent weeks trying to set up the proper python and cuda env and then simply gave up and that taught me a big lesson. it was a mess. never again.

i started to use docker for this, it's really a must when you want to get serious about distributing these things. containers help us encapsulate the model along with all its dependencies, the correct python versions and so on. it guarantees consistency across different platforms. we’re talking ubuntu, windows, macos, it doesn’t matter, if the customer has docker installed the model will probably work (of course you have to care about architectures for this thing, but let's leave that for later).

here’s how you might structure a basic dockerfile for such a project, imagine a model based on pytorch:

```dockerfile
# base image with python and dependencies
FROM pytorch/pytorch:latest-cuda12.1-cudnn8-runtime

# set working directory
WORKDIR /app

# copy your project files, including the model
COPY . /app

# install python dependencies
RUN pip install -r requirements.txt

# expose a port if needed (for example an api)
EXPOSE 8000

# command to run the application
CMD ["python", "main.py"]

```

the `requirements.txt` in this example should include all the packages your model uses, like `torch`, `numpy`, `scikit-learn`, whatever. this file is crucial to specify the dependencies to `pip`, here is an example:

```
torch==2.1.0
numpy==1.26.2
scikit-learn==1.3.2
pandas==2.1.3
fastapi==0.104.1
uvicorn==0.24.0.post1
pydantic==2.5.2
```

then we need a `main.py` file, a simple api is usually good to start. this will load the model and expose a simple endpoint to make predictions, here's an example with fastapi that is easy to use and relatively simple:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pickle

app = FastAPI()

# load the model here (adjust the path accordingly)
try:
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"failed to load model: {str(e)}")

# define input data model
class InputData(BaseModel):
    features: list

# define output data model
class OutputData(BaseModel):
    prediction: float

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    try:
      input_array = np.array(input_data.features)
      with torch.no_grad():
          prediction_tensor = model(torch.tensor(input_array).float())
          prediction = prediction_tensor.item()
      return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

in this setup, the client would send a post request to `/predict` with a json body looking like `{"features": [1,2,3,4]}`.

this combination is really a start to distribute your models. but let's go into more details of the things you will face.

next, consider model serialization. you can't expect customers to have the exact same version of your ml framework, for example, you may train your model with pytorch and the client needs to use tensorflow, or vice-versa, things like that happens a lot. a good idea is to use formats like onnx, it can help make it more portable, but the api creation will be more complex, and sometimes can lead to strange behaviors if the framework and versions are very different. but in my opinion this is a good idea if you are thinking about the long term, and that you will be supporting different platforms or model types. i think that is worth it to investigate this. another good approach is to have a serialization based on a common format like json or something else simple, the downside is that you may need to implement a lot of code. in our previous example i used python’s pickle, for that to work you really need the python versions to match. it’s not the best way, but easy for this example.

then, there's the api endpoint itself. consider using something like grpc, or even just rest using fastapi, this really depends on your preference, for me, something simple like fastapi is enough, but if you need things like bidirectional streaming or something more complex you should consider grpc. consider the case of large data for input. you really need to optimize the format for a large array and avoid big data transfers. something as simple as compression may solve most of the problems. and you should implement batch prediction in your api, it improves a lot the throughput if the user is trying to send a lot of data.

another important point is monitoring and logging. you can't just send a black box to the client. you need to provide some way to know if things are running fine or even if the model accuracy is dropping. integrate some logging system and also some way for the client to access it. think about metrics that may be important. model latency for example is something that your clients will probably worry about.

about hardware resources, you will need to specify how much ram and cpu will be necessary for the model to run correctly. a lot of your clients may not have gpus, so you should think about cpu inference as an option. you can provide models with different sizes or even different precisions. for example, your large accurate model could be a premium version and a small less accurate version could be the basic version for cpu inference.

security is a real concern. do you want the client to have full access to your model? or you want something more secure? you may need to encrypt the model weights before sending to the client. there are solutions for this, but it adds complexity to the whole process. this part is tricky, i've seen a lot of things being done, from simple obfuscation of code to complex schemes for loading the model only in trusted environments. you will have to decide for your own what is the correct approach, and it depends also a lot on your domain and the risk of someone stealing the model.

and then there is the documentation part. you need very clear documentation that explain how to set up the environment, run the container, call the api and understand the inputs and outputs of your model. you should include simple usage examples, the clients will thank you for that. i mean, you cannot expect someone that isn’t in your field to understand your inner code or ml concepts. you have to write the documentation with the least experienced user in mind.

testing is a critical point. before sending the model to the customer, you need a suite of tests to check if the model is running as expected, if the api is responding correctly and that the documentation matches with the actual implementation. the test suite should be executed in different environments. ideally in environments that will resemble as much as possible the environments of your client.

as for resources, i would suggest "hands-on machine learning with scikit-learn, keras & tensorflow" by aurelien geron. it is a great book, not only about models, but how to deploy things properly. for more details about docker, the official documentation is the bible. for the api creation, take a look at the fastapi documentation and the grpc documentation if you want to go that route. another book that i like is "designing data-intensive applications" by martin kleppmann, is not specific to ml, but it's a great one if you want to know more about how real world large scale systems are designed and built and how to properly handle large data.

one last thing, i had a funny story when i was creating a model to classify images, i accidentally trained the model with the labels all scrambled, the result was a model that was 100% sure it was predicting things, even when it was completely wrong. a funny mess, the labels were all shuffled up, it did classify, just in the wrong way, the point here is: check your labels carefully.

so yeah, selling ml models is more complex than you may think, but it's a very rewarding journey, just have a plan, and start by solving the dependency issues. use containers, create a simple api and make sure that you log the things that you can. it's a process, you will improve over time, just start small and improve from there, and please don't be like me that one time, and never forget that check your labels.
