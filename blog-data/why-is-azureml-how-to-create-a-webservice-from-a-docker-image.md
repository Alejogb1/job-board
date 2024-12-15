---
title: "Why is azureml how to create a webservice from a docker image?"
date: "2024-12-15"
id: "why-is-azureml-how-to-create-a-webservice-from-a-docker-image"
---

alright, so you’re looking at deploying an azure machine learning webservice from a docker image, and yeah, it's not always super straightforward the first time around, been there done that. let me break down how i usually approach this, drawing from some of the stuff i’ve tripped over in the past.

first off, let's tackle the "why?" part. why docker images? because they offer a consistent, reproducible environment. think of it, you build your application and all its dependencies into a container, it will run the same way regardless of the target environment. this is killer for ml deployments where dependency hell can be a real problem. i remember one project where we spent more time fixing library conflicts than we did on the actual model, docker saved the day for that. also, it's not a black box anymore. you can verify each layer and see that your exact code is running as expected.

now, azureml is designed to handle these docker images smoothly. under the hood, azureml is very much container-based. it leverages docker to containerize the model serving code, and related dependencies. instead of having to manage virtual machines and configurations, everything is neatly tucked into a docker image. you push it to the registry, azureml grabs it and uses it to spin up the web service.

so how do we go about it? here’s a typical workflow, in my experience:

1.  **create your docker image:** this is where you get all your code, model, dependencies, and the scoring script together into one image. your dockerfile should be clean. it should start from an optimized base image, install your packages, copy in your model and scoring script, and then set the command or entrypoint to start your service. the scoring script is a critical part, as this dictates how the inference is executed.

    here’s an example dockerfile that i’ve used for one particular image i needed to deploy:

    ```dockerfile
    from mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest

    # install necessary dependencies
    run pip install -r requirements.txt

    # copy the model and scoring script
    copy model.pkl /app/model.pkl
    copy score.py /app/score.py

    # set the entry point to run the scoring script using a custom uvicorn command
    entrypoint ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "8080"]
    ```

    note, i'm using `uvicorn` here as an example of a web server, you can use `flask` or anything that can respond to requests. also, that `requirements.txt` should contain all python packages your script uses, of course. and don’t forget that base image `mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest` which comes with the common ml tools, saving you some time.

    and here’s a simple `score.py` example, it uses `fastapi` which i prefer due to its automatic validation and generation of documentation, pretty useful if you ask me:

    ```python
    from fastapi import FastAPI
    import pickle
    import numpy as np

    app = FastAPI()

    with open('/app/model.pkl', 'rb') as f:
        model = pickle.load(f)


    @app.post("/predict")
    async def predict(data: dict):
        try:
            input_data = np.array(data['data']).reshape(1, -1)
            prediction = model.predict(input_data)
            return {"prediction": prediction.tolist()}
        except Exception as e:
            return {"error": str(e)}
    ```

    make sure to save your model as a `pickle` or `joblib` file (or any other format you prefer). also, i’m assuming your model takes an array of numbers and returns a single prediction here.

2.  **build and push your image:** once you have your dockerfile and code ready, you can build the image locally and then push it to an azure container registry (acr).

    ```bash
    docker build -t my-ml-image .
    docker tag my-ml-image myacr.azurecr.io/my-ml-image:v1
    docker push myacr.azurecr.io/my-ml-image:v1
    ```
    replace `myacr.azurecr.io` with your container registry address and `my-ml-image` with your image name. also note that versioning can become important, v1, v2 and so on is a good idea, especially for iterative updates.

3.  **deploy with azureml:** now, here’s where you tell azureml to use your docker image. you do that using the azureml sdk, the python one specifically.

    ```python
    from azureml.core import Workspace
    from azureml.core.model import Model
    from azureml.core.webservice import AciWebservice, Webservice
    from azureml.core.image import ContainerImage

    # load workspace
    ws = Workspace.from_config()

    # define the acr image
    image_config = ContainerImage(
        name="my-ml-image",
        image_location=f"{ws.container_registry}.azurecr.io/my-ml-image:v1",
        )
    
    # define deployment configuration
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    
    # deploy the webservice
    service = Webservice.deploy_from_image(
        deployment_config=deployment_config,
        image=image_config,
        name="my-ml-webservice",
        workspace=ws
    )
    
    service.wait_for_deployment(show_output=True)

    print(service.state)
    print(service.scoring_uri)
    ```

    what’s happening in this script? first you load your azureml workspace, then you define your container image pointing at the acr, then you define the deployment config and finally you deploy a webservice with the `deploy_from_image` command. you get the scoring uri at the end. now your service should be up and running. this is a good baseline to work with.

    this approach is way cleaner and more manageable than trying to use the cli interface, i prefer coding in python so this is a natural fit. also you can easily parametrize everything, if you need more resources, or if you need to test on other environments, just tweak the `deploy_configuration`.

    there are alternatives to aci, like deploying on aks, but aci is the go-to choice for smaller deployments. for large scale deployments where you need a k8s cluster, aks is the way to go.

regarding resources, i'd skip the usual blog posts and recommend some heavier reads. "deep learning with python" by francois chollet, is a great start for the ml part, even if it does not cover deployments directly. for containerization, “docker in practice” by ian miell and aidan hobart is also really useful. and for general mlops concepts, "machine learning design patterns" by valliappa lakshmanan is a good read.

i should mention also that debugging this kind of setup can be a pain. always check your logs. azureml logs are your best friend in debugging problems. one time i spent 2 hours trying to figure out why my service was not working, turned out i was pointing to the wrong model file in the container image. a simple look at the logs would have saved me some time.

finally, always remember to properly clean your resources when you don't need them anymore. cloud costs can become very high if you forget about your resources. the only thing that should scale infinitely is your brainpower, not your cloud budget.

anyway, i think that covers the basics of deploying an azureml web service from a docker image, at least from my experience. feel free to ask if anything is not clear. good luck and happy coding!
