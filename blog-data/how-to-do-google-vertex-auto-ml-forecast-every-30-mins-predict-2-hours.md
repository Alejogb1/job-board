---
title: "How to do Google Vertex Auto-ML Forecast every 30 mins predict 2 hours?"
date: "2024-12-15"
id: "how-to-do-google-vertex-auto-ml-forecast-every-30-mins-predict-2-hours"
---

alright, so you're trying to get vertex ai automl forecasting to crank out predictions every half hour, looking two hours into the future. i've been down this road, not exactly with the same timing, but close enough that i think i can point you in the good direction. this isn't a walk in the park, but it's totally doable. i remember back when i was setting up a system for predicting server load, we were dealing with similar frequency, though we needed 15 minutes predictions. it involved some serious head-scratching at first.

first, let’s break down the elements. we have vertex ai, which is google's machine learning platform, and automl, which is their offering to train models without a ton of code. and you want to use this to create time-series forecasts. the trick here is handling the prediction frequency and the time horizon while maintaining consistency. basically you can't directly specify a prediction cadence every 30 minutes within automl training options. automl isn’t setup to run on a schedule like that. it trains a model once based on your data and then that model is available for predictions, typically done through a batch prediction or online prediction. so, we need to get a bit creative.

here's the basic flow we're going for:

1.  **train the model:** you set up an automl forecasting model just like normal. you'll feed it your historical data. during training you don't have to worry about 30-min frequency. we’ll tackle that after the model is trained
2.  **create a prediction function:** we’ll create a simple script that calls vertex ai prediction api and then schedule this script to execute every 30 mins.
3.  **handle 30-minute intervals:** our script will pull the most recent data point and feed it into the trained model and get the prediction for 2 hours from that point.
4.  **repeat:** this process will run every 30 minutes.

let's get into code examples, i think that's where i can be more helpful.

**training the model**

this part is pretty standard vertex ai stuff, i won't go into details on that, but you can use this information:

*   make sure your historical data has timestamps and a value to predict, the timestamp has to be something vertex ai can parse automatically, normally date time format such as:  `YYYY-MM-DD hh:mm:ss`.
*   you define the time column and target column and the forecast horizon as your 2 hours you want to predict. it's very important you specify the right time column
*   upload your data to cloud storage bucket.
*   create an automl tabular forecasting model through the vertex ai console or api. specify the time column as the time column and forecast horizon to two hours, there is a parameter for forecast horizon and you have to pass 2.

**creating prediction script**

now, lets write the python function for predictions. it's going to be something like this:

```python
import google.auth
from google.cloud import aiplatform
from google.protobuf import json_format
import json
from datetime import datetime, timedelta

def predict_automl_forecast(project_id, model_id, location, instance_data):
    """
    makes a prediction using a deployed model endpoint
    """
    credentials, project = google.auth.default()
    aiplatform.init(project=project, credentials=credentials, location=location)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{model_id}"
    )

    prediction = endpoint.predict(instances=[instance_data])
    predictions = [json.loads(json_format.MessageToJson(prediction.predictions[i])) for i in range(len(prediction.predictions))]
    return predictions


def get_new_prediction(project_id, model_id, location, timestamp, feature_value):

    instance_data = {
        'feature_name_1': [feature_value], #use your column feature name from training
        'time_column_name': [timestamp.isoformat(timespec='seconds')], #use your time column name from training
    }

    predictions = predict_automl_forecast(project_id, model_id, location, instance_data)
    #here predictions will have the value predicted for two hours ahead from 'timestamp'

    return predictions


if __name__ == '__main__':
    project_id = "your-project-id"  # replace this
    model_id = "your-endpoint-id" # replace this
    location = "us-central1" # replace this

    # lets create a dummy new value with a timestamp
    current_time = datetime.now()
    feature_value = 100 # replace this with your latest reading
    predictions = get_new_prediction(project_id, model_id, location, current_time, feature_value)

    print(predictions)
```

some details about the script:

*   **project\_id, model\_id, location:** these are placeholders. you'll need to swap them out with your real project id, vertex ai endpoint id and google cloud location.
*   **instance\_data:** this is key, you'll create this data structure every 30 mins with the current data point. the column name for the feature 'feature\_name\_1' and 'time\_column\_name' must be exactly the same as the ones used when training the model.
*   **predict\_automl\_forecast:** i'm simply calling the prediction endpoint of your trained model.
*   **get\_new\_prediction:** is simply a helper function to create the required payload for the prediction.
*   the output `predictions` contains the values predicted for two hours into the future. it's in json format. the json is a little messy, you can parse it to extract only the numbers predicted.

**scheduling**

now, how to run the script every 30 minutes. there are a bunch of ways, but i can recommend a simple one, cloud scheduler:

*   create a cloud function and put the previous python function inside it
*   create a cloud scheduler job to call your cloud function every 30 minutes
*   make sure the cloud function and the cloud scheduler have the needed permissions to use vertex ai.
*   if you plan to run this script somewhere else you have to make sure that you handle authentication properly.

here's some of code for a cloud function, you'll need to tweak it to match the function before:

```python
import google.auth
from google.cloud import aiplatform
from google.protobuf import json_format
import json
from datetime import datetime, timedelta


def predict_automl_forecast(project_id, model_id, location, instance_data):
    """
    makes a prediction using a deployed model endpoint
    """
    credentials, project = google.auth.default()
    aiplatform.init(project=project, credentials=credentials, location=location)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{model_id}"
    )

    prediction = endpoint.predict(instances=[instance_data])
    predictions = [json.loads(json_format.MessageToJson(prediction.predictions[i])) for i in range(len(prediction.predictions))]
    return predictions


def get_new_prediction(project_id, model_id, location, timestamp, feature_value):

    instance_data = {
        'feature_name_1': [feature_value], #use your column feature name from training
        'time_column_name': [timestamp.isoformat(timespec='seconds')], #use your time column name from training
    }

    predictions = predict_automl_forecast(project_id, model_id, location, instance_data)
    #here predictions will have the value predicted for two hours ahead from 'timestamp'

    return predictions

def cloud_function_entry(request):

    project_id = "your-project-id"  # replace this
    model_id = "your-endpoint-id" # replace this
    location = "us-central1" # replace this

    # lets create a dummy new value with a timestamp
    current_time = datetime.now()
    feature_value = 100 # replace this with your latest reading
    predictions = get_new_prediction(project_id, model_id, location, current_time, feature_value)

    print(predictions)

    return f'predictions called at {current_time.isoformat(timespec="seconds")}, predictions {str(predictions)}'
```

and here is how a scheduler should look like:

*   **frequency:** `*/30 * * * *` (runs every 30 mins)
*   **target:** select the cloud function as target.

**additional notes**

*   **real-time data:** this setup assumes you've got a way to get new data every 30 minutes. you have to replace the `feature_value` from the example with your data. if not, you have to adapt the code.
*   **input data format:** your real input data might be more complex than the example. make sure the `instance_data` matches the format that your trained model expects. i once spent a whole afternoon debugging a similar error because i was mixing strings and numbers, it was something dumb i could have avoided by testing.
*   **error handling:** add robust error handling to your prediction function. network issues are common. you should log exceptions, and handle prediction failures. that’s key for a reliable system.
*   **monitoring:** implement monitoring on your scheduler jobs and cloud functions to make sure it’s all working as expected. you can use cloud monitoring for that.
*   **scalability:** as your data and prediction needs grow, consider more robust ways to run the prediction. this approach with cloud function and scheduler is for simple cases but might become expensive and harder to handle.

**recommendations**

*   **google cloud documentation:** the official google cloud documentation for vertex ai is the most important place. they have tons of examples and explanations that will be helpful. you must study it.
*   **"forecasting: principles and practice" by rob hyndman and george athanasopoulos:** this is not specific to automl but its a must read if you want to do forecasting correctly. it covers the math behind forecasting methods in detail.
*   **time series analysis by jonathan d. cryer and kung-sik chan:** another excellent text book about time series.

this might seem like a lot, but it's a pretty standard process when using automl in this particular way. you'll be passing the most recent data to your vertex model and getting your forecast for two hours forward. just take it step by step. no need to be perfect from the beginning, start simple and then iterate until you reach the setup you need.

and remember, the computer said to the programmer: "you've got all my logic, but i have all the data".
