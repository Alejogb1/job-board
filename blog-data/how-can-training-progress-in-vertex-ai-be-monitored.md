---
title: "How can training progress in Vertex AI be monitored?"
date: "2024-12-23"
id: "how-can-training-progress-in-vertex-ai-be-monitored"
---

Okay, let's unpack this one. Monitoring training progress in Vertex AI, or indeed any machine learning environment, is not merely about watching numbers go up and down. It’s a critical part of ensuring that your models actually learn what they're supposed to, and more importantly, it allows you to intervene when things go sideways. Over my years, I’ve seen far too many training runs that looked promising initially only to reveal fundamental flaws much later, often at great expense. So, let's get down to the details of how to stay on top of that in Vertex AI.

The first layer of monitoring, and arguably the most fundamental, comes from the built-in tools that Vertex AI provides. During training, Vertex AI automatically logs a number of crucial metrics, which are easily accessible through the cloud console. These include standard metrics like training loss, validation loss, accuracy, and other metrics relevant to your chosen model and task (precision, recall, F1 scores etc.). It's not just about raw numbers though; it's the *trends* that matter.

For example, a common situation I’ve encountered is a rapidly decreasing training loss coupled with a stagnating or even increasing validation loss. This signals overfitting. Seeing that early allows me to implement regularization techniques, such as dropout, early stopping, or weight decay, before I’ve wasted significant compute time. Conversely, if both training and validation loss are consistently high, it might point to an issue with the data, model architecture, or the learning rate. Without this continuous, granular insight, debugging becomes a shot in the dark.

Beyond the default metrics, Vertex AI also allows you to define and log custom metrics. This is absolutely indispensable when you have very specialized models or evaluation needs. Let's say you're building a model for a very specific classification task where the cost of false positives is far greater than false negatives. In that scenario, tracking precision alongside recall or a custom metric like “cost-weighted accuracy” becomes crucial. Vertex AI makes this pretty straightforward. You can log these custom metrics during your training script itself.

Here’s a basic code snippet in Python using the Vertex AI SDK, illustrating how to log both built-in and custom metrics. This is a highly simplified example, you will, of course, incorporate your specific logic for model training:

```python
import vertexai
from vertexai.preview.training import CustomTrainingJob, TrainingJob
from google.cloud import aiplatform

PROJECT_ID = "your-gcp-project-id"
REGION = "your-gcp-region"
STAGING_BUCKET = "your-staging-bucket"

vertexai.init(project=PROJECT_ID, location=REGION)

def train_and_monitor_model(training_image_uri, script_path, machine_type="n1-standard-4"):
    """
    Trains and monitors a custom model using Vertex AI and logs metrics.
    """

    job = aiplatform.CustomJob.from_script(
        display_name="custom_training_job_logging_metrics",
        container_uri=training_image_uri,
        script_path=script_path,
        machine_type=machine_type,
        staging_bucket=STAGING_BUCKET
    )
    job.run()

if __name__ == "__main__":

    # dummy training script (replace with your actual code)
    SCRIPT_PATH = "training_script.py"
    with open(SCRIPT_PATH, "w") as f:
      f.write("""
import time
from google.cloud import aiplatform

def train():
    metrics_callback = aiplatform.training.TensorboardCallback(
        log_dir="gs://your-staging-bucket/tensorboard_logs" # make sure this is different from STAGING_BUCKET
    )
    for i in range(100):
        loss = 1.0/(i+1)  # simulating decreasing loss
        accuracy =  i/100.0 # simulating increasing accuracy
        aiplatform.training.log_metrics({ "loss": loss, "accuracy": accuracy, "custom_metric": loss*2 }, step=i)
        time.sleep(0.1)

if __name__ == "__main__":
    train()
    """)

    TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-12:latest"

    train_and_monitor_model(TRAINING_IMAGE_URI, SCRIPT_PATH)
```

This first example is foundational; we set up the `CustomTrainingJob` which runs your `training_script.py` containing your training logic. The crucial part is inside the training loop where we use `aiplatform.training.log_metrics` to report loss, accuracy, and a custom metric. Note that I recommend using a TensorBoard callback and logging metrics to a separate Google Cloud Storage directory to allow easy visualization via the TensorBoard feature in Vertex AI.

Now, while logging metrics is great, often a richer visualization is what helps gain deeper understanding. For that, Vertex AI integrates seamlessly with TensorBoard. Once you specify a directory for your TensorBoard logs, which we did in the prior snippet, these logs can then be viewed in the Vertex AI console. I usually monitor the TensorBoard graphs for things like the learning rate schedule, weights distributions, and activation histograms. These can all shed light on complex issues such as vanishing gradients or exploding activations, which are notoriously hard to pinpoint by just looking at numerical metrics. The key here is the use of histograms for the weights, biases, and activations which helps to identify model architecture issues as well.

Furthermore, the integration with TensorBoard isn’t just about pretty charts. It’s about allowing you to compare different runs or experiment with hyperparameter tuning. This brings me to the next crucial aspect: hyperparameter tuning and experiment tracking. Vertex AI offers the Hyperparameter Tuning service, which automates the process of finding the optimal hyperparameter values for your model. During tuning, the service automatically logs the results of each trial run, allowing you to easily compare performance and select the best model. These trials are also all fully monitored using logging as well, so issues during specific trials can be readily seen. This is incredibly important when working with complex models that have a large number of hyperparameters.

Let's see a quick example of how to run a hyperparameter tuning job with a few parameters we want to tune:

```python
import vertexai
from vertexai.preview.training import CustomTrainingJob, HyperparameterTuningJob
from google.cloud import aiplatform

PROJECT_ID = "your-gcp-project-id"
REGION = "your-gcp-region"
STAGING_BUCKET = "your-staging-bucket"

vertexai.init(project=PROJECT_ID, location=REGION)

def train_and_tune_model(training_image_uri, script_path, machine_type="n1-standard-4"):

    """
    Trains and tunes a custom model using Vertex AI and logs metrics.
    """
    
    tuning_job = aiplatform.HyperparameterTuningJob(
      display_name="hyperparameter_tuning_job_logging_metrics",
      trials=3,
      max_parallel_trials=2,
      custom_job=aiplatform.CustomJob.from_script(
        display_name="training_job_with_metrics",
        container_uri=training_image_uri,
        script_path=script_path,
        machine_type=machine_type,
        staging_bucket=STAGING_BUCKET
      ),
      metric_spec={"loss": "minimize"},
        parameter_spec={
            "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0.0001, max=0.01),
            "batch_size": aiplatform.hyperparameter_tuning.IntegerParameterSpec(min=16, max=128),
      }
    )
    tuning_job.run()

if __name__ == "__main__":
   # dummy training script (replace with your actual code)
    SCRIPT_PATH = "training_script_with_tuning.py"
    with open(SCRIPT_PATH, "w") as f:
      f.write("""
import time
import argparse
from google.cloud import aiplatform

def train(args):
    metrics_callback = aiplatform.training.TensorboardCallback(
        log_dir="gs://your-staging-bucket/tensorboard_logs_2" # make sure this is different from STAGING_BUCKET
    )
    for i in range(100):
        loss = 1.0/(i+1)
        aiplatform.training.log_metrics({ "loss": loss}, step=i) # only need to log the target metric
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    train(args)

    """)

    TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-12:latest"

    train_and_tune_model(TRAINING_IMAGE_URI, SCRIPT_PATH)

```

In this second snippet, the key change is that we now define a `HyperparameterTuningJob` which wraps a `CustomJob`, effectively reusing the same script as before, but modifying it by adding CLI arguments for learning rate and batch size. Also, note we only log the 'loss' metric here. The tuning service tries different combinations of hyper parameters and picks the one that minimizes the 'loss'. The main takeaway is that we do *not* need to change our logging approach to log custom metrics.

Finally, it’s important to remember that model training isn't a fire-and-forget process. I almost always create alerts based on key metrics like training time and early termination conditions. Vertex AI offers Cloud Monitoring integration, which lets you create alerts based on custom metrics. This is critical, especially for long-running jobs, so you get notified if there's a dramatic deviation from expected behavior. For example, you may want to stop the training job when loss stops improving for a specified number of steps.

Let’s see an example of a simple notification system using custom monitoring alerts:

```python
import vertexai
from vertexai.preview.training import CustomTrainingJob, TrainingJob
from google.cloud import aiplatform
from google.cloud import monitoring_v3

PROJECT_ID = "your-gcp-project-id"
REGION = "your-gcp-region"
STAGING_BUCKET = "your-staging-bucket"

vertexai.init(project=PROJECT_ID, location=REGION)

def create_alert_policy(
    project_id,
    training_job_id,
    display_name="training-loss-alert"
    ):
      """
      Creates a basic alert policy to monitor training loss.
      """

      client = monitoring_v3.AlertPolicyServiceClient()
      project_name = f"projects/{project_id}"

      filter = f'resource.type="aiplatform.googleapis.com/CustomJob" AND resource.labels.job_id="{training_job_id}" AND metric.type="aiplatform.googleapis.com/training/custom_metric" AND metric.label.metric_id="loss"'
      
      condition = {
          "display_name": "loss-decrease",
          "condition_threshold": {
          "filter": filter,
              "comparison": "COMPARISON_LT",
              "duration": "300s",  # check the value over 5 minutes
              "threshold_value": 0.2 # loss < 0.2
            }
       }
      
      alert_policy = {
        "display_name": display_name,
        "combiner": monitoring_v3.enums.AlertPolicy.CombinerType.OR,
        "conditions": [condition],
        # You can set up notification channels for email, sms, etc. here
        #"notification_channels": [notification_channel_name],
      }

      request = monitoring_v3.CreateAlertPolicyRequest(
          name=project_name, alert_policy=alert_policy
      )
      response = client.create_alert_policy(request=request)
      print(f"Created alert policy: {response.name}")
      return response

if __name__ == "__main__":
    # dummy training job
    SCRIPT_PATH = "training_script.py"
    with open(SCRIPT_PATH, "w") as f:
        f.write("""
import time
from google.cloud import aiplatform
def train():
    for i in range(100):
        loss = 1.0/(i+1)
        aiplatform.training.log_metrics({ "loss": loss}, step=i)
        time.sleep(0.1)
if __name__ == "__main__":
    train()
            """)

    TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-12:latest"
    
    job = aiplatform.CustomJob.from_script(
        display_name="custom_training_job_alert_policy",
        container_uri=TRAINING_IMAGE_URI,
        script_path=SCRIPT_PATH,
        machine_type="n1-standard-4",
        staging_bucket=STAGING_BUCKET
    )
    job.run()
    alert = create_alert_policy(PROJECT_ID, job.resource_name.split("/")[-1])
```

This third snippet sets up a very simple Cloud Monitoring alert based on a threshold of loss. Note the important parts: it specifies the job type via the resource type filter, the job id itself, the name of the custom metric we are tracking and finally the threshold based on a duration and a value. You would, of course, adapt this to your specific requirements and also configure notification channels to receive those alerts.

To conclude, effective training monitoring in Vertex AI isn’t just one thing; it’s a combination of using the built-in metrics, logging custom metrics, leveraging TensorBoard, automating hyperparameter tuning, and setting up alerts. This holistic approach ensures that you’re not just passively waiting for a model to finish training but actively guiding it to the best possible outcome. Regarding learning resources, I suggest delving into the official Google Cloud documentation for Vertex AI, particularly the sections on Custom training, hyperparameter tuning and monitoring, which provide very detailed explanations. Furthermore, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron provides solid grounding in model training and monitoring in general. And if you want to dive deep into advanced techniques like model compression, I recommend *Deep Learning for Vision Systems* by Mohamed Elgendy. These resources have helped me countless times and will equip you to get the most out of Vertex AI's monitoring features.
