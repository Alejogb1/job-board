---
title: "Can Airflow operators run on an external service, and communicate with Airflow to update the progress of the DAG?"
date: "2024-12-15"
id: "can-airflow-operators-run-on-an-external-service-and-communicate-with-airflow-to-update-the-progress-of-the-dag"
---

yes, absolutely. it's a common pattern, and i've spent a good chunk of my career setting this exact thing up.  the core idea is that airflow doesn't need to be the one directly executing the heavy lifting; it's more like a conductor managing an orchestra. the operators in airflow can act as messengers triggering work elsewhere and then getting status updates back.

think of it this way: you’ve got airflow managing the overall workflow, maybe orchestrating some heavy data processing, machine learning training, or any other long-running task. instead of bogging down your airflow worker nodes with these compute-intensive jobs, you offload them to an external service. this could be anything: a kubernetes cluster, an aws batch environment, a cloud function, a dedicated server, or even a custom application. the key is that airflow needs to know what's happening over there to keep the dag running and know when it's time to move onto the next task.

the communication part is usually handled through an api or some form of messaging. the operator initiates the work on the external service and then periodically checks for updates. there are a few ways you can architect that:

**1. polling:** this is the simpler approach. after triggering a job externally, your operator repeatedly checks the external service for its status. this is where you would use http calls or whatever the api of the external service is, to fetch the status.
```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests
import time

class ExternalServiceOperator(BaseOperator):
    @apply_defaults
    def __init__(self,
                 external_service_url,
                 job_id,
                 poll_interval=60,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_service_url = external_service_url
        self.job_id = job_id
        self.poll_interval = poll_interval

    def execute(self, context):
        job_url = f"{self.external_service_url}/jobs/{self.job_id}"
        self.log.info(f"Starting job on external service, job id: {self.job_id}")

        response = requests.post(job_url)  # trigger start of work

        if response.status_code != 200:
           raise Exception (f"error triggering external service job, code {response.status_code}")

        while True:
            response = requests.get(job_url)
            response.raise_for_status()

            job_status = response.json().get('status')
            self.log.info(f"job status is: {job_status}")
            if job_status == 'completed':
                self.log.info(f"job with id: {self.job_id} complete")
                return
            elif job_status == 'failed':
                raise Exception(f"job with id: {self.job_id} failed")
            else:
                time.sleep(self.poll_interval)
```
this example assumes a simple api with endpoints like `/jobs/{job_id}` where you can trigger it with a `post` and fetch status using a `get`. it's a basic example, of course, and you might have to modify based on the api of your external service, but it shows the core concept of triggering work and checking its status in a loop. notice the `response.raise_for_status()`, that handles api errors and will stop if any of the requests is not a 200.

**2. callbacks/webhooks:** for a more real-time approach, the external service can notify airflow about updates using callbacks or webhooks. this removes the need for constant polling and makes things more efficient. your operator triggers the external job, then the external service sends a signal to airflow's api when the job completes, or fails.
```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests

class ExternalWebhookOperator(BaseOperator):
    @apply_defaults
    def __init__(self,
                 external_service_url,
                 job_id,
                 airflow_callback_url,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_service_url = external_service_url
        self.job_id = job_id
        self.airflow_callback_url = airflow_callback_url

    def execute(self, context):
        job_url = f"{self.external_service_url}/jobs/{self.job_id}"
        self.log.info(f"starting job on external service, job id {self.job_id}")

        payload = {"callback_url": self.airflow_callback_url, "job_id": self.job_id}
        response = requests.post(job_url, json=payload)  # send work and callback

        if response.status_code != 200:
          raise Exception (f"error triggering external service job, code {response.status_code}")

        self.log.info(f"waiting for callback notification for job: {self.job_id}")
```
in this case, your external service now needs to be capable of sending http posts. on the airflow side you would need some endpoint where airflow can receive callbacks, and update the state of the task. you can achieve this by using the airflow api, in particular the callbacks api, but for simple cases i've used simple flask api server running parallel to airflow that receives the callback and uses the airflow api to update the state of the task to success or failure. this approach is more involved on the initial setup but it greatly reduces resource usage specially if you have long running tasks.

**3. message queues:**  another pattern is to use message queues like rabbitmq or kafka for communication. your operator publishes a message to the queue to initiate work, and the external service consumes the message, does its processing and publishes another message back to another topic for airflow to consume and update the dag. this is a more advanced approach, but provides better decoupling and resilience.
```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from kafka import KafkaProducer, KafkaConsumer
import json
import time

class ExternalKafkaOperator(BaseOperator):
    @apply_defaults
    def __init__(self,
                 bootstrap_servers,
                 request_topic,
                 response_topic,
                 job_id,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.job_id = job_id
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.consumer = KafkaConsumer(self.response_topic, bootstrap_servers=self.bootstrap_servers,  value_deserializer=lambda m: json.loads(m.decode('utf-8')), auto_offset_reset='earliest')

    def execute(self, context):
      self.log.info(f"starting job on kafka: {self.job_id}")
      message = {"job_id": self.job_id, "status": "started"}

      self.producer.send(self.request_topic, value=message)
      self.producer.flush()
      self.log.info(f"waiting for response on topic: {self.response_topic} for job: {self.job_id}")
      timeout = time.time() + 300 # wait 5 mins
      while True:
          for message in self.consumer:
            if message.value.get("job_id") == self.job_id:
                if message.value.get('status') == 'completed':
                  self.log.info(f"job with id: {self.job_id} complete")
                  return
                elif message.value.get('status') == 'failed':
                    raise Exception(f"job with id: {self.job_id} failed")
          if time.time() > timeout:
             raise Exception("timeout waiting for kafka response")
          time.sleep(10)
```
in this example i use kafka, but you could use rabbitmq. it shows how an operator sends an initial message to a topic with the job details, the external service consumes that, does it work and then sends a message to another topic, where the operator consumes the message and updates the state of the task. the key benefit here is decoupling since the operator doesn't need to know the address of the external service just its kafka topic. this is generally used in complex architectures that might have many microservices.

i've done this kind of integration with airflow multiple times, from basic batch processing on aws batch to more involved machine learning training pipelines on kubernetes. i remember one time i was using a custom cli application for running experiments and i had to write a custom operator to interface with it. i did it using http calls, because the application exposed a simple api. at first i did a very aggressive polling, like every 5 secs which was silly because the job could last many hours and i ended up putting unnecessary stress on the external service. after a quick refactor i increased the polling period and moved to an exponential backoff. it just goes to show that the simplest solution can cause problems if not implemented with care. on another occasion i had to integrate with a serverless function that didn't have webhooks, so i also went for the polling approach. it's always a trade off between simplicity and efficiency.

for further reading, i would recommend focusing on distributed systems design and patterns. books like "designing data-intensive applications" by martin kleppmann are excellent at providing the fundamentals of building reliable, scalable systems that use these communication methods. also, if you're using kubernetes, understanding the operators pattern is a must, check out kubernetes documentation, it's a great resource. for specific airflow stuff, the official airflow documentation has a lot of material, also the apache airflow github repository has many examples, so worth checking it out, and the airflow slack channel has a great and helpful community if you have questions. oh, and one last thing, never underestimate the power of a good log message. i mean, how else are you going to know what's going on when it inevitably breaks down? the other day i was debugging a dag and i found a message like "error id 123 occurred" and i swear i spent a whole afternoon trying to find what error 123 actually meant until i realised it was in my code and i didn't log the error message itself, rookie mistake, but it's good to learn from these.
