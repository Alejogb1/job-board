---
title: "How does a tf-serving which use GPUs concurrency work?"
date: "2024-12-15"
id: "how-does-a-tf-serving-which-use-gpus-concurrency-work"
---

alright, so you're asking about how tf-serving handles concurrency when gpus are in the mix. it's a good question and it gets to the heart of how these systems perform well under load. i've spent a good chunk of my career elbow-deep in this kind of infrastructure, so let me break it down from my experience.

first off, tf-serving by itself is built around a request/response model. think of it as a fancy http server that, instead of serving web pages, serves machine learning model predictions. when a request comes in, it needs to get processed and sent back quickly. now, this becomes tricky when you add gpus into the equation. gpus aren't just magical speed boosters; they have their own quirks and limitations, especially when multiple requests want to use them at once.

the core idea is that tf-serving uses concurrency primitives to manage how requests are executed on the gpu. it doesn't just let them all pile up and fight for gpu resources; it uses a scheduler to manage queues of requests. let's say we have a simple model that calculates the square of an input. the request comes in, it goes to the queue. the scheduler will pick requests from the queue and dispatch them to the gpu. this part is important: you can't just throw multiple requests at the gpu at the same time. the gpu works most efficiently when it's processing a batch of data. 

tf-serving internally orchestrates this batching process. when there is enough waiting requests in the queue it will create a batch of inputs and process them all in one shot on the gpu. then, it distributes the outputs of this batch to the respective awaiting requests. so, this whole process happens mostly under the hood, but you can tweak some settings. let me show you some examples.

here's a simplified conceptual snippet of how a client might send requests:

```python
import requests
import json

server_url = "http://localhost:8501/v1/models/my_model:predict"

def send_request(input_data):
    data = json.dumps({"instances": [input_data]})
    headers = {"content-type": "application/json"}
    response = requests.post(server_url, data=data, headers=headers)
    return response.json()

# example: multiple requests using threading for concurrency simulation
import threading

def worker(input_value):
  result = send_request({"input_tensor": [input_value]})
  print(f"input: {input_value}, output: {result}")

threads = []
for i in range(10):
    thread = threading.Thread(target=worker, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

this python code shows the client part sending requests, it is sending the requests concurrently using python threads, but that is not how the gpu is used, what happens inside tf-serving on the server side is the more interesting part.

internally tf-serving uses the underlying tensorflow runtime, to control gpu concurrency. tensorflow has an internal mechanism for creating a 'session' that allows running calculations. and a session needs a computational graph to execute. when tf-serving loads a model, it creates a graph and a session, then these resources are shared among concurrent requests, it is tf-serving who is responsible for managing how a batch is constructed and executed on the gpu.

let's look at a conceptual representation of how the tf-serving internal request handler might function (simplified, of course), this part is not a code, but rather an abstraction:

```
class RequestHandler:
    def __init__(self, model, gpu_device):
        self.model = model
        self.gpu_device = gpu_device
        self.request_queue = []
        self.lock = threading.Lock() # Lock for thread-safe operation.
        self.batch_size = 32 # example batch size value
        self.is_processing = False
        self.batch_thread = threading.Thread(target=self._process_batch, daemon = True)
        self.batch_thread.start()


    def enqueue_request(self, request_data, response_callback):
         with self.lock:
             self.request_queue.append((request_data, response_callback))


    def _process_batch(self):
        while True:
          with self.lock:
              if len(self.request_queue) >= self.batch_size and not self.is_processing:
                self.is_processing = True
                batch = self.request_queue[:self.batch_size] # get a batch of request
                self.request_queue = self.request_queue[self.batch_size:]  # Remove processed requests
                # now, we prepare to process the batch
                inputs = [req_data for req_data, _ in batch]
              else:
                 self.is_processing = False
                 continue # Wait until we have enough to make a batch
          
          if self.is_processing:
            outputs = self.model.predict(inputs, self.gpu_device) # process batch on the gpu
            for i,(_, callback) in enumerate(batch):
              callback(outputs[i]) # execute callback for each request

# simplified usage (for demonstration)
def dummy_model_predict(batch_inputs, device):
  print(f"device {device}: processing inputs: {batch_inputs}")
  return [input_val * input_val for input_val in batch_inputs]

def callback_response(result):
    print(f"response result: {result}")

gpu_device_id = 0 # let's assume this is our gpu device
model = dummy_model_predict  # use our dummy model for simplicity
handler = RequestHandler(model,gpu_device_id)
handler.enqueue_request([2],callback_response)
handler.enqueue_request([3],callback_response)
handler.enqueue_request([4],callback_response)
handler.enqueue_request([5],callback_response)
handler.enqueue_request([6],callback_response)
handler.enqueue_request([7],callback_response)
handler.enqueue_request([8],callback_response)
handler.enqueue_request([9],callback_response)
handler.enqueue_request([10],callback_response)
handler.enqueue_request([11],callback_response)
handler.enqueue_request([12],callback_response)
handler.enqueue_request([13],callback_response)
handler.enqueue_request([14],callback_response)
handler.enqueue_request([15],callback_response)
handler.enqueue_request([16],callback_response)
handler.enqueue_request([17],callback_response)
handler.enqueue_request([18],callback_response)
handler.enqueue_request([19],callback_response)
handler.enqueue_request([20],callback_response)
handler.enqueue_request([21],callback_response)
handler.enqueue_request([22],callback_response)
handler.enqueue_request([23],callback_response)
handler.enqueue_request([24],callback_response)
handler.enqueue_request([25],callback_response)
handler.enqueue_request([26],callback_response)
handler.enqueue_request([27],callback_response)
handler.enqueue_request([28],callback_response)
handler.enqueue_request([29],callback_response)
handler.enqueue_request([30],callback_response)
handler.enqueue_request([31],callback_response)
handler.enqueue_request([32],callback_response)
handler.enqueue_request([33],callback_response)
#this will run only one batch
```

this snippet should give you a taste of how a batch is created, processed in the gpu and then returns its results to the clients through callbacks.

the last snippet focuses on the configuration parameters for tf-serving model config file. here's an example of how you might configure batching and gpu options:

```json
{
  "model_config_list": [
    {
      "config": {
        "name": "my_model",
        "base_path": "/path/to/my/model",
        "model_platform": "tensorflow",
        "version_policy": {
          "latest": {
             "num_versions": 1
          }
        },
        "gpu_config": {
            "per_process_gpu_memory_fraction": 0.8,
            "allow_growth": true
          },
        "batching_config": {
          "max_batch_size": 32,
          "batch_timeout_micros": 10000,
          "num_batch_threads": 4
          }
      }
    }
  ]
}
```
the configuration file here defines `gpu_config` section, and the `batching_config`, the settings shown here are important to control concurrency using gpus. `per_process_gpu_memory_fraction` limits the fraction of gpu memory available for the model. `allow_growth` determines if the gpu memory is allocated dynamically instead of taking all the resources at the beginning. `max_batch_size` limits the maximum size of batch, `batch_timeout_micros` defines the maximum time to wait before a batch is processed and `num_batch_threads` indicates the amount of batch processing threads, if set to 0 or not defined the model will use default values.

now, these configs are important, but keep in mind the actual performance depends on many other factors, the size of the model, model complexity, gpu memory, number of cores and other settings of the system and infrastructure, like how many instances of the model are serving. i once spent a whole week just trying to get the batch sizes and thread counts just right for a model i was working on. it was like trying to fit a square peg in a round hole, but eventually, i got it working beautifully.

it's also worth exploring more advanced techniques like dynamic batching, where the batch size adjusts to the load. there are lots of interesting details you can look into. it is important to understand the details of batching. but it is a good place to start.

for resources, i would recommend looking into the tensorflow documentation itself, especially the sections on tf-serving and gpu usage. it will be very helpful if you study it in detail. also the book 'high performance tensorflow in production' by david gasperec is a great resource for this kind of problem. and lastly 'cuda by example' by jason sanders, edward kandrot, i know, not directly related to tf-serving but it's important to understand the internal architecture of the gpu to understand how these things really work, that helps to have a deep understand on what happens behind the scenes. that book is golden.

anyway, this is how i see the problem from my experience, i hope it is useful for you. let me know if you have any other questions.
