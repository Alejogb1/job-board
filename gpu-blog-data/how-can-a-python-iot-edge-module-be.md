---
title: "How can a Python IoT Edge module be built for a Raspberry Pi?"
date: "2025-01-30"
id: "how-can-a-python-iot-edge-module-be"
---
Python's adaptability makes it a strong choice for IoT edge processing, particularly on resource-constrained devices like the Raspberry Pi. From my experience deploying numerous edge solutions over the past five years, I've found a structured approach, involving containerization and careful resource management, to be crucial for building robust and maintainable Python-based IoT modules.

The core challenge when developing for edge environments, specifically on Raspberry Pi, involves balancing computational demand with the limitations of the device. This is why directly deploying raw Python scripts often proves insufficient for reliable, production-ready solutions. Therefore, we typically build a containerized application, often employing Docker, which isolates our Python logic within its runtime, providing consistent execution across varied environments. This approach also simplifies deployments and versioning. Our application is then packaged as an edge module, ready to be pushed to the IoT device via an IoT hub.

The process of creating a Python IoT edge module on a Raspberry Pi generally follows these key steps. First, the Python application itself is developed, encompassing the core logic for data acquisition, processing, and transmission. Second, this application is containerized using Docker, including a Dockerfile that specifies the necessary dependencies and environment. Third, this Docker container is converted into a deployable edge module, often using tools specific to the chosen cloud platform, such as Azure IoT Edge or AWS Greengrass. Finally, the resulting module is deployed to the Raspberry Pi through the respective cloud management interface. This process involves a series of configurations to ensure the module can communicate correctly with the cloud platform and other edge services.

Let's explore the Python application aspect. Assume that we want a module that reads sensor data, performs simple moving average smoothing and publishes the processed data at regular intervals. Below is a simplified code structure:

```python
import time
import random
import json
from azure.iot.device import IoTHubModuleClient, Message

def get_sensor_data():
    # Simulate sensor reading
    return random.uniform(20, 30)

def moving_average(data, window_size):
    if len(data) < window_size:
        return sum(data) / len(data) if data else 0
    else:
        return sum(data[-window_size:]) / window_size

def create_message(sensor_value, avg_value):
    msg_body = {
        "temperature": sensor_value,
        "average_temperature": avg_value
    }
    return Message(json.dumps(msg_body))

def main():
    client = IoTHubModuleClient.create_from_edge_environment()
    client.connect()

    data_history = []
    window_size = 5
    while True:
        sensor_value = get_sensor_data()
        data_history.append(sensor_value)
        avg_value = moving_average(data_history, window_size)
        message = create_message(sensor_value, avg_value)
        print(f"Sending message: {message.data}")
        client.send_message_to_output(message, "output1")

        time.sleep(10)

    client.shutdown()

if __name__ == "__main__":
    main()
```

This code first imports the necessary libraries. It then simulates sensor data acquisition with `get_sensor_data`, and calculates the simple moving average with `moving_average`. It constructs a JSON formatted message using `create_message` and transmits the message using the `azure-iot-device` library, which is essential for connecting the module with the Azure IoT platform. The main loop runs indefinitely, publishing sensor and processed data every ten seconds. Here, I am using `azure-iot-device` as my cloud platform SDK, you should choose according to your cloud preference.

The second piece of the puzzle involves containerizing this Python application. We’d need a `Dockerfile` to accomplish this. Here's an example structure:

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

This `Dockerfile` builds the container image by first inheriting from a lean python base image, setting the working directory, copying over the `requirements.txt` file, installing the required Python packages specified in `requirements.txt`, copying the application code, and then specifying the command to execute when the container starts, which in this case, is running the python script `app.py`. The `requirements.txt` file should contain dependencies like `azure-iot-device` and any other python modules needed in the app. This `Dockerfile` is tailored to a Python 3.9 application; other versions might require modifications. Also, pay attention to the `requirements.txt` as this file ensures all dependencies are installed, eliminating runtime issues.

Finally, to transform this container into an edge module, we would use tooling provided by the cloud platform. For instance, within Azure, we might leverage the Azure IoT Hub extension for VS Code or use Azure CLI for this conversion and subsequent deployment. Once converted, the module is then pushed to the IoT hub, which pushes it to the target edge device based on the deployment manifests. Below is an example of how you might define a deployment manifest in Azure:

```json
{
    "modulesContent": {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.0",
                "runtime": {
                    "type": "docker",
                    "settings": {
                        "minDockerVersion": "1.25"
                    }
                },
                "systemModules": {
                    "edgeAgent": {
                        "type": "docker",
                        "settings": {
                            "image": "mcr.microsoft.com/azureiotedge-agent:1.4",
                            "createOptions": "{}"
                        }
                    },
                    "edgeHub": {
                        "type": "docker",
                        "status": "running",
                        "restartPolicy": "always",
                        "settings": {
                            "image": "mcr.microsoft.com/azureiotedge-hub:1.4",
                            "createOptions": "{\"HostConfig\":{\"PortBindings\":{\"8883/tcp\":[{\"HostPort\":\"8883\"}],\"443/tcp\":[{\"HostPort\":\"443\"}],\"5671/tcp\":[{\"HostPort\":\"5671\"}]}}}"
                        }
                    }
                },
                "modules": {
                    "SensorModule": {
                        "version": "1.0",
                        "type": "docker",
                        "status": "running",
                        "restartPolicy": "always",
                        "settings": {
                             "image": "your-registry/your-sensor-image:latest",
                             "createOptions": "{}"
                        }
                    }
                }
            }
        },
        "$edgeHub": {
            "properties.desired": {
                "schemaVersion": "1.0",
                "routes": {
                    "SensorModuleToCloud": "FROM /modules/SensorModule/* INTO $upstream"
                },
                "storeAndForwardConfiguration": {
                    "timeToLiveSecs": 7200
                }
            }
        },
        "SensorModule": {
          "properties.desired": {
             "sampling_interval": 10
         }
        }
    }
}
```

This manifest is a JSON file defining how to configure the IoT Edge runtime. It includes sections for `edgeAgent`, which manages the modules, `edgeHub` which acts as the message broker, and the user-defined modules such as `SensorModule`. This `SensorModule` section specifies the Docker image to pull. Make sure to replace `your-registry/your-sensor-image:latest` with the actual location of your module's docker image. This manifest example also includes a simple route that directs all messages from `SensorModule` to the cloud via the `$upstream` route. In the properties of `SensorModule`, configuration parameters like sampling intervals can be specified.

Developing Python IoT Edge modules on Raspberry Pi, requires this layered approach to effectively address the constraints of the edge device. Containerization ensures consistency and repeatability. Cloud management platforms provide the necessary tooling to deploy and manage these modules effectively.

For further exploration, I would recommend delving into books and online courses focusing on containerization with Docker, Python IoT development, and the specific cloud platform documentation you select. Understanding network configurations within a containerized environment and the architecture of your chosen IoT platform will prove essential in building successful, production-grade edge solutions. Additionally, practical experimentation, starting with simple examples and gradually incorporating more complexity, is always the best approach to deepen your understanding of the process.
