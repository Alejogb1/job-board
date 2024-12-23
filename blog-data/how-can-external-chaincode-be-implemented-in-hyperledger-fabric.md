---
title: "How can external chaincode be implemented in Hyperledger Fabric?"
date: "2024-12-23"
id: "how-can-external-chaincode-be-implemented-in-hyperledger-fabric"
---

Okay, let's unpack this. I've had my share of encounters with complex distributed systems, and implementing external chaincode in Hyperledger Fabric definitely sits firmly in that category. It's not a simple point-and-click affair, but once you understand the underlying mechanisms, it becomes quite manageable. In essence, external chaincode, often referred to as "chaincode as a service," lets you execute your smart contract logic outside the Fabric peer process. This decoupling can bring substantial benefits in terms of resource isolation, language flexibility, and even improved security in some use cases.

The standard Fabric chaincode model has its limitations. Everything runs within the peer's environment, constrained by the available resources and the limited set of programming languages. External chaincode addresses this head-on, permitting you to implement your smart contract in any language and even run it on specialized hardware. I remember one particularly challenging project years ago, where we had to integrate legacy C++ code with our blockchain network; external chaincode saved our project’s timeline, and I want to share that approach.

At a high level, the implementation hinges on establishing a communication channel between the Fabric peer and your external chaincode process. This communication usually follows a gRPC-based protocol, defined by Fabric. The peer sends transaction invocation requests to your external process, which executes the smart contract logic and responds with results. This communication is typically bi-directional, allowing the chaincode to query ledger state or to perform other Fabric-specific operations.

Here’s where it gets practical, and I’ll illustrate my points with some simplified code examples. Consider this scenario: we're developing a supply chain application. We need to perform complex calculations for temperature tracking that are computationally expensive and ideally handled using a high-performance scientific computing library not available for standard Fabric chaincode. We’d opt for external chaincode written using Python.

**Example 1: Defining the gRPC Service (in Python, external chaincode)**

Firstly, we define the gRPC service for our chaincode using the protobuf definition provided by the Hyperledger Fabric project (`chaincode.proto`).  While the exact setup might require some digging into the Fabric documentation (I'd recommend the Fabric SDK documentation and especially the `chaincode.proto` file for a precise definition), here’s a simplified version of what your Python service would look like:

```python
from concurrent import futures
import grpc
import chaincode_pb2
import chaincode_pb2_grpc


class TemperatureProcessor(chaincode_pb2_grpc.ChaincodeServicer):

    def Init(self, request, context):
        print("init called")
        return chaincode_pb2.Response(status=200, payload=b"Initialization successful")


    def Invoke(self, request, context):
        function_name = request.payload.decode().split(",")[0]
        args = request.payload.decode().split(",")[1:]

        if function_name == "processTemperature":
           try:
              temperature_value = float(args[0])
              processed_value = self._calculate_processed_temperature(temperature_value)
              return chaincode_pb2.Response(status=200, payload=str(processed_value).encode())
           except (IndexError, ValueError) as e:
              return chaincode_pb2.Response(status=500, payload=str(e).encode())
        else:
            return chaincode_pb2.Response(status=400, payload=b"Invalid function name")

    def _calculate_processed_temperature(self, temperature):
        # Simulate complex calculation; you'd replace this with your real logic
        return temperature * 1.05


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chaincode_pb2_grpc.add_ChaincodeServicer_to_server(TemperatureProcessor(), server)
    server.add_insecure_port('[::]:7052') #Note this port can be anything not already in use.
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
```

This Python code sets up a gRPC server listening on port 7052 (you’d need to ensure this port is not in use). It implements the `ChaincodeServicer` interface defined in the `chaincode.proto`, including the `Init` and `Invoke` methods. Note that these methods have to be implemented and follow the Fabric-specific message structures. The `_calculate_processed_temperature` method acts as a placeholder for our specific calculations.

**Example 2: Configuring the Fabric Peer (configuration file)**

Next, you need to configure the Fabric peer to use the external chaincode. This typically involves modifying the `core.yaml` file (or its equivalent based on your setup) to point to the endpoint of your external chaincode. Below is a snippet of what such a configuration might look like:

```yaml
chaincode:
  externalBuilders:
    - path: /path/to/your/external/builder
      name: python-external-chaincode
      proprietary: false # Indicates this is an out-of-tree builder
      properties:
        address: 127.0.0.1:7052
        dial_timeout: 10s
        tls_required: false # TLS configuration may vary, depending on use case
        keepalive_time: 60s
        keepalive_timeout: 20s
```

In this configuration, `address` points to the gRPC endpoint where your Python process is listening (127.0.0.1:7052 in this case).  The `name` is a reference that needs to be set in the channel config, which will match chaincode that utilizes this builder. You also see parameters such as `dial_timeout`, `tls_required`, `keepalive_time`, and `keepalive_timeout`, which manage how the Fabric peer communicates with your external service. The value for path isn't directly important; it's used only when the builder is in tree. Here we are setting proprietary to false which indicates that it is not and the `path` parameter is not applicable.

**Example 3: Deploying and Invoking the External Chaincode (Fabric CLI)**

Finally, you use the Fabric CLI to deploy and invoke your chaincode, ensuring that your configuration points correctly to the `python-external-chaincode`. This process usually involves packaging the chaincode with specific parameters, which include the builder name from the previous step.

```bash
# Package the chaincode
peer chaincode package -n temperature-processor -v 1.0 -p /path/to/my/chaincode --lang external --path mycc.tar.gz

#Install and approve the chaincode on a channel
peer chaincode install -p mycc.tar.gz
peer lifecycle chaincode approveformyorg -o localhost:7050 --channelID mychannel --name temperature-processor --version 1.0 --package-id <package-id> --sequence 1 --init-required --signature

# Commit the chaincode (assuming approvals are granted)
peer lifecycle chaincode commit -o localhost:7050 --channelID mychannel --name temperature-processor --version 1.0 --sequence 1 --init-required  --signature

#Invoke the chaincode (example)
peer chaincode invoke -o localhost:7050 -C mychannel -n temperature-processor -c '{"Args":["processTemperature", "25"]}'
```

The chaincode is installed and approved like normal, the difference comes in when the chaincode is packaged and committed. The package path (here, /path/to/my/chaincode) refers to the directory where the files, including the tar.gz package are located for the external builder. Also, the `--lang external` indicates that it is an external builder. The commit command also reflects that this builder is to be used for the channel through parameters passed in. Finally the invoke command passes the function name and the temperature as a string. When the peer receives this command, it packages it up to be sent to the external builder via gRPC and, on successful completion, returns the processed temperature value.

Implementing external chaincode introduces additional complexity in areas such as security, deployment, and monitoring. You would have to manage the external chaincode process separately, including any dependencies it might have. Also, setting up proper logging and monitoring for your external service is important for debugging issues. You also have to consider securing the communication between the Fabric peer and your external service, often through TLS, which was omitted for simplicity here.

To delve deeper into the nuances of external chaincode, I'd recommend exploring the official Hyperledger Fabric documentation, including the fabric-samples repository. The "Developing Applications" section, specifically the documentation related to chaincode lifecycle and external chaincode, is a great starting point. The "Programming Model" section of the documentation details how chaincode functions interact with the Fabric infrastructure, offering crucial understanding for implementing your own logic. The gRPC documentation itself, available at grpc.io, is also invaluable for comprehending the underlying communication protocol. Understanding the specific gRPC messages defined in `chaincode.proto` is critical for correctly implementing your external service.

In conclusion, implementing external chaincode in Hyperledger Fabric is a powerful way to overcome some of the limitations of standard chaincode. It opens up a plethora of possibilities but requires careful attention to configuration, security, and deployment. I've found that a thorough understanding of the Fabric architecture and gRPC is crucial for successful implementation, and experience does play a pivotal role in troubleshooting.
