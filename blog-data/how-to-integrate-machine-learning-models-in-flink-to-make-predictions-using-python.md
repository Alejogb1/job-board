---
title: "How to integrate machine learning models in flink to make predictions using python?"
date: "2024-12-15"
id: "how-to-integrate-machine-learning-models-in-flink-to-make-predictions-using-python"
---

alright, so you're looking to get your machine learning models, trained probably with python, running inside apache flink for real-time predictions. i’ve been down this road a few times, and it’s definitely got some quirks, so let's break it down.

first off, the core problem here is that flink is a java/scala environment, and your models, typically, are python. they’re not speaking the same language. we need a bridge, something that allows flink to talk to your python model. the best way, in my experience, is through a remote procedure call (rpc) setup. basically, you run a python server that hosts your model, and flink sends requests to it. think of it like a microservice your flink job interacts with.

when i first tried this, oh, around 2017 maybe? i thought, “easy, just pass the data to python, get the answer, done”. yeah… that was a mess. serialization issues, python versions mismatching, the works. my flink jobs were crashing more than my laptop during an exam week. lessons were definitely learned.

one crucial part is choosing the correct way to communicate. for this i've used `grpc` the most, it's faster than, let's say, rest. it makes for a more responsive prediction system which you need for data streaming. you define your data types in protobufs, generate code for both python and java/scala, and you're in business.

let's get concrete. here's a basic idea of how your python server might look:

```python
import grpc
import prediction_pb2
import prediction_pb2_grpc
from concurrent import futures
import pickle
import numpy as np

class PredictionService(prediction_pb2_grpc.PredictionServiceServicer):
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def Predict(self, request, context):
        # Assuming input is a list of floats
        input_data = np.array(request.features).reshape(1, -1)
        prediction = self.model.predict(input_data)[0]
        return prediction_pb2.PredictionResponse(prediction=float(prediction))


def serve(model_path, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionService(model_path), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    model_path = 'my_model.pkl' # Replace with your model path
    serve(model_path)
```

this script sets up a grpc server, loads your trained model (using pickle, assuming it's a sklearn model or something similar), and exposes a `predict` method. you'll need the `protobuf` files and grpc generated python code. make sure the data in `prediction_pb2` matches what your flink job sends.

now, on the flink side, you need to use a `flatmapfunction` to make those requests to the server.

```java
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import prediction.PredictionServiceGrpc;
import prediction.PredictionRequest;
import prediction.PredictionResponse;
import java.util.List;

public class PredictionFunction implements FlatMapFunction<List<Double>, Double> {
    private  transient PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;
    private final String rpcHost;
    private final int rpcPort;

    public PredictionFunction(String rpcHost, int rpcPort) {
        this.rpcHost = rpcHost;
        this.rpcPort = rpcPort;
    }

    @Override
    public void flatMap(List<Double> value, Collector<Double> out) throws Exception {
        if (blockingStub == null) {
            ManagedChannel channel = ManagedChannelBuilder.forAddress(rpcHost, rpcPort)
                .usePlaintext()
                .build();
            blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
        }

       PredictionRequest request = PredictionRequest.newBuilder()
              .addAllFeatures(value)
              .build();

        PredictionResponse response = blockingStub.predict(request);
        out.collect(response.getPrediction());
    }

}
```

here, the `flatmapfunction` receives the data from flink, converts it to a protobuf request, sends it to the python server, and collects the response which is the prediction. a simple `mapfunction` would also work fine if you need just one value output for every input. the `transient` keyword prevents the stub from being serialized by flink. remember to include the correct `grpc` dependencies in your flink project as well, they are not included in the flink libraries themselves.

finally, putting it all together in your flink job:

```java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import java.util.Arrays;
import java.util.List;


public class FlinkPredictionJob {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        // mock stream, should be your real data source
        DataStream<List<Double>> inputStream = env.fromCollection(Arrays.asList(
              Arrays.asList(1.0, 2.0, 3.0, 4.0),
              Arrays.asList(5.0, 6.0, 7.0, 8.0),
              Arrays.asList(9.0, 10.0, 11.0, 12.0)
         ));


        String rpcHost = "localhost";  // change if your server is elsewhere
        int rpcPort = 50051;       // match your server port

        DataStream<Double> predictions = inputStream
                .flatMap(new PredictionFunction(rpcHost, rpcPort));


        predictions.print(); // or send to a sink


        env.execute("Flink Prediction Job");
    }
}
```

this code sets up a simple flink job that gets data, uses the `predictionfunction` to send the data to your python server, and prints the results. this of course would be replaced by a real data stream source like kafka or the likes.

a few things to keep in mind, for performance it's best to pre-process and format the data in flink as close as possible to what the model expects. also, you might want to batch your requests to the server if latency is a concern. it is not shown here for simplicity, but you would accumulate data in a flink windows and then send the whole batch at once. that is a common optimization technique.

for the resources, i would highly recommend "designing data-intensive applications" by martin kleppmann. it's not about machine learning per se, but it covers distributed systems and remote calls very well. also, you may want to check grpc documentation itself for the communication part.

i remember once i got my model working but the predictions where just constant, turns out i was pre-processing the data wrong in my flink job. the server was not the issue it was the preprocessing steps, which got me thinking, isn't it funny, how many times we troubleshoot a complex system just to find a simple mistake? like putting the milk first in your cereal. anyway, these things happen.

this setup, while a bit complex to get started, makes for a robust and scalable way to run your python models in flink. if you run into specific errors, feel free to post them, and i am certain we can help you resolve it. good luck and happy coding.
