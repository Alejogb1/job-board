---
title: "How to disable smoothing in MediaPipe pose estimation landmarks (Unity)?"
date: "2025-01-30"
id: "how-to-disable-smoothing-in-mediapipe-pose-estimation"
---
Disabling smoothing in MediaPipe pose estimation, particularly when used within Unity, requires understanding the underlying processes and available configuration options within the MediaPipe framework itself. The landmarks outputted by MediaPipe are, by default, temporally smoothed using a Kalman filter or a similar mechanism. This smoothing aims to reduce jitter and produce a more stable output across frames. However, this inherent smoothing can be undesirable in specific applications that require low-latency or raw landmark data. I've encountered scenarios, particularly in real-time interaction systems, where the smoothing artifacts negatively impacted the precision of the application. Direct control over the smoothing process, while not exposed directly in high-level Unity MediaPipe integrations, can be achieved by manipulating the MediaPipe graph configuration.

The crucial aspect to grasp is that MediaPipe doesn't offer a simple boolean flag to disable smoothing. Instead, it implements temporal filtering through a dedicated calculator, and to disable it, we must either bypass this calculator or configure it to provide essentially no smoothing. These calculators operate within the MediaPipe graph, a declarative structure that defines the flow of data and processing steps. In the context of pose estimation, the smoothing usually occurs *after* the core pose detection process. Therefore, manipulation requires directly addressing this part of the graph.

The most effective approach I've found involves modifying the text proto file that defines the MediaPipe graph, which is usually an extension of the standard pose estimation graph provided by MediaPipe. While MediaPipe for Unity typically encapsulates this within its plugin, the underlying structure remains accessible. The standard pose estimation graph contains a `KalmanFilterCalculator`. To disable smoothing, we essentially want to bypass this calculator’s temporal filtering behavior. We achieve this by modifying the `options` within the Kalman filter configuration and potentially also adjusting its input and output connections.

Here are the strategies I typically use, reflected through concrete code examples based on my past implementations:

**Example 1: Bypassing the KalmanFilterCalculator**

This example assumes we can directly access and modify the graph configuration file (`pose_landmark_cpu.pbtxt`, a typical filename for the pose estimation graph). This often requires unpacking the assets provided by the MediaPipe Unity integration. I have had to do this when the pre-configured options did not meet requirements.  The `pbtxt` file resembles a configuration script, not a traditional source code file.

```protobuf
# Simplified example showing a relevant section of the original pose_landmark_cpu.pbtxt
# ... other graph declarations ...

node {
  calculator: "KalmanFilterCalculator"
  input_stream: "LANDMARK:filtered_landmarks"
  input_stream: "FRAME_TIME:frame_timestamp"
  output_stream: "LANDMARK:smoothed_landmarks"
}


node {
  calculator: "PassThroughCalculator"  # A calculator that passes input unchanged
  input_stream: "LANDMARK:filtered_landmarks"
  output_stream: "LANDMARK:smoothed_landmarks"
}

# node { # original node, removed
#   calculator: "KalmanFilterCalculator"
#   input_stream: "LANDMARK:filtered_landmarks"
#   input_stream: "FRAME_TIME:frame_timestamp"
#   output_stream: "LANDMARK:smoothed_landmarks"
# }


# ... other graph declarations ...
```

**Commentary:**

In this example, I've effectively removed the original `KalmanFilterCalculator` node by commenting it out. Then, I introduced a `PassThroughCalculator`,  redirecting the output stream previously destined for the Kalman filter to this new node. The `PassThroughCalculator` simply relays the input landmark data without any alteration, thus removing the smoothing effect. It is essential to ensure that all input/output streams match after modifying the nodes. This is the most direct method to remove the Kalman filter.  While it does remove the smoothing, in practice it may require further adjustment to make sure all streams match. This technique requires replacing existing calculators with our own, which is not possible without modifying the configuration files. It's generally safe when dealing with relatively simple modifications, but care should be taken with more complex graphs.

**Example 2: Minimal Smoothing Configuration**

Instead of bypassing the Kalman filter altogether, you can configure it to minimize its smoothing effect. This involves adjusting parameters within the calculator’s options. Again, this requires access to the `pbtxt` file.

```protobuf
#Simplified example showing modified KalmanFilterCalculator options

node {
  calculator: "KalmanFilterCalculator"
  input_stream: "LANDMARK:filtered_landmarks"
  input_stream: "FRAME_TIME:frame_timestamp"
  output_stream: "LANDMARK:smoothed_landmarks"
  options {
    [mediapipe.KalmanFilterCalculatorOptions.ext] {
      process_noise: 10.0      # Increase noise to make it trust observations more
      measurement_noise: 0.01   # Reduce observation noise confidence
      state_transition_noise: 10.0  # Increase the transition noise
      min_time_delta: 0.0001 # Reduces influence of past frames
    }
  }
}
```

**Commentary:**

Within the `KalmanFilterCalculator` node, the `options` field is expanded to access the `KalmanFilterCalculatorOptions`. Here, I've significantly increased the `process_noise` and `state_transition_noise` values. By doing so, we are telling the Kalman filter that the predicted state of the landmarks is very uncertain and that new measurements should be given more weight. I have also reduced the `measurement_noise` which means that the sensor data coming in has low noise and should also be trusted by the Kalman filter.  The combination of high process/transition noise and low measurement noise forces the filter to rely primarily on the current observation, minimizing smoothing. The `min_time_delta` parameter is reduced to reduce the impact of past frames by making the system consider any two frames to be distinct with very little reliance on previous measurements.   This approach allows you to keep the Kalman filter in the graph but heavily diminish its effect.  Tweaking these specific values requires some experimentation for the user's particular use case.

**Example 3: Configuring Different Calculators for Smoothing**

In some cases, you may find that instead of a `KalmanFilterCalculator`, another calculator like a `MovingAverageCalculator` is used for smoothing in the graph. The principle for disabling remains similar. Here is how you could modify the parameters of such a calculator.

```protobuf
#Simplified example showing modified MovingAverageCalculator options

node {
  calculator: "MovingAverageCalculator"
  input_stream: "LANDMARK:filtered_landmarks"
  output_stream: "LANDMARK:smoothed_landmarks"
    options {
        [mediapipe.MovingAverageCalculatorOptions.ext] {
            window_size: 1  #Set the window size to just 1 frame to only take current values
          }
    }
}

```

**Commentary:**

Here I’ve modified the `MovingAverageCalculator` which uses a simple average to provide smoothing. Setting `window_size` to 1 effectively means that the average is calculated over one frame and produces the same result as the input value. So, each frame is effectively the average of just itself resulting in no smoothing. This is similar to the effect of the `PassThroughCalculator` earlier, however, it achieves it by configuring the calculator rather than replacing it. Like the parameters for the Kalman Filter, the `window_size` parameter requires experimentation depending on the specific use case.

**Resource Recommendations:**

*   **MediaPipe Documentation:**  The official MediaPipe documentation, while not Unity-specific, provides detailed explanations of the graph structure, available calculators, and configuration options. Understanding the underlying concepts from the source is essential.
*   **MediaPipe Examples:**  Examining the example graphs that come with MediaPipe's pose estimation implementations can reveal which calculators are involved in smoothing. This provides practical starting points for analysis and adjustment. The examples also give concrete uses of the calculators and their options.
*   **Kalman Filtering Theory:** Understanding the fundamental principles of Kalman filtering will help in interpreting the parameters within the KalmanFilterCalculator and make more informed decisions. Resources that focus on practical applications of Kalman filters are generally the most useful for these kinds of situations.
*   **Graph Description Language (protobuf):**  Familiarity with protobuf is necessary for reading and modifying MediaPipe graph configuration files, typically stored as `pbtxt` files.

In conclusion, while disabling smoothing in MediaPipe pose estimation in Unity is not directly provided as a feature flag, understanding and modifying the MediaPipe graph configuration is feasible and necessary for certain real-time application scenarios. By either bypassing the smoothing calculator or adjusting its parameters, it's possible to achieve a balance between stability and low-latency performance, tailoring the framework to meet the specific requirements of diverse applications. Direct manipulation of the protobuf file is generally the only way to achieve the needed modifications. The exact configuration can vary depending on the specific implementation or version of the MediaPipe pose estimation.
