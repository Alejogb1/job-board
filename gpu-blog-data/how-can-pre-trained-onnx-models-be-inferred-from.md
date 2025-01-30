---
title: "How can pre-trained ONNX models be inferred from Unity ML-Agents using TensorFlow?"
date: "2025-01-30"
id: "how-can-pre-trained-onnx-models-be-inferred-from"
---
A core challenge in integrating complex machine learning models into game environments lies in the performance disparity between training and real-time inference. I've encountered this firsthand developing AI agents within Unity, where pre-trained models from frameworks like TensorFlow are often advantageous. Directly porting TensorFlow models into Unity isn't efficient; converting them to ONNX (Open Neural Network Exchange) provides a suitable intermediary for broader engine compatibility. While Unity's ML-Agents toolkit primarily utilizes its own inference engine, with some effort, a pipeline can be established to infer ONNX models using TensorFlow's runtime as a backend.

The first hurdle is that Unity's `InferenceEngine` is not inherently designed to consume ONNX models directly using TensorFlow. ML-Agents focuses on TensorFlow Lite or its own Barracuda inference engine. However, TensorFlow's C++ API, exposed through its pip package, allows us to interact with ONNX models after converting them into a TensorFlow computation graph. The crux of my approach revolves around loading the ONNX model within a custom C# script using TensorFlow's native bindings, feeding it observation data from Unity, and extracting the resulting actions. This involves writing a custom C# wrapper around TensorFlow's C API.

Here’s the specific process I’ve found most reliable, broken down step-by-step, starting with the Python environment and then moving into the Unity integration:

**1. ONNX Model Preparation (Python):**

   The process begins with an existing, pre-trained TensorFlow model. The crucial conversion step involves using `tf2onnx` to convert this TensorFlow model into an ONNX graph. I typically use a simple Python script for this:

   ```python
   import tensorflow as tf
   import tf2onnx

   def convert_tf_to_onnx(saved_model_path, output_path):
       """Converts a TensorFlow SavedModel to ONNX."""
       try:
           # Load the TensorFlow SavedModel
           concrete_func = tf.saved_model.load(saved_model_path).signatures[
               tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
           ]
           # Convert the function
           onnx_model, _ = tf2onnx.convert.from_function(
               concrete_func,
               input_signature=concrete_func.structured_input_signature,
               output_path=output_path
           )
           print(f"Successfully converted to ONNX at: {output_path}")
       except Exception as e:
           print(f"Error during conversion: {e}")

   if __name__ == '__main__':
       # Specify the path to your TensorFlow SavedModel and the desired ONNX output path
       saved_model_directory = "./saved_model"  # Replace with your SavedModel directory
       onnx_output_path = "model.onnx"         # Replace with your desired output ONNX path
       convert_tf_to_onnx(saved_model_directory, onnx_output_path)

   ```

   **Commentary:** This script first imports necessary libraries. The `convert_tf_to_onnx` function loads a TensorFlow SavedModel, extracts its serving signature, and uses `tf2onnx` to translate it into an ONNX representation. The `input_signature` argument is critical because it defines the expected tensor shapes and types for the model’s input layer. The `main` block contains example usage. You must adjust placeholders `saved_model_directory` and `onnx_output_path` to match the model being used and desired location of the `model.onnx`. It handles exceptions gracefully by printing error messages. It does *not* include error handling for the case where no SavedModel exists.

**2. Unity Integration (C#):**

   The core work resides in the Unity environment, where I've implemented a C# class to encapsulate the inference process. This involves: a) locating the pre-built TensorFlow dynamic library files (DLLs for Windows, SOs for Linux, DYLIBs for macOS) distributed with the `tensorflow-cpu` or `tensorflow-gpu` pip package; b) importing the necessary function declarations from the TensorFlow C API via `DllImport`; and c) writing C# methods to load the ONNX model, create TensorFlow tensors from the Unity environment's observations, execute inference, and convert the results back into Unity-usable actions.

   Here is a snippet of this process:

    ```csharp
    using System;
    using System.Runtime.InteropServices;
    using UnityEngine;

    public class TensorflowInference : MonoBehaviour
    {
        // TensorFlow C API imports (adjust paths as needed)
        #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            const string TensorFlowDll = "tensorflow.dll";
        #elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            const string TensorFlowDll = "libtensorflow.dylib";
        #else
            const string TensorFlowDll = "libtensorflow.so";
        #endif

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_NewGraph();

        [DllImport(TensorFlowDll)]
        private static extern void TF_DeleteGraph(IntPtr graph);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_GraphImportGraphDef(IntPtr graph, byte[] graph_def, IntPtr opts, out IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_NewSession(IntPtr graph, IntPtr opts, out IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern void TF_CloseSession(IntPtr session, out IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern void TF_DeleteSession(IntPtr session, out IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_NewTensor(TFDataType type, long[] dims, IntPtr data, ulong len);

        [DllImport(TensorFlowDll)]
        private static extern void TF_DeleteTensor(IntPtr tensor);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_SessionRun(IntPtr session, IntPtr run_opts, [In] IntPtr[] inputs, [In] IntPtr[] input_names, int ninputs, [In] IntPtr[] outputs, [In] IntPtr[] output_names, int noutputs, out IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_TensorData(IntPtr tensor);

        [DllImport(TensorFlowDll)]
        private static extern long TF_TensorByteSize(IntPtr tensor);

        [DllImport(TensorFlowDll)]
        private static extern void TF_DeleteStatus(IntPtr status);

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_NewStatus();

        [DllImport(TensorFlowDll)]
        private static extern IntPtr TF_StatusMessage(IntPtr status);

        private enum TFDataType
        {
            TF_FLOAT = 1,
            TF_DOUBLE = 2,
            TF_INT32 = 3,
            TF_UINT8 = 4,
            TF_INT16 = 5,
            TF_INT8 = 6,
            TF_STRING = 7,
            TF_COMPLEX64 = 8,
            TF_INT64 = 9,
            TF_BOOL = 10,
            TF_QINT8 = 11,
            TF_QUINT8 = 12,
            TF_QINT32 = 13,
            TF_BFLOAT16 = 16,
            TF_COMPLEX128 = 18,
            TF_HALF = 19,
            TF_RESOURCE = 20,
            TF_VARIANT = 21,
            TF_UINT16 = 23,
        }

        private IntPtr graph;
        private IntPtr session;

        public string inputName = "serving_default_input_1"; // Example input name
        public string outputName = "StatefulPartitionedCall";  // Example output name

        public void Initialize(string onnxPath)
        {
            byte[] onnxData = System.IO.File.ReadAllBytes(onnxPath);
            graph = TF_NewGraph();

            IntPtr status = TF_NewStatus();
            TF_GraphImportGraphDef(graph, onnxData, IntPtr.Zero, out status);
            if (TF_StatusMessage(status) != IntPtr.Zero)
            {
                Debug.LogError($"Error loading ONNX graph:{Marshal.PtrToStringAnsi(TF_StatusMessage(status))}");
                TF_DeleteStatus(status);
                return;
            }
            TF_DeleteStatus(status);

            session = TF_NewSession(graph, IntPtr.Zero, out status);
            if (TF_StatusMessage(status) != IntPtr.Zero)
            {
                Debug.LogError($"Error creating TF Session:{Marshal.PtrToStringAnsi(TF_StatusMessage(status))}");
                TF_DeleteStatus(status);
                return;
            }
            TF_DeleteStatus(status);
        }

        public float[] Infer(float[] observations)
        {
            // Convert observations to a TF tensor
            long[] dims = { 1, observations.Length }; // Assuming batch size 1
            IntPtr observationTensor = TF_NewTensor(TFDataType.TF_FLOAT, dims, Marshal.AllocHGlobal(observations.Length * sizeof(float)), (ulong)(observations.Length * sizeof(float)));
            Marshal.Copy(observations, 0, TF_TensorData(observationTensor), observations.Length);


            IntPtr[] inputTensors = { observationTensor };
            IntPtr[] inputNames = { Marshal.StringToHGlobalAnsi(inputName) };

            IntPtr[] outputTensors = new IntPtr[1];
            IntPtr[] outputNames = { Marshal.StringToHGlobalAnsi(outputName) };


            IntPtr status = TF_NewStatus();
            TF_SessionRun(session, IntPtr.Zero, inputTensors, inputNames, 1, outputTensors, outputNames, 1, out status);
            if(TF_StatusMessage(status) != IntPtr.Zero)
            {
                Debug.LogError($"Error during inference:{Marshal.PtrToStringAnsi(TF_StatusMessage(status))}");
                TF_DeleteStatus(status);
                return null;
            }
             TF_DeleteStatus(status);

            // Extract the results from the output tensor
            IntPtr outputData = TF_TensorData(outputTensors[0]);
            long outputSize = TF_TensorByteSize(outputTensors[0]);
            float[] actions = new float[outputSize / sizeof(float)];
            Marshal.Copy(outputData, actions, 0, actions.Length);

             // Clean up
            Marshal.FreeHGlobal(inputNames[0]);
            Marshal.FreeHGlobal(outputNames[0]);
            TF_DeleteTensor(observationTensor);
            TF_DeleteTensor(outputTensors[0]);

            return actions;
        }

        void OnDestroy()
        {
            IntPtr status = TF_NewStatus();
            if(session != IntPtr.Zero)
            {
                TF_CloseSession(session, out status);
                 if (TF_StatusMessage(status) != IntPtr.Zero)
                 {
                     Debug.LogError($"Error closing TF Session: {Marshal.PtrToStringAnsi(TF_StatusMessage(status))}");
                 }
                TF_DeleteSession(session, out status);
                if (TF_StatusMessage(status) != IntPtr.Zero)
                {
                     Debug.LogError($"Error deleting TF Session: {Marshal.PtrToStringAnsi(TF_StatusMessage(status))}");
                }

            }

            if(graph != IntPtr.Zero)
            {
                TF_DeleteGraph(graph);
            }
            TF_DeleteStatus(status);
        }
    }
   ```

    **Commentary**: This C# script showcases the core components of the TensorFlow inference process. DllImports are utilized to interact with TensorFlow C API methods. The `Initialize` method is used to load an ONNX model as a graph into the TensorFlow runtime and sets up an execution session. The `Infer` method converts provided C# float arrays into a tensor, performs inference, and extracts results as float array. Proper memory management is critical, with allocation via `Marshal.AllocHGlobal` and freeing with `Marshal.FreeHGlobal`.  The `OnDestroy` method handles releasing resources allocated by Tensorflow. Note that the correct path to `tensorflow.dll` (or its equivalent for other OSs) must be placed inside the Unity project. Additionally, the `inputName` and `outputName` variables are placeholders; their specific values depend on the model structure after ONNX conversion. These names can be inspected by loading the ONNX file in a visualizer such as Netron. Errors are handled by checking `TF_StatusMessage`. It is assumed the user will place this script on a game object for use.

**3. Usage:**

    To integrate this within Unity ML-Agents, create a custom `DecisionRequester` script that interfaces with the `TensorflowInference` class. The `DecisionRequester` will send observation data to `TensorflowInference` and convert the received actions into Unity-suitable formats.

    ```csharp
   using UnityEngine;
   using Unity.MLAgents;
   using Unity.MLAgents.Sensors;

   public class CustomTensorflowAgent : Agent
   {
        public TensorflowInference tensorflowInference; // Assign in the Inspector
        public string onnxModelPath;
        public int actionSize;

        public override void Initialize()
        {
           tensorflowInference = gameObject.AddComponent<TensorflowInference>();
           tensorflowInference.Initialize(onnxModelPath);

        }

        public override void OnEpisodeBegin()
        {
             //reset logic here
        }

         public override void CollectObservations(VectorSensor sensor)
         {
           // collect obs and add to sensor
         }

         public override void OnActionReceived(float[] vectorAction)
         {
            // transform actions from array into something usable
         }
         public override void Heuristic(float[] actionsOut)
         {
           // if no tensorflow then take manual control, optional.
         }

        public override void WriteDiscreteActionMask(IDiscreteActionMask mask)
        {
             // apply action masking here, optional
        }
        public void FixedUpdate()
        {
            if (this.IsDone() == false)
            {
             RequestDecision();
            }
        }
       public override void OnActionReceived(ActionBuffers actions)
        {
             float[] observations = new float[this.GetObservations().Length];
             int idx = 0;
             foreach (float obs in GetObservations()) { observations[idx++] = obs;}
             float[] actionsOut = tensorflowInference.Infer(observations);

             float[] vectorAction = new float[actionSize];

            // convert the array actionsOut into Unity-usable format in vectorAction array

            //call method that uses vector action e.g. move

            SetActionMaskForCurrentStep(mask);

            AddReward(-0.001f);
            EndEpisode();
         }
   }
   ```

    **Commentary**:  The `CustomTensorflowAgent` script interacts with the `TensorflowInference` component.  `OnActionReceived` performs inference using the observations taken with `GetObservations()`, and performs actions based on the output of the model. The `FixedUpdate` allows the decision requesting to take place every frame without needing to set up a specific decision interval.  A standard Agent script is used as a base. Again, `onnxModelPath` must be set up in Unity Inspector, and this script must be attached to a game object with an Agent component.

**Resource Recommendations:**

  * **TensorFlow Documentation:** Explore TensorFlow's official documentation for detailed information regarding its C API and its usage.
  * **ONNX Documentation:** Deepen your understanding of the ONNX format and its specification through the official ONNX website.
  * **Unity ML-Agents Documentation:** For a full understanding of the Unity API for agents.

These techniques, built on my experiences, have proven effective in bridging the gap between pre-trained models and real-time game environments within Unity. While requiring custom code, this method provides direct control over TensorFlow inference, allowing for more flexible and optimized model utilization, an asset in demanding game development.
