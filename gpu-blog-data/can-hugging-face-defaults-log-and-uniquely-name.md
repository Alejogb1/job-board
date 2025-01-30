---
title: "Can Hugging Face defaults log and uniquely name all MLflow artifacts?"
date: "2025-01-30"
id: "can-hugging-face-defaults-log-and-uniquely-name"
---
Hugging Face's integration with MLflow, while powerful, doesn't inherently provide mechanisms to guarantee universally unique naming and logging of all artifacts without careful configuration. I've experienced situations where default behaviors lead to artifact collisions and overwrites, especially when training multiple model variants or using distributed training setups. The core issue stems from Hugging Face's *Trainer* class and its default artifact logging approach, which relies heavily on generic filenames and directory structures. MLflow, on the other hand, manages its artifact storage based on run IDs and paths, and without explicit guidance, the two systems can clash.

The central problem is that, by default, Hugging Face's *Trainer* saves checkpoint files (model weights, training arguments, etc.) within a subdirectory named "checkpoint-*". This is a common pattern but lacks the context needed to create globally unique identifiers when considered outside a single, isolated training run. MLflow's autologging, though generally helpful, won't automatically remedy this. It essentially captures what the *Trainer* saves, using those potentially overlapping names. Furthermore, when using different training procedures within the same project, the 'checkpoint' naming strategy could result in overwriting artifact information as MLflow manages all the artifacts via the run ID which only increments upon starting a brand new run. This issue is further amplified when running multiple experiments under the same project, each with multiple training runs potentially writing to the same checkpoint locations. The crucial task, therefore, is to intercept the artifact saving process and inject our own custom logic to ensure uniquely named artifacts within the MLflow tracking environment.

My primary strategy for accomplishing this has revolved around leveraging the callback mechanism of Hugging Face's *Trainer*. Instead of relying solely on the default behavior, I create custom callbacks to override the standard save processes and directly interact with MLflow. This allows me to generate unique paths for artifacts using the MLflow run ID, timestamp, model name, and even specific training parameters if needed. I've found it essential to carefully identify the specific points in the *Trainer* lifecycle where artifacts are created and to integrate the unique naming logic there. I typically augment the `on_save`, `on_train_begin`, and `on_train_end` methods of my custom callback class to get complete control over artifact persistence.

Below are three code examples showcasing the steps I use to achieve deterministic, uniquely named artifact logging within the Hugging Face and MLflow ecosystem:

**Example 1: A Basic Custom Callback**

This example defines a simple callback to add a timestamp to the checkpoint name. It illustrates the general pattern of intercepting the `on_save` event.

```python
import os
import time
from transformers import TrainerCallback
from mlflow import log_artifact

class UniqueCheckpointCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_save(self, args, state, control, **kwargs):
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      output_dir = args.output_dir
      
      for filename in os.listdir(output_dir):
        if "checkpoint" in filename:
          src = os.path.join(output_dir, filename)
          dest = os.path.join(output_dir, f"{filename}_{timestamp}")

          os.rename(src, dest)
          for file in os.listdir(dest):
            file_path = os.path.join(dest,file)
            log_artifact(file_path)
```

This callback extracts the current timestamp and appends it to the checkpoint directory name. Note that `os.rename` directly modifies the save path. It then logs the whole checkpoint with `log_artifact`, not individual files. This is important to avoid excessive logging and preserve checkpoint structure. It should be passed into the trainer object using the `callbacks` argument.

**Example 2: Augmenting with Run ID and Model Name**

Building upon the first example, this callback enhances the artifact naming using the MLflow run ID and model name. This offers a greater degree of specificity.

```python
import os
import time
from transformers import TrainerCallback
from mlflow import log_artifact
from mlflow import active_run

class UniqueCheckpointCallbackV2(TrainerCallback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def on_save(self, args, state, control, **kwargs):
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      run_id = active_run().info.run_id
      output_dir = args.output_dir

      for filename in os.listdir(output_dir):
        if "checkpoint" in filename:
          src = os.path.join(output_dir, filename)
          dest = os.path.join(output_dir, f"{self.model_name}-{run_id}-{filename}_{timestamp}")
          
          os.rename(src,dest)

          for file in os.listdir(dest):
            file_path = os.path.join(dest,file)
            log_artifact(file_path)
```

Here, I've introduced the model name as a constructor argument to further differentiate the saved artifacts. I retrieve the active MLflow run ID to provide a high level of uniqueness. The file path generation logic is otherwise identical, but it now incorporates the run-specific information that’s required to solve the initial problem.

**Example 3: Full Integration with `on_train_end` and Individual File Logging**

This example illustrates how to move the artifact logging process into `on_train_end` for a final checkpoint upload. This further improves uniqueness, but also handles the potential for saving artifacts individually to log metadata alongside model weights.

```python
import os
import time
from transformers import TrainerCallback
from mlflow import log_artifact, log_metric
from mlflow import active_run
import json


class UniqueCheckpointCallbackV3(TrainerCallback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_dir = None


    def on_save(self, args, state, control, **kwargs):
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      output_dir = args.output_dir

      for filename in os.listdir(output_dir):
        if "checkpoint" in filename:
          self.checkpoint_dir = filename
          src = os.path.join(output_dir, filename)
          dest = os.path.join(output_dir, f"{filename}_{timestamp}")
          os.rename(src, dest)
          
    
    def on_train_end(self, args, state, control, **kwargs):
      if self.checkpoint_dir is None:
        return
      
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      run_id = active_run().info.run_id
      
      output_dir = args.output_dir
      
      final_dest = os.path.join(output_dir, f"{self.model_name}-{run_id}-{self.checkpoint_dir}_{timestamp}")
      os.rename(os.path.join(output_dir,f"{self.checkpoint_dir}_{timestamp}"), final_dest)
        
      for filename in os.listdir(final_dest):
          file_path = os.path.join(final_dest, filename)
          log_artifact(file_path)

      log_metric("final_checkpoint_size", sum([os.path.getsize(os.path.join(final_dest,file)) for file in os.listdir(final_dest)]))

```

Here the checkpoint directory naming is done at each checkpoint save, but the logging is delayed until the end of training, meaning that all intermediate checkpoints are not logged by default. When `on_train_end` is called, the final logging logic is run by renaming the saved checkpoint directory one last time using the `run_id` and logging all artifacts found within the checkpoint directory. This has been a preferred strategy since it reduces overall clutter in artifact storage, and keeps the MLflow tracking UI concise. Additionally, I added an example of logging metadata with the artifacts by logging the overall checkpoint size as an MLflow metric, which has been a very useful strategy in understanding overall storage costs for experiments.

In my experience, the methods outlined above have effectively addressed the default artifact naming limitations of Hugging Face *Trainer* within the MLflow ecosystem.  To improve the experience further, I suggest considering implementing more robust checks to ensure the generated filenames are truly unique in the face of extremely high levels of parallelism. Moreover, it’s crucial to carefully manage the artifact storage location. A poorly structured bucket or volume can lead to confusion and performance issues.

For additional resources, I recommend reviewing the official Hugging Face Transformers documentation on callbacks, and the MLflow documentation concerning artifact logging and tracking. Also, exploring advanced Python file manipulation methods can be beneficial when dealing with complex artifact path requirements. I have found that examining examples of other practitioners facing similar problems on platforms such as GitHub or forums dedicated to machine learning engineering often provides valuable insights and practical solutions. Careful integration of these resources has enabled the development of robust, scalable ML systems.
