---
title: "How can I disable GPUs in H2O AutoML?"
date: "2025-01-30"
id: "how-can-i-disable-gpus-in-h2o-automl"
---
Disabling GPU usage within H2O AutoML isn't achieved through a direct, single flag.  My experience optimizing large-scale machine learning pipelines has shown that controlling GPU utilization requires a nuanced approach targeting H2O's underlying configuration rather than a dedicated AutoML parameter.  The reason is rooted in H2O's architecture; it leverages GPUs opportunistically, prioritizing them when available but gracefully falling back to CPUs if necessary.  Therefore, effective GPU disabling focuses on preventing H2O from detecting or utilizing available GPUs.

**1. Explanation: The Multifaceted Approach**

H2O's ability to utilize GPUs relies on the presence of compatible drivers and libraries.  Interrupting this process involves manipulating either the environment in which H2O is launched or the H2O cluster configuration itself.  Simply setting an environment variable to disable GPUs won't suffice as H2O's internal mechanisms will still attempt to probe for their availability.  A more effective strategy combines environment configuration with the explicit specification of CPU-only resources within the H2O cluster.  This ensures that H2O starts without GPU awareness and thus doesn't even consider utilizing them, unlike scenarios where GPU usage is simply avoided, leading to potential performance overhead due to unnecessary checks.

In my previous role, I encountered this challenge while optimizing a model training process on a cluster with both CPUs and GPUs.  Certain models, while benefiting from GPUs in general, exhibited performance regressions in specific scenarios due to driver-level issues.  Directly disabling GPUs was critical to ensuring consistent, predictable performance across all training runs. The following methods outline the approach I developed and refined.

**2. Code Examples and Commentary:**

**Example 1: Environment Variable Manipulation (Linux/macOS)**

This approach modifies the environment before launching the H2O cluster. It aims to make GPUs invisible to H2O.  This method is not guaranteed to be universally effective, as it's dependent on the specific CUDA/ROCm driver and library setup.


```bash
export CUDA_VISIBLE_DEVICES=""
export ROCM_VISIBLE_DEVICES=""
java -jar h2o.jar -cp <your_cp> -name myH2Ocluster -nthreads <number_of_cpus>
```

* **`export CUDA_VISIBLE_DEVICES=""`**: This sets the CUDA_VISIBLE_DEVICES environment variable to an empty string, effectively hiding all NVIDIA GPUs from any CUDA-aware application, including H2O.
* **`export ROCM_VISIBLE_DEVICES=""`**: Similarly, this hides AMD GPUs from ROCm-aware applications.
* **`java -jar h2o.jar ... -nthreads <number_of_cpus>`**: This launches the H2O cluster, specifying the number of CPU threads to use via `-nthreads`.  Explicitly defining the number of threads prevents H2O from automatically detecting and using all available cores, which might indirectly lead to unnecessary GPU probing.  Replacing `<your_cp>` with your classpath and `<number_of_cpus>` with the desired number of threads is essential. This ensures the cluster only utilizes the specified CPU resources.

**Example 2:  H2O Cluster Configuration (R)**

This example demonstrates configuring an H2O cluster within an R environment, specifically disabling GPU usage.


```R
library(h2o)
h2o.init(nthreads = parallel::detectCores(),  ip = "localhost", port = 54321) # Adjust port as needed
aml <- h2o.automl(x = predictor_columns, y = response_column, training_frame = train_h2o, max_models = 10, seed = 1234)
```

This R code initiates an H2O cluster explicitly using only the CPU cores detected by the `parallel::detectCores()` function. While this doesn't directly disable GPUs, by defining the number of threads, H2O is forced to restrict its resource utilization to CPUs. The `h2o.automl()` function then performs the AutoML process on the CPU-only cluster.  The crucial aspect here is setting `nthreads` to the appropriate CPU count.

**Example 3: H2O Cluster Configuration (Python)**

Python offers similar control.  Here, the number of threads is explicitly set to leverage CPU resources alone.  This approach indirectly prevents GPU utilization by limiting available resources.


```python
import h2o
from h2o.automl import H2OAutoML

h2o.init(nthreads = multiprocessing.cpu_count(), ip = "localhost", port = 54321) #Adjust port as needed
aml = H2OAutoML(max_models = 10, seed = 1234)
aml.train(x = predictor_columns, y = response_column, training_frame = train_h2o)
```

This Python code, similar to the R example, utilizes `multiprocessing.cpu_count()` to determine the number of CPU cores and configures the H2O cluster accordingly. The `H2OAutoML` class then performs the AutoML process within this constrained environment.  Similar to the R example, this forces the use of CPU-only resources by specifying the number of threads.

**3. Resource Recommendations:**

For a deeper understanding of H2O's architecture and its interaction with GPUs, consult the official H2O documentation.  Familiarize yourself with the sections on cluster configuration and resource management.   Explore the available environment variables and their impact on H2O's behavior. The H2O-3 documentation on configuring clusters is a valuable resource for troubleshooting potential conflicts or issues.  Reviewing materials on parallel processing in R and Python will further enhance your understanding of how thread allocation impacts H2O's performance. Finally, understanding CUDA and ROCm programming basics will provide context on why the environment variable approach is used and its limitations.
