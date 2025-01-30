---
title: "Why is Anaconda installing TensorFlow 1.15 instead of 2.0?"
date: "2025-01-30"
id: "why-is-anaconda-installing-tensorflow-115-instead-of"
---
The root cause of Anaconda installing TensorFlow 1.15 instead of TensorFlow 2.0 often stems from environment inconsistencies and misconfigurations within the conda package manager.  My experience troubleshooting this issue across numerous data science projects has highlighted the critical role of environment specifications and channel prioritization in resolving this discrepancy.  A seemingly minor oversight in environment management can easily lead to the installation of an outdated TensorFlow version.

**1. Explanation:**

Anaconda's package management relies heavily on environment files (`.yml` or `.yaml`) and channel specifications. When creating a new environment, if an explicit TensorFlow version isn't specified, Anaconda defaults to the latest version available *within its configured channels*.  The default channels often contain older versions of TensorFlow alongside newer ones.  This is a deliberate strategy to support legacy projects relying on older versions. If the environment creation doesn't explicitly override the default behavior,  or if a conflicting dependency from an older environment is present, Anaconda might install TensorFlow 1.15 despite TensorFlow 2.0's broader availability.

Furthermore, the presence of a pre-existing TensorFlow 1.x environment can lead to dependency conflicts.  Even if you specify `tensorflow==2.0` in your environment file, if an earlier environment's dependencies conflict with the installation process, conda might choose the path of least resistance and install the compatible TensorFlow 1.15 version. This is because conda attempts to resolve dependencies across your entire system and prioritize compatibility.  It's important to realize conda isn't simply installing the latest package independent of other components; it's managing a holistic dependency tree.

Another important point is channel precedence. Anaconda can search across several channels, including the default conda-forge channel and potentially others you've added.  If a specific TensorFlow version exists in a higher-priority channel (such as a custom channel you added), Anaconda will install it regardless of what's available in other channels. This sometimes leads to unintended installations, particularly when users haven't explicitly managed their channel priorities.

Finally,  it's essential to distinguish between Python versions.  TensorFlow 2.0 requires a specific minimum Python version. If you are accidentally creating an environment with a Python version that is incompatible with TensorFlow 2.0, the installer might default to TensorFlow 1.15 as a fallback.


**2. Code Examples with Commentary:**

**Example 1: Correct Environment Creation**

```yaml
name: tf2_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8 #Ensure Python version compatibility with TF 2.0
  - tensorflow==2.8.0 #Explicitly specify TensorFlow version
  - pip #for installing packages not in conda channels
  - scikit-learn
  - numpy
  - matplotlib
```

This `environment.yml` file demonstrably addresses the most common problems. It specifies:

* `conda-forge`:  This is generally recommended for its well-maintained packages.
* `tensorflow==2.8.0`: This explicitly states the desired TensorFlow version, removing ambiguity.  Note I use a stable and commonly supported TensorFlow 2.x version, not 2.0 specifically.
* A compatible Python version (Python 3.8 in this case).  Check TensorFlow's documentation for compatibility information.


To create this environment, run: `conda env create -f environment.yml`


**Example 2:  Illustrating a Potential Conflict**

```bash
conda create -n tf_env python=3.7 tensorflow
```

This example is problematic. It lacks explicit version control for TensorFlow and uses a potentially incompatible Python version (3.7 may not support the latest TensorFlow 2.x versions). This increases the likelihood of installing TensorFlow 1.15 if it's a readily available compatible package in the default channels.

**Example 3: Removing Conflicting Environments**

```bash
conda env list #List your environments
conda env remove -n tf_env_old #Remove conflicting environment. Replace tf_env_old with the name.
```

Before creating a new environment, it's crucial to identify and remove any pre-existing environments that may contain conflicting TensorFlow versions or dependencies.  This step directly addresses the issue of dependency conflicts, ensuring a clean installation. Always carefully review the output of `conda env list` before executing `conda env remove`.


**3. Resource Recommendations:**

* Consult the official Anaconda documentation for comprehensive information on environment management and package installation.  Pay close attention to the sections on channels and dependency resolution.
* Refer to the official TensorFlow documentation for compatibility information between TensorFlow versions and Python versions. This is paramount in avoiding installation issues.
* Read the documentation for the `conda` package manager.  Understanding its inner workings concerning dependency resolution is crucial for efficient troubleshooting.


In my experience, combining careful environment creation using detailed `environment.yml` files with a clear understanding of conda's channel prioritization and dependency resolution mechanisms is the most reliable method for ensuring the correct TensorFlow version is installed.  Ignoring these aspects often results in the frustrating and unexpected installation of outdated versions.  Thorough attention to detail throughout the entire environment management workflow significantly reduces the likelihood of encountering this specific issue.
