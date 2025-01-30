---
title: "How can I install TensorFlow 1.X.X on Python 2.7 in Ubuntu?"
date: "2025-01-30"
id: "how-can-i-install-tensorflow-1xx-on-python"
---
TensorFlow 1.X versions present particular challenges for installation on legacy systems, especially when combined with Python 2.7, which is no longer officially supported. My direct experience managing legacy machine learning infrastructure leads me to consistently recommend virtual environments to mitigate conflicts between system-level Python installations and the specific requirements of older TensorFlow versions. This approach is essential given the incompatibilities arising from newer packages and libraries.

Installing TensorFlow 1.X on Python 2.7 within Ubuntu requires navigating dependency constraints and ensuring that the correct versions of supporting libraries are installed. This configuration is most commonly necessary when dealing with pre-existing models and applications that were built on the older framework and Python version. This process is not straightforward; incorrect versions of supporting libraries will render TensorFlow unusable. My experience has shown me that a step-by-step and carefully managed virtual environment setup reduces troubleshooting considerably.

The primary tool I employ is `virtualenv`. It allows for an isolated environment where the correct version of TensorFlow can be installed without affecting the system's global Python packages. The process begins by creating a virtual environment, activating it, then installing the proper TensorFlow wheel file. Given the age of the environment, it is imperative to locate the specific TensorFlow wheel file that aligns with both Python 2.7 and the available hardware. You must also consider if GPU acceleration is required.

First, I typically begin by installing `virtualenv` using `pip`:
```bash
sudo apt-get update
sudo apt-get install python-pip
sudo pip install virtualenv
```
This ensures that `virtualenv` is available on your system. After confirming its installation, I establish a new virtual environment for the TensorFlow 1.x setup:
```bash
virtualenv -p /usr/bin/python2.7 tf_1_x_env
```
Here, `/usr/bin/python2.7` explicitly specifies the Python 2.7 interpreter location. `tf_1_x_env` is the directory where the environment files are stored. Activating the environment is next:
```bash
source tf_1_x_env/bin/activate
```
Upon activating the virtual environment, the shell prompt will indicate that you are now operating within the `tf_1_x_env`. Any packages installed will reside within this contained environment. It should be noted that the exact path to `python2.7` may vary depending on system configuration; you should check the correct path beforehand, typically by using the command `which python2.7`. Failure to activate the environment will install packages to your global environment and can interfere with your system and with other packages.

The next stage involves installing the correct TensorFlow wheel. The required version depends on whether GPU support is needed and which specific TensorFlow 1.X version is required, typically 1.15.0 is the terminal version. The appropriate wheel needs to be downloaded manually from external sources. I have found these hosted on the TensorFlow GitHub repository release pages, although they can occasionally be found on other repositories. Since direct access to such resources is inappropriate here, I would strongly advise a thorough web search, using terms such as 'tensorflow 1.15.0 python 2.7 linux wheel file' or similar. Consider that compatibility with the CUDA version also needs to be checked if you plan to utilize GPU acceleration. The following examples presume you have a CPU-only wheel:
```bash
pip install tensorflow-1.15.0-cp27-cp27mu-linux_x86_64.whl
```
Assuming the file was located, this instruction installs the downloaded wheel using pip. The filename will vary depending on whether the package is compiled for the CPU only or with GPU acceleration, along with the exact TensorFlow version. I always double-check the installed TensorFlow version by running a small test program within the virtual environment.

Here is a Python code example to verify installation:
```python
import tensorflow as tf
print(tf.__version__)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
If the output corresponds to the TensorFlow 1.X version that you installed, and you observe "Hello, TensorFlow!", then the installation is successful. If not, check the previous steps again carefully. Verify the virtual environment is activated, and verify the correct wheel was installed. The Python code snippet provided here requires TensorFlow to function, so it's a reasonable test for correct installation.

Finally, if any other supporting libraries are required for the project, these will have to be installed within the virtual environment in a similar manner, considering versions specific to Python 2.7. For example, many projects may require `numpy`, `scipy` or `pandas`:
```bash
pip install numpy==1.16.0
pip install scipy==1.2.0
pip install pandas==0.24.0
```
It is vital that the `numpy` and other library versions are compatible with your chosen TensorFlow version. Versions beyond those specified here may fail, often due to ABI differences between Python versions, or other incompatibilities. A full list of version dependencies is not usually provided, but a degree of experimentation may be necessary. If any errors occur at the point of installation, check the error message provided by pip, which will often provide some clues as to the cause of the problem.

In conclusion, installing TensorFlow 1.X on Python 2.7 in Ubuntu involves creating an isolated virtual environment, activating the environment, and installing the correct TensorFlow wheel file. Other supporting packages must be installed with specific versions. This process is complex, and requires careful management of version dependencies. This approach minimizes conflicts and ensures that the legacy TensorFlow setup can function effectively. Troubleshooting should focus on version incompatibilities, incorrect paths, and ensuring the proper activation of the virtual environment. It is my experience that this process is reliable when carried out correctly, however careful attention to detail is paramount, and experimentation may be necessary.

For resource recommendations, I would suggest looking at documentation for `virtualenv` through a dedicated `virtualenv` project site. Also, TensorFlow documentation archives exist for older versions of the framework, although specific wheel file links are seldom directly provided. The PyPI website, where many Python packages are hosted, offers search functionality that can be of assistance. Finally, community forums, such as Stack Overflow, contain information on similar installation issues, but remember to exercise caution and check the solutions carefully. These community resources can sometimes offer more specific guidance than official documentation, especially regarding obscure version compatibility issues.
