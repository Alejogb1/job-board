---
title: "How can I utilize an AMD GPU with TensorFlow in WSL on Windows 10?"
date: "2025-01-30"
id: "how-can-i-utilize-an-amd-gpu-with"
---
Achieving optimal TensorFlow performance leveraging an AMD GPU within the Windows Subsystem for Linux (WSL) on Windows 10 necessitates a careful configuration, as direct GPU access from WSL is not inherently supported. The challenge stems from the fact that WSL1 relies on a translation layer for system calls, which cannot directly interface with the host’s GPU hardware. While WSL2 offers a more direct approach to hardware interaction, it still requires specific driver implementations to enable AMD GPU acceleration for TensorFlow. Therefore, the solution involves a combination of using WSL2, installing the appropriate drivers within the WSL environment, and configuring TensorFlow to utilize the installed AMD drivers.

The primary obstacle is the indirect path of communication between the TensorFlow operations within WSL and the host’s physical GPU. WSL1, due to its architecture, is essentially incapable of GPU passthrough. WSL2, however, is underpinned by a lightweight virtual machine, allowing for greater potential in hardware interaction. Even with WSL2, the standard Windows graphics drivers are insufficient, as they are not directly exposed to the WSL environment. This necessitates installing specific AMD drivers inside the WSL distribution itself. Moreover, the version of TensorFlow, and the supporting `tensorflow-amd-gpu` package, must be compatible with the installed driver version.

The general procedure comprises three essential stages. First, converting the WSL distribution to WSL2. Second, installing the necessary AMD ROCm drivers within the WSL2 environment. And finally, installing the correct version of TensorFlow with AMD GPU support. This process differs significantly from a native Linux setup.

Here’s a breakdown of the necessary steps, along with illustrative code and detailed explanations:

**1. Ensure WSL2 is active and your distribution supports it.**

The initial step is fundamental. You must ensure your WSL environment is operating in WSL2 mode. You can achieve this with the following PowerShell commands:

```powershell
wsl --list --verbose
```
This command lists all installed WSL distributions, along with their respective versions. Check if the distribution you plan to use is listed as "2" under the "VERSION" column. If it isn't, you will need to convert it:

```powershell
wsl --set-version <Distribution Name> 2
```

Replace `<Distribution Name>` with the actual name of the distribution (e.g., `Ubuntu-20.04`). This conversion can take some time. Once complete, use the `wsl --list --verbose` command again to confirm the version is set to 2.

**2. Driver Installation inside WSL2.**

The key lies in installing AMD's ROCm drivers, specifically designed for GPU compute, *inside* the WSL2 environment, not on the Windows host. It's crucial to get the versions aligned for successful TensorFlow integration. The installation instructions can vary slightly based on the chosen Linux distribution, but generally involve adding the ROCm repository and installing the relevant packages. Here is a typical installation sequence:

```bash
sudo apt update
sudo apt install wget
wget https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/5.4 main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms rocm-libs
```

*   `sudo apt update`: This command refreshes the package list, essential before installing new packages.
*   `wget https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key -O - | sudo apt-key add -`: This downloads the ROCm repository's public key and adds it to your system’s trusted keys. This step ensures that the packages you install from that repository are genuine and haven't been tampered with.
*   `echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/5.4 main' | sudo tee /etc/apt/sources.list.d/rocm.list`: This line adds the AMD ROCm repository to the list of sources where your system will look for packages, specifying version 5.4. You might need to change the version depending on compatibility.
*  `sudo apt update`: Refreshes the package list again, this time incorporating the newly added repository.
*   `sudo apt install rocm-dkms rocm-libs`: This installs the core ROCm packages. `rocm-dkms` handles driver compilation, while `rocm-libs` provides the necessary libraries for GPU-based computation.

After running these commands, you'll need to reboot the WSL instance for the changes to take effect.  You can do this by closing the WSL terminal or running `wsl --shutdown`.

**3. TensorFlow Installation with AMD GPU Support.**

With the necessary drivers in place, you can proceed to install TensorFlow with AMD GPU support. The `tensorflow-amd-gpu` package provides the necessary linkages. It is essential to pick a version of tensorflow that aligns with your ROCm driver version.  Here is a typical installation sequence:

```bash
python3 -m pip install tensorflow==2.10.0
python3 -m pip install tensorflow-amd-gpu
```
* `python3 -m pip install tensorflow==2.10.0`: Installs a specific version of TensorFlow (2.10.0 in this example). Replace `2.10.0` with the desired TensorFlow version compatible with your installed ROCm drivers.
* `python3 -m pip install tensorflow-amd-gpu`: Installs the AMD-specific GPU version of TensorFlow, ensuring it’s linked against the installed ROCm libraries.

Post-installation, a verification step is crucial to confirm TensorFlow can access the GPU:

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    print("GPU device name: ", gpu_device.name)
    print("GPU device details: ", gpu_device)
else:
    print("No GPU device found.")
```
This script will output the number of detected GPUs and, if found, the GPU's name and details. If a GPU is successfully detected, it signifies that TensorFlow is correctly configured to use the installed ROCm drivers.

Debugging is often needed due to version incompatibilities. If the GPU is not being detected, several troubleshooting steps can be taken. Start by checking that the installed ROCm and TensorFlow versions are compatible. Review the official AMD ROCm documentation for version mappings. Verify the environment variables needed by the TensorFlow GPU library are correctly set. Sometimes, a complete uninstall and reinstall of ROCm and TensorFlow, ensuring all previous versions are removed correctly, is the fastest path to resolution.

For further information and detailed steps, consider researching the official ROCm installation guide provided by AMD. Additionally, TensorFlow's official documentation, particularly concerning hardware acceleration, is essential. Various online forums and communities related to TensorFlow and ROCm often have user discussions on specific troubleshooting encountered during this setup process. Finally, research guides from prominent technology websites and blog posts frequently provide updated step-by-step procedures that reflect the latest software releases and compatibility fixes.
