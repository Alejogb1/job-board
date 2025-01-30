---
title: "Why is Blender using the Intel UHD Graphics 630 instead of the GeForce RTX 2080?"
date: "2025-01-30"
id: "why-is-blender-using-the-intel-uhd-graphics"
---
Blender's selection of the Intel UHD Graphics 630 over the GeForce RTX 2080 indicates a misconfiguration within the application's graphics settings or the system's overall driver management.  In my experience troubleshooting graphics-related issues across various 3D applications, including extensive work with Blender, the most common culprit isn't hardware failure, but rather incorrect software settings that prioritize the integrated graphics card.

**1.  Explanation: Understanding Graphics Card Selection in Windows**

Windows manages multiple graphics adapters via a system known as the Microsoft Graphics Driver.  This driver architecture decides which adapter handles rendering for each application.  Crucially, it doesn't always default to the most powerful card. The operating system considers power consumption, application requirements, and user-defined preferences. If an application isn't explicitly configured to use a dedicated GPU like the RTX 2080, it may fall back to the more energy-efficient integrated GPU, the Intel UHD Graphics 630,  by default. This is particularly true for applications that haven't been explicitly optimized to leverage high-end GPU features or haven't correctly detected the presence of a discrete card. This often manifests as performance issues far below expectations, and is often initially mistaken for a problem with the hardware itself.

Several factors contribute to this:

* **Application Preferences:** Blender, like many applications, allows users to specify which graphics card to utilize.  A misconfiguration here, perhaps inadvertently selecting the integrated graphics, is a prime suspect.

* **Driver Issues:** Outdated or corrupted graphics drivers for either the Intel UHD Graphics 630 or the GeForce RTX 2080 can lead to improper detection and selection by the operating system.  Inconsistent or conflicting driver versions can frequently cause rendering to default to a less powerful option.

* **Power Management Settings:** Windows power plans can actively limit performance to conserve energy.  If a high-performance power plan isn't selected, the system might favor the integrated graphics even if the dedicated card is available.

* **BIOS Settings:** Although less frequent, the system BIOS (Basic Input/Output System) might have settings that influence the priority given to different graphics cards.  Incorrect configuration here could override application-level settings.


**2. Code Examples and Commentary:**

The following examples illustrate how to verify and adjust graphics card selection, assuming a Windows environment.  Note that paths and specific options may vary based on the version of Blender and your system configuration.

**Example 1:  Checking Blender's Internal Graphics Settings**

```python
import bpy

print(bpy.context.preferences.system.compute_device_type) #prints the currently selected compute device type
print(bpy.context.scene.render.engine) #prints the active render engine
```

This Python snippet, executed within Blender's text editor, prints the currently selected compute device (e.g., 'CUDA' for Nvidia GPUs) and the active render engine (Cycles, Eevee).  It helps diagnose if Blender is even aware of and using the RTX 2080 for computation.  The output should indicate the RTX 2080 if configured correctly.  If it shows the Intel integrated card, it's a strong indication of a Blender-specific misconfiguration.


**Example 2: Verifying Nvidia Control Panel Settings**

This example doesn't involve code but rather using the Nvidia Control Panel. The steps are crucial:

1. Open the Nvidia Control Panel.
2. Navigate to "Manage 3D settings."
3. In the "Program Settings" tab, locate Blender.exe in the list of programs.  If it's not present, add it.
4. Under "Select the preferred graphics processor for this program," ensure "High-performance NVIDIA processor" is selected.
5. Apply changes and restart Blender.

This ensures the Nvidia Control Panel explicitly directs Blender to use the RTX 2080.


**Example 3: Examining Windows Graphics Settings (Power Options)**

This isn't a code example but involves adjusting power settings:

1. Open the Windows Control Panel.
2. Navigate to "Power Options."
3. Select "High performance" as your power plan.  Alternatively, create a custom power plan and configure it to prioritize maximum performance for the GPU.

High-performance modes generally allow for better resource allocation for the dedicated graphics card, preventing the system from prioritizing power saving over processing speed.


**3. Resource Recommendations**

I would suggest consulting the official documentation for Blender and your specific Nvidia GeForce RTX 2080 graphics card.  Additionally, thoroughly reviewing the Windows Graphics Settings and the Nvidia Control Panel documentation will prove invaluable.  Finally, examining the BIOS settings of your motherboard might be necessary for a comprehensive diagnosis.  Focus on the sections dedicated to integrated and dedicated graphics handling.  In complex cases, seeking assistance from Nvidia's support channels or relevant online forums for troubleshooting graphics card issues may be necessary for deeper diagnostics and support.  Remember to verify driver versions and check for updates, as outdated drivers are a common source of problems.
