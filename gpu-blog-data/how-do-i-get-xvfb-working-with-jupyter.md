---
title: "How do I get xvfb working with Jupyter Notebook on an M1 Mac?"
date: "2025-01-30"
id: "how-do-i-get-xvfb-working-with-jupyter"
---
On M1-based macOS systems, successfully integrating X Virtual Framebuffer (xvfb) with Jupyter Notebook often presents a challenge due to architectural differences and library compatibility issues, specifically around X11 implementation. This response details my experience getting this configuration operational. I will describe the primary problems, provide solution examples, and suggest learning resources.

The core issue arises from the fact that the M1 architecture's ARM64 processor is fundamentally different from the Intel x86_64 architecture for which many X11 libraries are originally compiled. While XQuartz provides an X server for macOS, it does not seamlessly bridge the gap when dealing with headless server applications needing a virtual display, as xvfb does. The problem manifests primarily as either xvfb failing to initialize correctly or related graphical libraries within Jupyter Notebook’s environment (like matplotlib or OpenCV) not being able to connect to the virtual display. Simply installing xvfb via conventional means (like `brew install xvfb`) will not suffice. A layered approach is necessary.

The most consistent method I've discovered involves explicitly setting the `DISPLAY` environment variable before initiating either the Jupyter Notebook server or running specific code that depends on xvfb. The crucial part is that xvfb must be started before any attempt is made to use it. I typically handle this with a wrapper script or within the Jupyter notebook cell itself to guarantee a pristine environment.  This usually involves first ensuring I have the proper X11 libraries that have been ported to run on arm64; if you just install it via homebrew it typically just adds x86 versions even though it appears to have installed, so check the `arch` of each library carefully (I've found the homebrew installed version of xquartz can be problematic).  It also requires setting up an environment so that Python-based libraries, such as matplotlib, can connect to that server, which usually means using `matplotlib.use("Agg")`. Note that you cannot use `matplotlib.use("TkAgg")` or similar if you are using `xvfb` as no windowing system will be active for it to connect to.

**Code Example 1: Starting xvfb from within a Jupyter Notebook cell**

This snippet demonstrates how to start xvfb from within a Jupyter Notebook cell. This is useful for notebooks that need an isolated xvfb session.

```python
import subprocess
import os
import time

def start_xvfb(display_num=99):
    xvfb_process = None
    display = f":{display_num}"
    try:
        xvfb_process = subprocess.Popen([
            "Xvfb",
            display,
            "-screen", "0", "1280x1024x24",
            "-nolisten", "tcp",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)  # Allow xvfb to initialize
        if xvfb_process.poll() is not None:
            raise Exception(f"xvfb failed to start on {display}: Exit code {xvfb_process.returncode}")
        
        os.environ["DISPLAY"] = display
        
        print(f"xvfb started successfully on display: {display}")
        return xvfb_process

    except Exception as e:
        if xvfb_process is not None:
            xvfb_process.terminate()
        raise e

def stop_xvfb(xvfb_process):
    if xvfb_process:
        xvfb_process.terminate()
        xvfb_process.wait()
        print("xvfb terminated.")


# start xvfb, catch exceptions
xvfb_process = None
try:
    xvfb_process = start_xvfb()

    # Now your graphics code can run, assuming matplotlib.use("Agg") is already set
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.savefig('xvfb_test_plot.png')
    plt.show() # does not do anything because no window
    plt.close()
    print("Figure saved to xvfb_test_plot.png")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    stop_xvfb(xvfb_process) # make sure we stop it
```

*   **Commentary:** The `start_xvfb` function spawns an xvfb process with a specific display number (defaulting to 99).  It also sets the environment variable `DISPLAY` before returning. The `subprocess.Popen` call executes the `Xvfb` executable with specified parameters like the display, screen resolution, and disabling TCP connections.  The `stop_xvfb` function handles process termination.  The `try`/`finally` blocks ensure that the subprocess terminates even if there is an exception. The `time.sleep()` call is essential to give the `Xvfb` server time to actually initialize.
*   **Important Note:** This code assumes that `xvfb` and `Xvfb` are in the system's PATH.

**Code Example 2:  Using xvfb with Matplotlib**

This example shows how to configure `matplotlib` to operate correctly with `xvfb` by setting the backend.  This code assumes `xvfb` has already been started as in the first example.

```python
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure matplotlib doesn't try to use a graphical backend
matplotlib.use('Agg') 

# Assumes DISPLAY is already set, if not set it here
# os.environ["DISPLAY"] = ":99"

# Generate a plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.cos(x)

plt.plot(x,y)
plt.savefig('xvfb_test_cos_plot.png')
print("Plot saved to xvfb_test_cos_plot.png")
plt.close()
```

*   **Commentary:** `matplotlib.use('Agg')` instructs matplotlib to use the "Agg" backend, which renders figures to an image file rather than relying on a GUI window.  This is crucial in a headless environment.  The comment indicates that the `DISPLAY` environment variable should be set prior to execution (such as from Code Example 1). Otherwise, Matplotlib will not know which display to use, even if we use the Agg backend.

**Code Example 3: xvfb within a simple Python script executed in a subshell**

This final example uses a shell script to set the DISPLAY variable before execution.  This is useful for testing a python command when we are working in the terminal.

```shell
#!/bin/bash
# start_xvfb.sh
export DISPLAY=:99

Xvfb $DISPLAY -screen 0 1280x1024x24 -nolisten tcp &
sleep 1 # give it time to start

python3 -c "import os; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np; x=np.linspace(0, 2*np.pi, 100); y=np.tan(x); plt.plot(x,y); plt.savefig('xvfb_test_tan_plot.png'); print('Plot generated');plt.close()"

killall Xvfb #kill the xvfb server
```

*   **Commentary:**  This script first sets the DISPLAY environment variable. Then, it launches the `Xvfb` server in the background.  After a brief pause, a python command that uses the set `DISPLAY` environment variable is executed.  The python command imports `matplotlib` and sets its backend using `matplotlib.use('Agg')`. Finally, it stops the xvfb server.  This method allows running xvfb using the shell without having to start an entire jupyter notebook.

**Recommendations for Further Learning:**

While this addresses the core issue, deep knowledge of these concepts is critical for reliable deployments. I found these resources to be quite helpful:

1.  **Operating System Documentation:** Review the documentation provided by Apple regarding macOS system libraries and environmental variables. This knowledge is fundamental to troubleshooting any issue at the operating system level, particularly on a new architecture like ARM64.

2.  **X11 Protocol Documentation:** While it's a deep dive, understanding the X11 protocol itself provides insight into the fundamental communications that `xvfb` and the applications depend on. There are several detailed texts that describe how `X` servers operate, and how they communicate with clients.

3. **Software Specific API:** Investigate the documentation of the software that depend on the virtual framebuffer. In particular, understand the various backends for matplotlib, the various ways to provide an `X11` connection to OpenCV, or any other library that may make calls to an X server. Knowing the expected parameters for a given library's API can save a lot of debugging time.
