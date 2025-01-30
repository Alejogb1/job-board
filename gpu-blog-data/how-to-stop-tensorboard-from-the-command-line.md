---
title: "How to stop TensorBoard from the command line in Windows?"
date: "2025-01-30"
id: "how-to-stop-tensorboard-from-the-command-line"
---
TensorBoard, by default, does not offer a straightforward command-line kill switch in Windows environments, a situation I’ve frequently encountered while managing long-running model training sessions. The process often persists even after closing the browser tab, leading to resource contention if not addressed promptly.  The core challenge lies in identifying the process ID (PID) associated with the specific Python instance hosting the TensorBoard server and then forcibly terminating it.

The standard `tensorboard --logdir=<path>` command initiates a Python process that spawns the actual web server.  Closing the command prompt window where you launched TensorBoard does not reliably terminate this background process. Instead, it detaches from the command prompt, continuing to consume resources.  To terminate it effectively, we must use Windows-specific command-line tools designed for process management. The primary method involves two steps: identifying the port TensorBoard is running on and then using that port to isolate and terminate the associated process.

My approach, honed over several projects involving complex deep learning model training, typically uses `netstat` to ascertain the listening port followed by `taskkill` to force the termination. `netstat` displays active network connections, listening ports, and associated processes. We then filter this output to pinpoint the port TensorBoard is using.  Once identified, the port is used to locate the PID using `tasklist`.  Finally, `taskkill`, equipped with the identified PID, terminates the TensorBoard process. This is a more reliable approach than relying on simplistic attempts to kill all Python processes.

Here's a breakdown with code examples:

**Example 1: Identifying the Listening Port**

I typically start by running a TensorBoard instance on a specified log directory:

```batch
tensorboard --logdir=.\logs
```
This initiates TensorBoard, and the output on the command line shows the assigned port. But, if we launched it earlier or the output is not available, we must determine it. The following command extracts the listening port:

```batch
netstat -ano | findstr LISTENING | findstr 6006
```

**Commentary:**
*   `netstat -ano`:  This is the fundamental command for displaying network connections, listening ports, and associated PIDs.
    *   `-a`: Shows all connections and listening ports.
    *   `-n`: Displays addresses and port numbers in numerical format.
    *   `-o`: Displays the associated PID.
*   `| findstr LISTENING`:  Pipes the output of `netstat` to `findstr`, filtering for lines that include the keyword `LISTENING`, showing only listening ports.
*   `| findstr 6006`: Again pipes the output to `findstr`, filtering for lines that include `6006`, the default port for TensorBoard.

  If your TensorBoard instance uses a non-default port (e.g., if you used the `--port <port_number>` argument), replace `6006` with your specific port. If the instance uses a dynamically assigned port, you would initially grep `python` process by `netstat -ano | findstr python` then look for listening port by hand in the console.

The output of this command will resemble:

```
  TCP    0.0.0.0:6006         0.0.0.0:0              LISTENING       12345
```

In this case, `12345` is the PID we’ll need. This method is reliable and less susceptible to errors than generic killing all python processes.

**Example 2: Identifying the PID**

While the previous example displays the PID within the `netstat` output, there are cases where it may be more reliable to first identify the process by port and then extract the PID. You can get the PID via command prompt:

```batch
tasklist /FI "SERVICES eq None" /FI "IMAGENAME eq python.exe" /v
```

**Commentary:**
*   `tasklist`: Displays all running processes.
    * `/FI "SERVICES eq None"`: Filters for process not running under a service.
    * `/FI "IMAGENAME eq python.exe"`: Filters for processes of a given name "python.exe"
    * `/v` Prints verbose details about the processes, including the command line arguments.

This prints out all python processes in the system including the full path to its command line arguments, its PID, name of the user and other details. We can then manually look for the command line arguments containing the port number and associated process.
While manual lookup is less automatic and more error prone, if it’s the only choice, it’s still the only reliable choice.

In the `tasklist` output, find the python.exe process that corresponds to your TensorBoard instance and copy its PID for the next step.

**Example 3: Terminating the TensorBoard Process**

Having identified the PID, we can now terminate the process using `taskkill`. Continuing the example from Example 1, where we determined the PID as `12345`, the command would be:

```batch
taskkill /F /PID 12345
```

**Commentary:**

*   `taskkill`: This command is used to terminate processes by their PID or image name.
    *   `/F`: This forces the termination of the process. The process will be terminated even if it’s busy or unresponsive. Use with caution.
    *   `/PID 12345`: Specifies that the process to be terminated is the one with PID `12345`. Replace `12345` with the actual PID obtained from Example 1 or Example 2.

Executing this command will terminate the TensorBoard process. You will no longer be able to access TensorBoard from your browser, and the associated Python process will be removed from the list of running tasks.

The above three commands can be combined into a simple batch file to make the process easier and automate-able:
```batch
@echo off
setlocal

for /f "tokens=5" %%a in ('netstat -ano ^| findstr LISTENING ^| findstr 6006') do set PID=%%a

if not defined PID (
  echo No TensorBoard instance found on port 6006.
  goto :end
)

taskkill /F /PID %PID%
echo TensorBoard process with PID %PID% terminated.

:end
endlocal
```
Save this as a .bat file, and you have automated the steps described above, making it easy to kill tensorboard processes reliably.

**Resource Recommendations**

For further information on Windows command-line utilities and their usage in process management, I recommend the following:

1.  **Windows Command Prompt documentation**: Microsoft's official documentation offers detailed explanations of various commands like `netstat`, `tasklist`, and `taskkill`, including their options and syntax. It's the definitive source for understanding these commands.
2.  **Online Tutorials**: Several online platforms provide tutorials on batch scripting and Windows command-line usage, providing practical demonstrations and examples. Look for resources specifically covering process management using the tools mentioned in this discussion.
3.  **PowerShell Documentation**: Though I primarily used the command prompt for this response, PowerShell, the advanced shell for Windows, provides more powerful tools for process management and automation. Familiarizing yourself with its equivalents to the commands described here can greatly enhance scripting capabilities in more complex workflows.

In summary, terminating TensorBoard from the Windows command line requires identifying the active port via `netstat`, then the associated process using `tasklist` and finally terminating it via `taskkill`, using the proper arguments. While not a single command, this multi-step approach provides a reliable solution to a frequent challenge when working with deep learning workflows.
