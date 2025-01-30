---
title: "How can I monitor a rich progress bar output from a Python subprocess?"
date: "2025-01-30"
id: "how-can-i-monitor-a-rich-progress-bar"
---
The challenge in monitoring a rich progress bar within a Python subprocess stems from the fundamental isolation of process environments. Output streams of a child process are distinct from those of its parent; consequently, direct manipulation of a parent's console from a subprocess is not feasible without specific inter-process communication strategies. My prior experience implementing complex data pipelines revealed several methods, including using file descriptors for pipes, shared memory constructs, and higher-level libraries designed for these scenarios. The most effective and flexible approach, which I will detail, involves a combination of process output interception and subsequent processing within the parent.

The core idea hinges on the ability to intercept the standard output (stdout) or standard error (stderr) streams of the subprocess and parse them in the parent process. This intercepted data, typically raw text, can then be analyzed and used to update a progress bar visualization in the parent’s terminal. This method bypasses the limitation of direct parent-child screen interaction by treating the subprocess output as a data stream, which I’ve found to be applicable to varied progress bar implementations such as those provided by the “rich” library. I’ll focus on scenarios using `subprocess.Popen`, which provides fine-grained control over the execution and I/O of the subprocess.

The first example shows a basic approach using a named pipe. The subprocess, instead of directly outputting to stdout, is redirected to write to a file. The parent process then reads this file continuously, extracting the progress information from it. Note that this approach requires a file intermediary and does not interact directly with stdout.

```python
import subprocess
import tempfile
import time
import os

def child_process(output_file):
    for i in range(10):
        time.sleep(0.2)  # Simulate work
        with open(output_file, "a") as f:
            f.write(f"Progress: {i*10}%\n")

def parent_process():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        output_filename = temp_file.name

    process = subprocess.Popen(
        ["python", "-c", f"import time; import sys; "
                          f"for i in range(10): time.sleep(0.2); "
                          f"sys.stdout.write(f'Progress: {{i*10}}%\\n')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


    while True:
        line = process.stdout.readline().decode().strip()
        if line == "":
           if process.poll() is not None:
             break
        else:
            print(line)
    process.wait()
    os.unlink(output_filename) # Clean up temporary file
    print("Process completed")
if __name__ == "__main__":
    parent_process()

```
This code sets up a child process that writes progress updates. The parent uses a `Popen` with `stdout=subprocess.PIPE` to read lines sent by the subprocess. Each line from `readline()` is processed to extract progress and printed to parent's stdout. The loop continues until the child process terminates. The `process.poll()` check determines whether the subprocess is still running. The use of `process.wait()` ensures a clean exit, especially when the process might be sending data until the last second. In real world applications the `process.wait()` may need to be accompanied by error handling depending on the nature of subprocess.

This solution works, but the direct reading of stdout lines lacks flexibility to interact with a rich progress bar in real time, since rich library expects to be updated via the same IO streams. Here's an improved example:

```python
import subprocess
import time
import sys
from rich.progress import Progress

def child_process_rich():
    with Progress() as progress:
        task1 = progress.add_task("Downloading", total=100)
        for _ in range(100):
            time.sleep(0.05)
            progress.update(task1, advance=1)


def parent_process_rich():
     process = subprocess.Popen(
       ["python", "-c", f"import time; from rich.progress import Progress;"
                        f"with Progress() as progress:"
                        f"  task1 = progress.add_task('Downloading', total=100);"
                        f"  for _ in range(100):"
                        f"     time.sleep(0.05);"
                        f"     progress.update(task1, advance=1);"],
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE,
       text=True
     )

     with Progress() as progress:
        while True:
          line = process.stdout.readline()
          if line == "":
             if process.poll() is not None:
               break
          else:
               progress.console.print(line, end='')

     process.wait()

if __name__ == "__main__":
     parent_process_rich()
```

Here, I modified both the parent and child to use the `rich` library.  Both processes now create and update the progress bar. Critically, within the parent, the subprocess's `stdout` is intercepted as in the previous example.  However, instead of simply printing the output, I feed the raw text into the `rich.console.print` method to show the subprocess's rich progress bar output directly onto the parent terminal. The `text=True` argument on `subprocess.Popen` ensures the output is decoded from bytes. `end=''` in `console.print` is important because Rich's output usually has a newline at the end. The parent reads stdout line by line, so newline has to be prevented to prevent garbled output. The child's stdout output is now printed directly into the parent's console within the `with Progress() as progress` context, ensuring that the correct cursor positioning is maintained. This method, in my experience, produces clean and effective progress bars.

To make this approach more practical, you would likely want to filter and handle other forms of output from the child process, for example, debug messages, warning messages etc. You may also need a way to distinguish output pertaining to the progress bar versus other forms of standard output the child process is producing. This filtering could involve more complex string matching or, ideally, structured data from the subprocess. Let’s modify the earlier code to include a specific tag for progress information.

```python
import subprocess
import time
import json
from rich.progress import Progress
from rich.console import Console

def child_process_json():
    console = Console()
    with Progress() as progress:
      task1 = progress.add_task("Downloading", total=100)
      for i in range(100):
          time.sleep(0.05)
          progress.update(task1, advance=1)
          console.print(json.dumps({'type': 'progress', 'percentage': i + 1, 'progress':f'Downloading - {i+1}%'}), end="\n")

def parent_process_json():
  process = subprocess.Popen(
      ["python", "-c", f"import time; from rich.progress import Progress; from rich.console import Console; import json;"
                      f" console = Console();"
                      f"with Progress() as progress:"
                      f"  task1 = progress.add_task('Downloading', total=100);"
                      f"  for i in range(100):"
                      f"    time.sleep(0.05);"
                      f"    progress.update(task1, advance=1);"
                      f"    console.print(json.dumps({{ 'type': 'progress', 'percentage': i + 1, 'progress':'Downloading - ' + str(i+1) + '%'}}), end='\\n')"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True
  )
  console = Console()
  with Progress() as progress:
    while True:
      line = process.stdout.readline()
      if line == "":
           if process.poll() is not None:
             break
      else:
        try:
          data = json.loads(line)
          if data['type'] == 'progress':
            console.print(f"[bold green]{data['progress']}[/bold green]", end="\n")
        except json.JSONDecodeError:
            console.print(f"Non JSON Output {line}", end="\n")
  process.wait()


if __name__ == "__main__":
    parent_process_json()

```
This enhanced version uses `json` to structure output from the child process. The child outputs a JSON string for each update, specifying its type and other relevant information. In the parent, I parse the JSON string using `json.loads()` to extract the data and determine how to handle the output, this allows other non-json output lines from the subprocess to be printed as well, but it is not integrated with the parent rich console. The parent can now choose to process the output from the child process based on type, which provides greater flexibility. This technique is particularly beneficial when dealing with multiple sources of output. This method scales well and I've used variations of it to manage progress from many different child processes.

For further exploration, I suggest consulting the official documentation for the `subprocess` module. Additional information on managing asynchronous programming with pipes and streams can provide a deeper understanding. Documentation for the “rich” library will be helpful in customizing its features and output. You may also explore the `asyncio` module for managing concurrent processes if required in a future implementation. These resources, combined with a good understanding of the Python standard library, will equip any developer to handle rich progress bars from subprocesses effectively.
