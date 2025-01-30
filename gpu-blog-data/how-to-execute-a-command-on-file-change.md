---
title: "How to execute a command on file change in bash, skipping iterations until the command completes?"
date: "2025-01-30"
id: "how-to-execute-a-command-on-file-change"
---
The core challenge in executing a command upon file modification in bash while preventing overlapping executions lies in reliably detecting file changes and managing process synchronization.  I've encountered this frequently in my work automating build processes and data ingestion pipelines, where immediate reaction to file changes is crucial, but concurrent command executions lead to instability.  Simple solutions often fall short because they fail to account for the asynchronous nature of file system events and the potential for rapid, successive modifications.

My approach centers on leveraging `inotifywait` for precise file change detection and process management tools for controlling command execution.  `inotifywait` provides a robust mechanism to monitor specific files or directories for modifications, significantly outperforming polling-based methods in efficiency and responsiveness.  Integrating this with a mechanism to guarantee sequential command execution eliminates the race conditions that plague simpler approaches.

**1. Clear Explanation:**

The solution involves a three-stage pipeline:

* **File Change Detection:** `inotifywait` monitors the target file or directory for specified events (e.g., `CLOSE_WRITE`, indicating file modification is complete).  This eliminates the need for polling, improving efficiency and accuracy.

* **Command Execution Control:**  A mechanism is needed to ensure that only one instance of the command is running at a time.  This is critical to prevent overlapping operations and potential data corruption.  I usually employ a lock file or a process ID check, depending on the complexity of the environment.

* **Looping and Event Handling:** The entire process is encapsulated within a loop that continuously monitors for file changes. Upon detection, the command executes; the loop then waits for the command's completion before processing the next change event. This sequential execution is key to avoiding overlapping commands.

This structured approach addresses the inherent concurrency issues in reactive file system processing.  The use of `inotifywait` minimizes resource consumption compared to polling mechanisms, and the controlled command execution prevents instability arising from parallel operations.

**2. Code Examples with Commentary:**

**Example 1: Basic Implementation with Lock File:**

```bash
#!/bin/bash

# Target file
target_file="/path/to/my/file.txt"

# Lock file
lock_file="/tmp/my_command.lock"

while true; do
  inotifywait -e close_write "$target_file" | while read path event; do
    # Acquire lock
    flock -n "$lock_file" || continue # Skip if lock is held

    # Execute command
    echo "Processing $target_file..."
    my_command "$target_file" &> /dev/null # Redirect output for cleaner execution

    # Release lock
    flock -u "$lock_file"

  done
done
```

This script uses a lock file (`my_command.lock`) managed by `flock`.  If the lock file is already held (another instance is running the command), `flock -n` fails, and the loop continues to the next `inotifywait` event.  Successful locking ensures exclusive access for command execution.  Error handling is simplified by redirecting output to `/dev/null`.

**Example 2:  Implementation using PID Check:**

```bash
#!/bin/bash

target_file="/path/to/my/file.txt"
command_pid_file="/tmp/my_command.pid"

while true; do
  inotifywait -e close_write "$target_file" | while read path event; do
    # Check if command is already running
    if [ -f "$command_pid_file" ]; then
      if kill -0 $(cat "$command_pid_file") 2> /dev/null; then
        continue # Skip if command is running
      fi
    fi

    # Execute command and save PID
    my_command "$target_file" &
    command_pid=$!
    echo "$command_pid" > "$command_pid_file"

    wait "$command_pid" # Wait for command to complete

    # Remove PID file
    rm "$command_pid_file"
  done
done
```

This alternative uses a PID file to track the running command.  Before launching `my_command`, the script checks if the PID file exists and if the corresponding process is still alive using `kill -0`.  The PID is written to the file, allowing `wait` to effectively synchronize execution.  Upon completion, the PID file is removed.

**Example 3: Enhanced Robustness with Error Handling:**

```bash
#!/bin/bash

target_file="/path/to/my/file.txt"
lock_file="/tmp/my_command.lock"
log_file="/tmp/my_command.log"

while true; do
  inotifywait -e close_write "$target_file" | while read path event; do
    flock -n "$lock_file" || continue

    echo "$(date +"%Y-%m-%d %H:%M:%S") - Processing $target_file..." >> "$log_file"

    my_command "$target_file" >> "$log_file" 2>&1 # Log both stdout and stderr

    if [ $? -ne 0 ]; then
      echo "$(date +"%Y-%m-%d %H:%M:%S") - ERROR: my_command failed." >> "$log_file"
    fi

    flock -u "$lock_file"
  done
done
```

This improved example incorporates error handling and logging.  Output and error messages from `my_command` are redirected to a log file, facilitating debugging.  The exit status of `my_command` is checked; any non-zero status indicates an error, which is logged accordingly. This provides greater resilience and easier troubleshooting.


**3. Resource Recommendations:**

The `inotify-tools` package provides `inotifywait` and other related utilities.  Understanding basic bash scripting, process management, and file locking mechanisms are essential.  Consult the manual pages for `inotifywait`, `flock`, and `wait` for detailed usage information and options.  A good understanding of shell scripting best practices, including error handling and logging, is crucial for robust solutions.  Consider exploring process supervision tools if dealing with more complex command execution scenarios or background processes.
