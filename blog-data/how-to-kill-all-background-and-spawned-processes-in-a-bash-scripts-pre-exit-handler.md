---
title: "How to kill all background and spawned processes in a bash script's pre-exit handler?"
date: "2024-12-23"
id: "how-to-kill-all-background-and-spawned-processes-in-a-bash-scripts-pre-exit-handler"
---

Alright, let's tackle this. I've been through this particular rabbit hole more times than I care to count, especially when dealing with complex data processing pipelines and test harnesses that spawn a slew of processes. The challenge of ensuring a clean exit, especially when subprocesses are involved, is a common hurdle. Just relying on a simple `trap` statement for handling exits often falls short, mainly because bash doesn't automatically propagate signals down to child processes. So, let me break down the approaches that I’ve found consistently reliable over the years, along with concrete examples.

The core issue is that background processes and processes started within subshells are, by default, detached from the parent bash process. When the parent exits or receives a signal like `SIGINT` (Ctrl+C), these children keep running, often becoming zombie processes, which, while not actively consuming CPU, can clutter the system and be a nuisance. What's more problematic is that these orphaned processes can continue writing to log files or modifying files, leading to unexpected results later on.

The fundamental technique I use is a combination of process group management and signal propagation. The trick is to put all of the potentially problematic spawned processes into the same process group as the main script, allowing the script to signal this group instead of sending signals only to its immediate children. This makes it possible for one single command to effectively terminate all related processes. Let’s unpack how to accomplish that.

First, let’s use `set -m`. This enables job control, which is essential for working with process groups. Now, instead of just launching background processes with a single `&`, I'll use parentheses to group a set of commands and run them in the background. This automatically puts all of those commands within the same process group as the subshell. When we run this, the backgrounded shell will be part of a process group with the main script which allows it to be targeted easily. I will now use the `trap` command to configure our exit handler. Inside the handler, I'll iterate over all backgrounded job ids and kill them with `kill -TERM` to give them a chance to close gracefully. If they don't respond, we use `kill -KILL` to force their termination. We also need to use `wait` to make sure that each backgrounded process is finished.

Here's an example:

```bash
#!/bin/bash
set -m

cleanup() {
  echo "Performing cleanup..."
  jobs -p | while read pid ; do
    echo "Killing job with PID: $pid"
    kill -TERM "$pid" 2> /dev/null
    sleep 1 # Give it a second to respond to SIGTERM
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null
      echo "Force killed job with PID: $pid"
    fi
    wait "$pid" 2>/dev/null
  done
  echo "Cleanup complete."
}

trap cleanup EXIT

echo "Starting background processes..."
(sleep 5; echo "Process 1 finished") &
(sleep 10; echo "Process 2 finished") &
echo "Main script continuing..."

sleep 15
echo "Exiting main script."
```

In this first snippet, I use `jobs -p` to retrieve the process ids of all backgrounded jobs. These are sent sequentially to the cleanup function to terminate them gracefully. The `kill -0 "$pid"` is a trick to check if the process with the specified pid still exists. If it returns exit code 0 then the process is still alive and if not then it has already finished.

The above works in many cases but is not perfect since it relies on bash’s job tracking. We might find ourselves in a position where the child processes are created outside of bash's job tracking mechanisms, or we are using an approach that does not involve background processes. Another approach utilizes process groups more directly. This approach avoids using job ids and relies on direct process group signalling. In this next version, I’ll also incorporate a variable to record the process group id for later use:

```bash
#!/bin/bash

set -m

# Store process group id
pgid=$$

cleanup() {
  echo "Performing cleanup..."
  echo "Killing process group: -$pgid"
  kill -TERM -$pgid 2> /dev/null
  sleep 2 # Give processes some time to respond
  kill -KILL -$pgid 2> /dev/null
  echo "Cleanup complete."
}

trap cleanup EXIT

echo "Starting processes in their own groups..."

(sleep 5; echo "Process 1 in pg: $pgid"; sleep 5; echo "Process 1 finished"; ) &
(sleep 10; echo "Process 2 in pg: $pgid"; sleep 5; echo "Process 2 finished";) &


echo "Main script continuing..."
sleep 15
echo "Exiting main script."
```

Here, I'm capturing the process id of the script, which, when negated, represents the process group id. Instead of individually killing jobs, I am now using the negative process group id with `kill` which signals the entire process group.

Sometimes, you need more control, particularly when child processes themselves create even more children. This often manifests in programs that launch daemon processes, and you need to manage them recursively. In such complex scenarios, a combination of process groups and dedicated cleanup scripts or executable can be more suitable. Here's an example that simulates a process spawning more processes, which then get terminated using a dedicated kill script:

```bash
#!/bin/bash

set -m
# We have to use a unique file name for the lock file so multiple scripts can run
# concurrently without overwriting each other’s locks.
LOCKFILE="/tmp/my_script_$(basename "$0" .sh)_lockfile.lock"
PROCESS_GROUP_FILE="/tmp/my_script_$(basename "$0" .sh)_pgid.txt"
cleanup() {
  echo "Performing cleanup..."
  if [ -f "$PROCESS_GROUP_FILE" ]; then
    read pgid < "$PROCESS_GROUP_FILE"
    echo "Killing process group: -$pgid"
    kill -TERM -$pgid 2> /dev/null
    sleep 2 # Give processes time to respond
    kill -KILL -$pgid 2> /dev/null
    echo "Cleanup complete."
    rm "$PROCESS_GROUP_FILE"
  else
    echo "No process group to cleanup"
  fi
  rm -f "$LOCKFILE"
}

trap cleanup EXIT

# Create Lock File
if [ -e "$LOCKFILE" ]; then
  echo "Another script is already running. Exiting" >&2
  exit 1
fi

touch "$LOCKFILE"

echo "Starting processes in their own groups..."

# Store process group id
pgid=$$
echo "$pgid" > "$PROCESS_GROUP_FILE"

(sleep 5; echo "Process 1 starting... PID: $$"; (sleep 5; echo "Process 1 child starting... PID: $$"; sleep 5; echo "Process 1 child finishing";); sleep 5; echo "Process 1 finished" ) &
(sleep 10; echo "Process 2 starting... PID: $$"; (sleep 5; echo "Process 2 child starting... PID: $$"; sleep 5; echo "Process 2 child finishing";); sleep 5; echo "Process 2 finished";) &


echo "Main script continuing..."
sleep 15
echo "Exiting main script."
```

In this example, we’re taking a more elaborate approach, especially when subshells have their own children. The main script records the original process group and writes it to a file. The cleanup now reads that file and then signals the entire process group which includes any child processes created within the subshells. This is more robust for nested processes. It also includes a basic lock file to ensure that only one instance is active.

To deepen your knowledge on this topic, I'd highly recommend reading the following sources:

*   **"Advanced Programming in the UNIX Environment" by W. Richard Stevens and Stephen A. Rago:** This book is a cornerstone for understanding the intricacies of process management and signal handling in unix-like systems. The chapters on processes, process groups, and signals are particularly relevant to this issue.
*   **The `bash` man page:** Particularly the sections on job control (`set -m`) and `trap`. The man page provides comprehensive details about `bash` functionality.
*   **POSIX Specification of the `kill` command:** This is helpful when you want to understand precisely how the kill command handles process groups and process IDs.

These are not casual reads, but for anyone working with complex bash scripts that spawn child processes, they're invaluable. In practice, the choice of implementation (directly signalling job ids, process groups, or using dedicated cleanup scripts) depends on the complexity of your specific scenario. I’ve found, through many iterations, that process group signaling combined with error handling and timeout mechanisms provides a good balance of robustness and simplicity.
