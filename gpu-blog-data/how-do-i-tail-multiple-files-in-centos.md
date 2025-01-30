---
title: "How do I tail multiple files in CentOS?"
date: "2025-01-30"
id: "how-do-i-tail-multiple-files-in-centos"
---
CentOS, like other Linux distributions, provides several robust methods for monitoring multiple files simultaneously, crucial for diagnosing issues across various application components. I've encountered scenarios where pinpointing the root cause of a problem required observing logs from the web server, database, and a custom application in real-time. Relying on individual `tail -f` commands in separate terminal windows quickly becomes unmanageable, prompting a search for more streamlined solutions. The key lies in leveraging either built-in utilities like `tail` with specific options or dedicated tools designed for this task.

Fundamentally, the challenge lies in efficiently aggregating and presenting real-time output from several disparate files.  A simple approach involves `tail`'s ability to accept multiple file paths as arguments.  However, without additional parameters, `tail` treats the files sequentially, outputting the last lines of each before monitoring them collectively.  This provides an initial snapshot, followed by the ongoing additions. For continuous monitoring,  `tail -f` must be supplemented with a means to delineate output sources to maintain context.

Let's look at the most common methods, beginning with basic `tail` functionality:

**Method 1: `tail -f` with File Identification**

The basic `tail -f` command can be adapted to handle multiple files using simple shell techniques. The essential addition is to include the filename with each line being printed. `tail` does not inherently do this, so we utilize shell redirection and a loop.

```bash
for file in /var/log/httpd/access_log /var/log/httpd/error_log /var/log/myapp/app.log; do
  tail -f "$file" | sed "s/^/[$file] /" &
done
wait
```

**Commentary:**

*   The `for` loop iterates through each file path specified. In this fictional example, these are hypothetical web server access logs, web server error logs, and application-specific logs.
*   `tail -f "$file"` monitors each file for new lines and continuously outputs them. The `"$file"` is enclosed in quotes to handle paths with spaces or other special characters.
*   The pipe (`|`) sends the output of `tail -f` to `sed`.
*   `sed "s/^/[$file] /"` inserts the filename surrounded by square brackets and followed by a space, at the beginning of each line using the `s` (substitute) command and caret (`^`) to indicate the beginning of the line. This crucial step allows us to see which file produced the output, particularly useful with multiple sources.
*   The ampersand (`&`) places each `tail` command into the background, allowing concurrent monitoring.
*   `wait` ensures the script doesn't exit until all background `tail` processes have completed. Without this, the script could exit immediately, prematurely ending your tail sessions.

This method proves useful in situations where a simple, quick view of multiple files is needed.  The main limitation is the lack of finer control over output appearance and an increased difficulty if you are handling many files. Also, killing this group of processes requires finding and terminating all the individual `tail` processes.

**Method 2: Using `multitail`**

`multitail` is a dedicated utility designed specifically for handling multiple log files. I've often relied on it for complex investigations involving several microservices, each generating its own log. It provides more structured output and several customization options that basic `tail` and shell loops can’t handle. It typically requires installation via the package manager, such as `yum install multitail`.

```bash
multitail -i -s 2 /var/log/httpd/access_log /var/log/httpd/error_log /var/log/myapp/app.log
```

**Commentary:**

*   `multitail` launches the application.
*   `-i` tells `multitail` to display a split screen view.
*   `-s 2` sets the number of screens to 2. The files are distributed across those screens, avoiding information overload.
*   The file paths represent the various logs that need to be monitored.  `multitail` will display these, and add any new lines at the end of the files.
*   The interactive controls of `multitail` enable scrolling, filtering, and zooming each screen.

`multitail` provides a significant advantage when you need visual organization and some degree of interaction with the output. You can cycle through screens, filter by keywords, or even highlight specific entries; functions you would struggle to achieve efficiently with basic command-line tools. Its main disadvantage is that it is not a default utility, and will need to be installed if you don't already have it.

**Method 3: Combining `tail` and `awk` for more complex filtering**

Sometimes, you need to do more than merely tag each log line. You may need to filter lines based on specific criteria, or extract particular fields. `awk` provides a solution here, allowing you to perform more involved parsing and transformation. I've found this invaluable when needing to track the progress of specific processes or identifying errors based on string patterns.

```bash
for file in /var/log/httpd/access_log /var/log/httpd/error_log /var/log/myapp/app.log; do
   tail -f "$file" | awk -v file="$file" '{print "[" file "] " $0 }'  | grep "error"  &
done
wait
```

**Commentary:**

*   The structure using the `for` loop and backgrounding are similar to method 1.
*   The output of `tail -f` is piped to `awk`
*   `awk -v file="$file" '{print "[" file "] " $0 }'` takes the filename from the variable, and prints it in the same manner as method 1. The `$0` represents the original line that comes from `tail`.
*   The additional pipe to `grep "error"` filters the output from all the logs, and only prints lines that include the word “error”.

While this is a basic example, it illustrates how you can extend this to perform more complex operations.  You can use `awk` to extract specific fields, filter based on numeric values, or even reformat the output.

**Resource Recommendations:**

Several resources offer more in-depth knowledge on these techniques. Books focusing on Linux command-line tools, such as "The Linux Command Line" by William Shotts, offer a comprehensive overview of utilities like `tail`, `sed`, `awk`, and general shell scripting principles. Specific online documentation for `multitail`, typically available from the author's website or project page, provides details on advanced customization options, interactive features, and usage scenarios. Additionally, engaging with online Linux forums or communities, particularly those focused on system administration and operations, can provide practical examples, real-world use cases, and insightful discussions on various monitoring and troubleshooting approaches. Finally, practicing these techniques on a virtual machine provides an environment for safe and extensive experimentation, solidifying your understanding of the commands and concepts. While this is not an exhaustive list of every conceivable option, these approaches provide practical and effective means to accomplish this task.
