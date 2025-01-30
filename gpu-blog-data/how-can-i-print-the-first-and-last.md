---
title: "How can I print the first and last lines of a file, with the middle portion omitted, using a pipe?"
date: "2025-01-30"
id: "how-can-i-print-the-first-and-last"
---
Having spent considerable time processing large log files, I frequently encounter the need to extract only the essential header and trailer information while discarding the bulk of intermediate data. Specifically, printing just the first and last lines of a file through a pipe is a common task when examining the structure of such files, without being overwhelmed by their content. The tools available within the Unix-like operating system environment allow for efficient solutions.

The primary challenge lies in managing the sequential nature of file processing through a pipe, where data flows continuously from one command to the next. We cannot directly access the 'last' line until the entire input stream has been exhausted, which presents a problem when intermediate lines should be discarded. However, by strategically combining `head`, `tail`, and the use of command substitution, we can achieve the desired outcome.

The first tool, `head`, is used to extract a specific number of lines from the beginning of the input. By itself, `head -1 file.txt` will print the first line. The second tool, `tail`, is used to extract from the end, with `tail -1 file.txt` retrieving the last. However, direct piping such as `head -1 file.txt | tail -1` will only output the *first* line, which is consumed by `head -1`, then the pipe feeds that single line to `tail -1`, giving the same first line back out. To address this, we need to capture the output of `tail -1` separately.

Command substitution is the key here; it allows us to execute a command and use its output as an argument to another command. I will demonstrate several variations, from a more verbose to a concise implementation, each illustrating a specific aspect.

**Example 1: Explicit Variable and Command Substitution**

```bash
first_line=$(head -1 file.txt)
last_line=$(tail -1 file.txt)
echo "$first_line"
echo "$last_line"
```

This example employs command substitution to store the results of `head -1 file.txt` and `tail -1 file.txt` into the variables `first_line` and `last_line`, respectively. The `echo` commands then print these stored values on separate lines. This approach is clear and easy to understand, making it suitable for scripting where readability is paramount. Although not directly piping the output from one to the other, it does show the principle of two sub commands independently getting their lines. This approach would be suitable if, for example, those extracted lines were then going to be used in additional steps, which would not be possible with pipes directly.

**Example 2: Direct Piping with Separate Subshells**

```bash
head -1 file.txt; tail -1 file.txt
```

This method avoids the explicit variable storage as shown in example 1. By using the semi-colon `;` to chain the commands sequentially in the same shell, each command runs, outputting to standard out. The output of the first `head -1 file.txt` will print the first line, and then `tail -1 file.txt` will execute after, printing the last line. While seemingly concise, this does not use a pipe. Rather, the semi-colon represents a command separator. The following command will take the output from the first command and pipe it to the second, if they are executed within a single subshell.

**Example 3: Pipe with Command Grouping and Subshells**

```bash
(head -1 file.txt; tail -1 file.txt)
```

This is the most concise solution and the most applicable to the original problem description. The use of parentheses `()` creates a subshell which groups the sequential commands, allowing us to print the output of both commands sequentially. The first line will be printed first, due to the execution order of the semi-colon command separator, followed by the last line. The parenthesis creates a subshell, where the commands are executed, and the output of the subshell is then output to standard out. While the parentheses don't technically create a pipe, it shows how the output of multiple commands can be grouped together and sequentially printed out.

**Example 4: Using `printf` with command substitution**

```bash
printf '%s\n%s\n' "$(head -1 file.txt)" "$(tail -1 file.txt)"
```

Here, command substitution is combined with `printf` to explicitly format the output. The output from both `head -1 file.txt` and `tail -1 file.txt` are captured and formatted as arguments to `printf`, which then prints each line to standard output using `\n` as a line separator. This method is very similar to the first example, but instead of using intermediate variables, the output is passed directly to printf for formatting. It also only involves a single `printf` command rather than the two `echo` commands used in the first example.

**Error Handling**

A robust implementation would include checks for file existence and handle cases where the file is empty or contains only one line. For simplicity, these examples do not implement error handling. For a more comprehensive solution, I would consider using conditional statements and the `-s` flag to check file size before attempting to process it. I should also make a check that the file exists and that it's readable.

**Resource Recommendations:**

To delve deeper into this topic and enhance your command-line proficiency, I recommend consulting the documentation for the following tools and concepts:

*   **`head` command:** Provides options for selecting lines from the beginning of a file. Understanding different use cases of this command is essential.
*   **`tail` command:** Focuses on extracting content from the end of a file, and can be extended to skip lines from the beginning.
*   **Command Substitution:** Explore how to embed command output as arguments. This will help in understanding complex shell scripting and working with pipes.
*   **Subshells:** Discover how to group commands and manipulate their execution environment with parentheses.
*   **`printf` command:**  Learn to precisely format output, especially for scripts with specific output requirements.
*   **Bash scripting tutorials:** Many tutorials and online resources can enhance your understanding of shell scripting. I recommend starting with basic variable assignment, conditional operators, and control flow statements to build a good foundation.

In summary, extracting the first and last lines from a file, while discarding the middle part, can be accomplished through a combination of `head`, `tail`, and command substitution. The exact approach will depend on the desired level of verbosity and the overall script requirements. The examples demonstrated highlight common methods and, in conjunction with additional study, should empower you to handle similar tasks in a variety of operational contexts.
