---
title: "How do rsync include/exclude patterns handle whitespace?"
date: "2025-01-30"
id: "how-do-rsync-includeexclude-patterns-handle-whitespace"
---
Rsync's handling of whitespace within include and exclude patterns is a frequent source of confusion, stemming from the subtle interplay between shell expansion and rsync's internal pattern matching.  My experience troubleshooting deployment scripts across various Unix-like systems highlighted this repeatedly.  The key takeaway is that whitespace within a pattern is *not* automatically treated as a wildcard or special character by rsync itself; rather, its interpretation is heavily dependent on how the shell processes the pattern before rsync receives it.

**1. Explanation:**

Rsync utilizes a pattern-matching engine based on regular expressions, but these are not directly interpreted from the command line. Instead, the shell (bash, zsh, etc.) first parses the command line, expanding any wildcards and globbing expressions *before* passing the resulting arguments to the rsync process.  This is crucial to understanding whitespace handling.

Consider an exclude pattern like `"file with spaces.txt"`.  The shell, by default, does *not* treat the spaces as wildcards. It treats the entire string as a single, literal filename.  Therefore, rsync will only exclude a file named exactly `"file with spaces.txt"`.  It will *not* exclude files like `"file with spaces (copy).txt"` or `"file_with_spaces.txt"` because the shell has already interpreted the pattern as a literal string.

Conversely, if you intend to exclude files matching a pattern *containing* spaces, you need to quote the pattern carefully and potentially use shell globbing or escape sequences.  Failure to do so will lead to the shell interpreting the spaces as argument separators, resulting in unexpected behavior and likely excluding nothing.  For example,  `--exclude=file with spaces.txt` (unquoted) will generally be interpreted by the shell as three separate arguments, causing rsync to fail to recognize the pattern.

The same principle applies to include patterns.  If you intend to include only files matching a specific pattern with spaces, you must use proper quoting and ensure the entire pattern is treated as a single unit by the shell.

Furthermore, the use of regular expressions directly within rsync's include/exclude options is not supported.  While rsync employs a pattern-matching mechanism, it's not a full-fledged regular expression engine.  Trying to use regular expression syntax directly within the `--include` or `--exclude` options will likely result in literal string matching, with spaces interpreted as literal characters.


**2. Code Examples with Commentary:**

**Example 1:  Correct Exclusion of a File with Spaces**

```bash
rsync -avz --exclude="file with spaces.txt" source/ destination/
```

This example correctly excludes the file `"file with spaces.txt"`.  The crucial aspect here is the use of double quotes around the filename. This prevents the shell from interpreting the spaces as argument separators. The pattern is passed to rsync as a single string.

**Example 2: Incorrect Exclusion Attempt (using Unquoted Pattern)**

```bash
rsync -avz --exclude=file with spaces.txt source/ destination/
```

This will likely fail to exclude `"file with spaces.txt"`.  The shell interprets this as three separate arguments, effectively making the exclude pattern meaningless to rsync.

**Example 3:  Matching Patterns with Wildcards (for Multiple Files)**

```bash
rsync -avz --exclude="*.txt with spaces*" source/ destination/
```

This example demonstrates using shell globbing to match multiple files with spaces in their names.  The `*` wildcard will match any sequence of characters before and after " with spaces".  Double quotes are essential to keep the entire pattern intact. Note that this relies on shell globbing, not rsync's internal pattern matching capabilities.  The shell expands this pattern *before* rsync receives it.


**3. Resource Recommendations:**

The rsync man page is an invaluable resource. Carefully reviewing the sections on `--include` and `--exclude` options, along with paying close attention to the description of argument parsing is crucial.  Consult a comprehensive Unix shell scripting guide for a detailed understanding of shell expansion, quoting, and the behavior of wildcards and globbing.  Finally, examining detailed examples in relevant documentation or tutorials will reinforce the concepts demonstrated in these examples.  Understanding the interaction between the shell and rsync is paramount to correct pattern matching.  Remember that the primary role of the shell is to process command-line arguments before passing them on to the respective program. Improper handling at this stage will often lead to unexpected results.  Testing thoroughly, with a focus on edge cases involving filenames with spaces and other special characters, is essential for confirming correct behavior and identifying any subtle issues. My own debugging experience involved many hours of painstaking review of shell expansions using `echo` and `set -x` to trace the exact arguments processed by rsync.  This meticulous approach is crucial for developing robust and reliable deployment scripts.
