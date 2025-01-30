---
title: "Why is the `find` command returning multiple rows in AIX?"
date: "2025-01-30"
id: "why-is-the-find-command-returning-multiple-rows"
---
The seemingly straightforward `find` command can indeed return multiple rows unexpectedly in AIX, often stemming from its interaction with symbolic links and the specific way AIX handles file system traversals. It’s not a bug but rather a feature, or more accurately, a behavior that requires careful understanding. In my experience managing AIX systems for over a decade, these unexpected multiple outputs invariably trace back to either encountering circular symbolic links or dealing with hard-linked files, especially when combined with the `-print` action of `find`.

The key to understanding this lies in how `find` evaluates directory entries and handles symbolic links. By default, `find` in AIX, as in most Unix-like systems, traverses the file system recursively starting from the given path(s). When `find` encounters a directory, it descends into that directory and repeats the process for each entry found within. If, during this recursive traversal, `find` encounters a symbolic link, it typically follows that link – unless explicitly told not to. This link resolution is where problems often occur.

If the symbolic link points to another directory which has previously been traversed, or even worse points back to its parent directory or a directory which contains it, you create a circular traversal. Imagine a simple situation: you have directory A, a subdirectory B, and a symbolic link C in A that points to B. If you start the find in A, it will first traverse A, then B, and then when it encounters C, it will go again into B. If there is no condition that prevent to repeat this process, `find` will traverse B several times. The `-follow` action (which is the default) follows symbolic links, while `-noleaf` tells it not to follow symbolic links that are not in the directory hierarchy provided.

Let's examine the common scenarios, using illustrative code examples and explanations.

**Example 1: Circular Symbolic Link**

Consider a directory structure as follows:

```
/tmp/test_dir/
├── dir1/
│   └── file1.txt
└── symlink_to_dir1 -> dir1
```

Here, we have `dir1` and a symbolic link `symlink_to_dir1` inside `/tmp/test_dir` that points to `dir1`. Now, running the following command:

```bash
find /tmp/test_dir -print
```

Will produce output similar to this, although the exact order and depth of traversing may vary depending on system configuration:

```
/tmp/test_dir
/tmp/test_dir/dir1
/tmp/test_dir/dir1/file1.txt
/tmp/test_dir/symlink_to_dir1
/tmp/test_dir/dir1
/tmp/test_dir/dir1/file1.txt
... and so on
```

Note that the content of `dir1` is listed multiple times. This happens because `find` first encounters `dir1`, then encounters `symlink_to_dir1`, and the default behavior causes it to follow the symlink, returning to `dir1`. Since the default action is to print all encountered paths, it results in the duplicate listings. This demonstrates a simple circular path where the `-print` action gets invoked multiple times for the same file paths.

The critical point here is that `find`, unless instructed otherwise, will follow symbolic links, regardless of whether those links create loops within the search path. This behavior, while seemingly problematic, is consistent with its intended design.

**Example 2: Avoiding Circular Traversal with `-noleaf`**

To prevent this repetitive traversal and the resulting multiple outputs, you can use the `-noleaf` option. The `-noleaf` action of the `find` command on AIX is a little misleading. While it prevents some loops, it also can skip over files unexpectedly.

Let’s run the previous example again, this time using the `-noleaf` option and removing the explicit `-print` as that is now the default:

```bash
find /tmp/test_dir -noleaf
```

The output will now look like this:

```
/tmp/test_dir
/tmp/test_dir/dir1
/tmp/test_dir/dir1/file1.txt
/tmp/test_dir/symlink_to_dir1
```

The `-noleaf` option instructs `find` not to follow the `symlink_to_dir1` because it’s not “in the directory being traversed”. It is technically not part of the directory structure. It only reports the symbolic link itself. However, there is a important caveat here, if there is a file in the same directory as `symlink_to_dir1` like below:

```
/tmp/test_dir/
├── dir1/
│   └── file1.txt
├── file2.txt
└── symlink_to_dir1 -> dir1
```

```bash
find /tmp/test_dir -noleaf
```

The result will be:

```
/tmp/test_dir
/tmp/test_dir/dir1
/tmp/test_dir/dir1/file1.txt
/tmp/test_dir/symlink_to_dir1
```

Notice that `file2.txt` is missing! It was not printed because `find` stop to descend when a non-directory, non-symlink file is found. The `-noleaf` option doesn't entirely prevent repeated visits (it only avoid it on the original path). It changes the behavior, skipping some path that would normally be printed, so it's not a general solution.

**Example 3: Utilizing `-type` and `-prune`**

A more targeted and usually reliable way to prevent multiple outputs is to use `-type l` in combination with `-prune`. This method allows explicit control over how symbolic links are handled. Consider the original scenario:

```bash
find /tmp/test_dir -type l -prune -o -print
```

The output will be:

```
/tmp/test_dir
/tmp/test_dir/dir1
/tmp/test_dir/dir1/file1.txt
/tmp/test_dir/symlink_to_dir1
```

In this command, `-type l -prune` instructs `find` that, when a symbolic link (type `l`) is encountered, it should *not* descend into that entry (using `-prune`). If this condition is not met, the logic falls back to the default `-print` ( `-o` stands for *or*). This produces the desired output, traversing the directory structure only once. The `-prune` option prevents searching below the symlink path, which effectively breaks the loop.

The `-type l -prune` solution works even if `-noleaf` does not, for example in the following scenario:

```
/tmp/test_dir2/
├── dir1/
│   └── file1.txt
└── symlink_dir1/
    └── file2.txt

    /tmp/test_dir2/symlink_dir1 -> /tmp/test_dir2/dir1
```

```bash
find /tmp/test_dir2 -noleaf
```

The output is:

```
/tmp/test_dir2
/tmp/test_dir2/dir1
/tmp/test_dir2/dir1/file1.txt
/tmp/test_dir2/symlink_dir1
```

As you see, it does not display the `file2.txt` because it skips to the other branch, but

```bash
find /tmp/test_dir2 -type l -prune -o -print
```

The output is:

```
/tmp/test_dir2
/tmp/test_dir2/dir1
/tmp/test_dir2/dir1/file1.txt
/tmp/test_dir2/symlink_dir1
/tmp/test_dir2/symlink_dir1/file2.txt
```

And now it displays all the files. `-prune` only stops the recursive loop when it's a symbolic link, and it does not stops traversing if it's a directory entry.

In summary, these examples show that the multiple rows produced by `find` are not a bug but the result of how symbolic links are treated by default. The `-noleaf` option can be helpful in some cases, but it skips files on other paths, making it dangerous in a more complex situation. The most reliable approach, especially for managing circular paths, is to explicitly use `-type l -prune -o -print`.

**Resource Recommendations**

To further enhance your understanding of the `find` command and AIX file system concepts, I suggest exploring the following resources. Refer to the official AIX documentation for precise specifications of the operating system utilities. You can also consult textbooks on Unix/Linux system administration, which provide a more general perspective on file system traversals and command-line utilities. Additionally, practice using the various options of the `find` command in a controlled testing environment to become comfortable with their behavior under various circumstances. Consider creating some shell scripts to test your knowledge of `find`. Finally, review the detailed man page for the `find` utility on AIX, which usually provides important context for understanding its operation and behaviors specific to the operating system.
