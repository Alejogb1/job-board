---
title: "What's the difference between these two sed commands?"
date: "2025-01-30"
id: "whats-the-difference-between-these-two-sed-commands"
---
`sed 's/foo/bar/g' input.txt`
`sed '/foo/s/foo/bar/g' input.txt`

The core distinction between these two `sed` commands lies in their scope of application within each line of the input. The first command, `sed 's/foo/bar/g' input.txt`, applies the substitution `s/foo/bar/g` to *every* line of `input.txt`, unconditionally. The second command, `sed '/foo/s/foo/bar/g' input.txt`, applies the same substitution, but *only* to lines containing the pattern `foo`. This subtle difference dramatically affects which lines are modified.

My background in managing legacy text processing scripts, particularly those dealing with configuration files and data log transformations, has made me keenly aware of these nuances. I've encountered scenarios where the indiscriminate replacement of the first form caused unintended data corruption, leading me to adopt the more precise targeting of the second form.

Let's break down the syntax. Both commands leverage `sed`'s substitution command, denoted by `s/old/new/flags`. The `old` pattern, `foo`, is what we’re looking for. The `new` string, `bar`, is the replacement. The `g` flag signals a global substitution, which means that *all* occurrences of `foo` on a line will be replaced, rather than just the first.

The difference emerges before the `s` command in the second example. The pattern `/foo/` acts as an address, or selector. This selector tells `sed` that the following substitution command should *only* be applied to those lines that match `/foo/`. Therefore, the first command’s substitutions are universal; all lines, matching or not, will be considered. The second command’s substitutions are conditional, filtered by the address.

To further clarify, consider some practical examples:

**Example 1: Unconditional Replacement**

Imagine `input.txt` contains the following:

```
This is foo text.
Another line with no foo.
This line has foo again.
The foo is also here.
```

Executing:

```bash
sed 's/foo/bar/g' input.txt
```

will produce:

```
This is bar text.
Another line with no bar.
This line has bar again.
The bar is also here.
```

Every line, including the second line lacking the term `foo`, will have its `foo` substring (if any) replaced by `bar`. The second line, though not containing `foo`, is processed. The output shows an unintended replacement since the substitution command is applied to every line.

**Example 2: Conditional Replacement**

Now, with the same `input.txt` content, executing:

```bash
sed '/foo/s/foo/bar/g' input.txt
```

will produce:

```
This is bar text.
Another line with no foo.
This line has bar again.
The bar is also here.
```

Here, the second line, which does not contain the pattern `/foo/`, remains completely unchanged. Only lines containing `foo` were subject to the substitution. This is a critical distinction when working with structured data where you need granular control over modifications.

**Example 3: Contextualized scenario: Modifying Configuration Files**

Consider a configuration file, `config.txt`, with the following content:

```
# This is a default configuration file
port = 8080
server_name = localhost
# debug = false
log_file = /var/log/app.log
# api_url = http://example.com/api
```

If I wanted to change all occurrences of "localhost" to "127.0.0.1," without accidentally impacting any commented-out lines, using:

```bash
sed 's/localhost/127.0.0.1/g' config.txt
```
would incorrectly modify the comments and the configuration values and thus, the content would become:

```
# This is a default configuration file
port = 8080
server_name = 127.0.0.1
# debug = false
log_file = /var/log/app.log
# api_url = http://example.com/api
```

However, using:
```bash
sed '/server_name/s/localhost/127.0.0.1/g' config.txt
```
correctly replaces only the relevant occurrence, preserving the integrity of comments:
```
# This is a default configuration file
port = 8080
server_name = 127.0.0.1
# debug = false
log_file = /var/log/app.log
# api_url = http://example.com/api
```
This demonstrates how the address filtering ensures the targeted replacement and avoids unintended side effects.

This selective substitution makes the second form significantly more powerful and less prone to errors in real-world scenarios. The first form is simple, but its simplicity comes at the cost of precision. Therefore, unless the intent is to globally modify every line, the conditional form using an address is generally the safer and more maintainable approach, particularly when handling configuration files or large text data files.

When considering efficiency, the performance difference between these commands may be negligible for small input files. However, when working with large files or frequently invoked scripts, using the conditional approach may result in minor performance savings. This occurs because `sed` avoids applying the computationally expensive substitution on each line, only on the lines that have been matched with the address.

In practical application, the judicious use of the address parameter has saved me countless hours of debugging scripts. I recommend always considering context, and if the modification needs to be confined to specific lines, use a `sed` command with an address.

**Further Study**

For those interested in delving deeper, I suggest exploring the following resources:

*   **Text Processing with GNU Sed:** A comprehensive guide to understanding `sed`'s capabilities, including its various address formats and regular expression syntax.
*   **Advanced Bash Scripting Guide:** This reference covers numerous text manipulation techniques that integrate with `sed`, highlighting the practical application of commands within shell scripts.
*   **UNIX Shell Programming:** This resource provides a holistic view of command-line tools and how they interact, placing `sed` within the broader context of system administration and data analysis.

These materials will equip you with the skills to apply `sed` effectively and avoid common pitfalls related to the difference between global and conditional substitutions.
