---
title: "Why do using rsync include exclude patterns matches whitespace?"
date: "2024-12-15"
id: "why-do-using-rsync-include-exclude-patterns-matches-whitespace"
---

alright, so you're hitting that classic rsync gotcha with include and exclude patterns matching whitespace, yeah? i've been there, stared into the abyss of my terminal, wondering why rsync was suddenly deciding that space was a filename. trust me, it's a common head scratcher. let me break down what's happening and how i've learned to work around it over the years.

the core issue boils down to how rsync interprets its patterns. when you supply an include or exclude pattern, it's not treated as a literal string. instead, rsync uses something called a globbing mechanism (which is a form of pattern matching) and that's where things can get a bit... surprising. whitespace, including spaces, tabs, and newlines, is just another character in this context.

let's say you had a folder structure like this:

```
my_folder/
├── file_a.txt
├── file b.txt
├── subfolder/
│   └── file_c.txt
└── another file.txt
```

and you wanted to exclude the file "file b.txt" but include the other ones. so you use something like this:

```bash
rsync -avz --exclude="file b.txt" my_folder/ destination_folder/
```

you'd *think* this would work, wouldn't you? only, rsync will go ahead and exclude any file that has a "file", a "b" and a ".txt" anywhere in their paths, not only the one with that name. the reason is that a space is seen as just another ordinary character. this is unexpected for many beginners, but it's an artifact of the globbing mechanism at play. so the rsync command just interpreted “file b.txt” as a pattern that can match file"some letter"b"other letters".txt". this also means it would exclude the file "another file.txt" and you get a result that's not what you expected.

i remember back in my early days, i had a similar problem backing up a website. i was trying to exclude a particular logs directory (it had spaces in the directory name), which ended up with tons of files and directories i never wanted excluded gone because i did not understand this issue. it wasn't a fun debugging session trying to restore a website from a backup that was incomplete. i learned my lesson the hard way, let me tell you.

the trick is that rsync's include and exclude patterns, similar to many shell globbing mechanisms, use a specific syntax. they treat whitespace literally unless you either escape it or use quotes. here are some solutions that can help:

**solution 1: escaping whitespace**

if you have a file or folder with a space, one approach is to use a backslash `\` to escape the space. this tells rsync to treat the space character literally, not as a pattern separator. the command would be something like:

```bash
rsync -avz --exclude="file\ b.txt" my_folder/ destination_folder/
```

this `\` escape works in most shells, but sometimes it gets interpreted by the shell first, and then the rsync command does not see the escaped string. it's like the shell saying "hey, i got this" and then passing a string to the command that does not have the escaping character anymore.

**solution 2: using single or double quotes**

a better solution is often to wrap your include and exclude patterns in single or double quotes. both effectively turn off globbing to some extent and help rsync to see the string with literal space characters. it's not that they are totally off, it's that they turn it off for the spaces, and the literal string is matched, instead of a complex pattern.

for single quotes:

```bash
rsync -avz --exclude='file b.txt' my_folder/ destination_folder/
```
for double quotes:

```bash
rsync -avz --exclude="file b.txt" my_folder/ destination_folder/
```

both single and double quotes work similarly, but single quotes are generally preferred in this case to avoid any shell interpolation. if you were to include shell variables in your patterns you would need to use double quotes instead of single quotes.

**solution 3: using more specific patterns**

sometimes a better approach is not just to escape, quote, or deal with whitespace, but to create more specific patterns. for instance, if you want to exclude *only* files named "file b.txt" directly inside `my_folder`, you could use the following pattern:

```bash
rsync -avz --exclude="/file b.txt" my_folder/ destination_folder/
```

the leading forward slash `/` indicates that the match must start at the base of the source path. therefore this avoids inadvertently exclude "another file.txt" because that one does not include "file b.txt" in the path itself. this approach is less prone to accidental matches in subdirectories too.

**more complex scenarios**

the real challenges appear when you start using wildcards alongside whitespace, let's say you have file names like these:

```
my_folder/
├── logs 2023-10-26.txt
├── logs 2023-10-27.txt
├── logs 2023-10-28.txt
└── subfolder/
    └── logs 2023-10-29.txt
```
and you want to exclude logs from October 26, but include the other logs. here you have a mix of both, a pattern and a whitespace, you could use the following, even though other approaches like regex would be better:

```bash
rsync -avz --exclude='logs 2023-10-26.txt' my_folder/ destination_folder/
```

rsync uses a simplified version of regular expressions in globbing, but it's not a full-fledged regex engine. for complex scenarios, you could investigate more powerful tools. the problem is that rsync excludes do not support look ahead or look behind regexes. this often leads to a painful experience. when i'm doing backups and really need complex exclusions, i tend to resort to `find` with `-not -path` and pipe that to rsync using the `--files-from` option. it is a bit more verbose but much more powerful than the exclude/include option. in these cases, rsync works as a dumb copy machine, and not as the more complex logic machine.

**lessons learned and best practices**

over the years, these things have become my default workflow:

1.  **always use quotes:** this might be overkill for many simple cases, but it’s the safest habit.
2.  **test patterns before using them:** i always do a dry run (`rsync -avn ...`) to be sure i get the desired behavior. it saves a lot of headaches when you catch those small details that would create a disaster.
3.  **be specific with paths:** avoid overly broad patterns. this is particularly important when dealing with nested folders.
4.  **consider `find` for advanced exclusions:** when rsync's globbing isn't enough, don't be afraid to use more powerful tools and pipe the output to `rsync --files-from`. you are going to be faster and more efficient.
5.  **document your patterns clearly:** always write a comment describing the patterns next to the command. this is for your future self, which will be grateful you did.

to deepen your understanding of pattern matching and globbing, i highly suggest reading about unix shell pattern matching, and exploring materials like "the linux command line" by william shotts. it has an excellent coverage of these topics. also, check the rsync manual ( `man rsync`). it has examples and detailed explanations. it will become your best friend when dealing with those weird edge cases. it's a long read, but worthwhile. finally, don't rely on the internet too much. most of what's out there is shallow, and the official manuals are usually the best source of truth.

i've had my share of debugging sessions caused by whitespace in filenames. we all have. but with a clear grasp of how rsync handles these patterns and by learning the tools, you can sidestep these issues. and remember: don’t worry if you’re struggling with it. rsync has been known to confuse seasoned developers (it’s a bit like trying to read assembly code on a monday morning). happy syncing!
