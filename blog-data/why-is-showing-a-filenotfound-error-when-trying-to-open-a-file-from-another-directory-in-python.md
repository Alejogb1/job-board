---
title: "Why is Showing a FileNotFound Error when trying to open a file from another directory in python?"
date: "2024-12-15"
id: "why-is-showing-a-filenotfound-error-when-trying-to-open-a-file-from-another-directory-in-python"
---

alright, so you're seeing that pesky `filenotfounderror` when trying to open a file from a different directory in python, yeah? i've definitely banged my head against that wall a few times. it's a common pitfall, and it almost always boils down to python not knowing exactly where to look for your file. let me break it down from my experience.

the core issue here is that python’s file operations, like `open()`, work relative to the current working directory, that’s the directory python thinks its running from. this isn't necessarily the same directory where your script `.py` file is located. so, if your script sits in `project/scripts/my_script.py` and you’re trying to open a file from `project/data/my_file.txt`, and your script uses a path like `'my_file.txt'`, it wont find it. that's because python is probably looking for it in `project/scripts/` not `project/data/`.

i remember this vividly from when i was working on a data pipeline project. we had all these different modules scattered around in different directories. the main script was supposed to load configuration files from a `config/` directory, but it kept throwing `filenotfounderror`. it took me a good hour, staring at the screen and mumbling to myself, before i realized the current working directory was not what i expected.

so, how do we fix this? there's a few ways.

**1. using absolute paths:**

this is the most straightforward, but also the least portable. an absolute path tells python exactly where to find the file, no matter what. you specify the full path from the root of the filesystem. for example, instead of `'data/my_file.txt'`, you might use something like `'/home/user/project/data/my_file.txt'` (on linux/macos) or `'c:\\users\\user\\project\\data\\my_file.txt'` (on windows).

here's some code that exemplifies that:

```python
import os

file_path = "/home/user/project/data/my_file.txt"  #linux
#file_path = "c:\\users\\user\\project\\data\\my_file.txt" # windows

try:
    with open(file_path, 'r') as file:
       content = file.read()
       print(content)
except FileNotFoundError:
    print("file not found at", file_path)

```

obviously, replace `/home/user/project/data/my_file.txt` with your actual file path. the `try-except` block gracefully handles the `filenotfounderror`.

the downside of absolute paths is that they are brittle. if you move your project to a different machine or change the directory structure, they'll break. so it’s mostly fine just if this is only for yourself and will not move from its directory but its bad practice, specially if you are sharing code, which you probably will be at some point.

**2. relative paths using `os.path`:**

this is my go-to approach most of the time. instead of hardcoding absolute paths, we use paths that are relative to the location of your script. the `os.path` module offers various tools to handle this.

here's an example:

```python
import os

# gets directory of current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# construct relative path to data directory
data_dir = os.path.join(script_dir, "..", "data")
file_path = os.path.join(data_dir, "my_file.txt")

try:
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
        print("file not found at", file_path)

```

in this code, `os.path.abspath(__file__)` gets the absolute path of the current python file, then `os.path.dirname` gets the directory of the script. then `os.path.join` helps to construct the full path, moving one level up (`..`) to the project folder then down to the `data` folder and finally to the file.

this approach is way more flexible, because you can move your whole project around and, as long as the directory structure remains the same, it will work without changing the path again, also is great when coding in a team. its all relative.

**3. handling relative paths more cleanly**

there are specific times where the above method is still not enough, and in complex structures where you have multiple scripts calling other scripts, it could become a little complex, there are other ways to mitigate these problems, like using environment variables or having a config file. but since we are in the scope of the "current file not found" lets focus on ways to fix that.

here's an approach that uses a more defined base directory

```python
import os

# sets base directory, for example the project main folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# constructs path from base directory
file_path = os.path.join(base_dir, "data", "my_file.txt")


try:
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
        print("file not found at", file_path)
```

in this example we get the directory of the script like in the previous one, but now we call it twice to go up to directory levels, this assumes that the project main directory is two levels up from the python script, this will depend on your directory structure, now the `file_path` is constructed from this `base_dir`, this can be useful if you want to have the same base directory for other scripts, and prevent issues from calling multiple files from different folders. it does rely that you know where the folder structure is.
it can also be used as the start point for a config file approach, where you define this variable once.

**further reading**

if you really want to deep dive into all the intricacies of file systems and path handling, i’d highly recommend:

*   **"operating system concepts" by abraham silberschatz, peter baer galvin, and greg gagne:** this book is a classic and has a great section on file system implementations and how paths work at a lower level. while its not python specific is a must if you want to fully understand the topic.
*   **python documentation on the `os` and `os.path` modules:** the official documentation is always a great resource. they have very detailed descriptions of all available functions, and great examples.

so, the next time you see a `filenotfounderror`, don't panic. just remember the rules of relative paths and current working directory, and you will get there, it's just that simple. oh and by the way, why did the programmer quit his job? because he didn't get arrays.
