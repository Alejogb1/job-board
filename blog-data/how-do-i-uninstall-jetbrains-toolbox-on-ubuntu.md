---
title: "How do I uninstall Jetbrains toolbox on Ubuntu?"
date: "2024-12-23"
id: "how-do-i-uninstall-jetbrains-toolbox-on-ubuntu"
---

Alright,  I've seen this scenario play out more times than I care to count, often involving colleagues new to linux environments battling with leftover files. Uninstalling Jetbrains Toolbox on Ubuntu isn't as straightforward as, say, using `apt remove` due to its unique installation method. Instead of relying on a package manager, it usually drops files into specific directories in your home folder. This leaves a bit of manual cleanup necessary, and that's where we’ll focus.

My initial encounter with this was back in the days of Ubuntu 16.04. A junior dev, let’s call her Alice, unintentionally installed the toolbox in a rather chaotic way. It was installed multiple times, actually, each attempt leaving its own scattered remnants. This resulted in conflicts when she tried to launch any of the IDEs. After a bit of troubleshooting, I had to craft a specific process to clean it all up. And that’s where my process comes from—experience and the desire to not relive that moment.

Fundamentally, Jetbrains Toolbox operates outside of typical package management, installing itself into your user's home directory, usually under `.local/share/JetBrains/Toolbox` and `.config/JetBrains/Toolbox` for the configuration. Sometimes, there are also leftover launchers located in `~/.local/share/applications`. To completely remove the toolbox and its associated files, we need to carefully go through these locations.

First, let’s consider the scenario where the toolbox was installed following the standard procedure. Here's a three-step approach, complete with command-line snippets:

**Step 1: Removing the Application Files:**

The first, and perhaps most important step, involves removing the directory containing the actual application itself and the associated configuration. We’ll use the `rm` command with the `-rf` flag. **Be extremely cautious with the `-rf` flag as it is destructive.** Make sure you’re targeting the correct directories before proceeding.

```bash
rm -rf ~/.local/share/JetBrains/Toolbox
rm -rf ~/.config/JetBrains/Toolbox
```

This command recursively removes all files and subdirectories under those paths. The `~` symbol represents your user's home directory. We're targeting the usual installation directory of the toolbox. The `-r` flag tells `rm` to remove directories, and the `-f` flag forces removal without prompting, so again, be very careful.

**Step 2: Removing Desktop Entry Files:**

Secondly, we need to remove the launcher files that often clutter the application menu. These usually end up in the `~/.local/share/applications` directory. It's safe to delete those.

```bash
find ~/.local/share/applications -name "jetbrains-toolbox*.desktop" -exec rm {} \;
```

This command is slightly more complex. Let’s unpack it. The `find` command is used to search for specific files. `~/.local/share/applications` indicates where to look. `-name "jetbrains-toolbox*.desktop"` specifies that we’re searching for files whose name starts with "jetbrains-toolbox" and ends with ".desktop". This would encompass anything like `jetbrains-toolbox.desktop` or `jetbrains-toolbox-beta.desktop`. The `-exec rm {} \;` part takes every file that the `find` command found and executes `rm` on it. The `{}` is a placeholder for the filename.

**Step 3: Removing Cache and Temporary Files (Optional but Recommended):**

Finally, we might want to clear any leftover cache or temporary files. Jetbrains Toolbox can sometimes store temporary files or user-specific information in directories such as `.cache`. While not always critical for removal, doing so ensures a clean environment.

```bash
rm -rf ~/.cache/JetBrains
```

This command, similar to the first one, uses `rm -rf` to recursively delete the contents of the Jetbrains cache directory.

These three steps should cover the most common scenarios for removing the Jetbrains Toolbox. However, there are cases where installations were done with different versions or were installed in unconventional ways. This can sometimes lead to odd edge cases. For example, if the toolbox was somehow installed in a different directory, you’ll need to locate that directory before attempting the `rm -rf`.

Furthermore, there can be remnants in the `.config` directory not directly linked to toolbox but rather to individual IDE configurations. If you’re planning on completely reinstalling the suite, you might want to back up these before removal.

Now, let’s illustrate how these commands operate using the following hypothetical scenarios. Suppose, I installed the toolbox, created a new project, then decided to remove it:

**Scenario 1: Standard Installation:**

After a standard installation, I'd find a directory structure similar to the following under my home directory:

```
~/.local/share/JetBrains/Toolbox/
├── bin
│   └── toolbox
├── runtime
└── ...other files
~/.config/JetBrains/Toolbox/
├── config.json
├── logs
└── ...other files
~/.local/share/applications/
├── jetbrains-toolbox.desktop
└── ...other files
```
Running the commands given earlier would effectively eliminate these folders and files, removing the toolbox installation.

**Scenario 2: A peculiar installation**

Let’s imagine, for the sake of example, that I accidentally extracted the toolbox into `~/my-dev-tools/JetBrains/Toolbox`. I then created a desktop entry manually. In this peculiar installation, standard instructions will fail to delete the toolbox.

The following commands are necessary instead:

```bash
rm -rf ~/my-dev-tools/JetBrains/Toolbox
find ~/.local/share/applications -name "jetbrains-toolbox*.desktop" -exec rm {} \;
```

Here, the `rm -rf` path is adjusted to match the actual installation directory. The find command remains the same since it is targeting the specific launcher file type.

**Scenario 3: Leaving traces behind**

In this scenario, I've deleted the installation as explained above. However, traces of configuration and preferences from my specific IDEs, such as PyCharm and IntelliJ IDEA, linger in `~/.config/JetBrains/`.

Although not essential for removing toolbox, removing these may be helpful for a truly clean environment before reinstallation:

```bash
rm -rf ~/.config/JetBrains/PyCharm*
rm -rf ~/.config/JetBrains/IntelliJIdea*
```

These commands will delete all PyCharm and IntelliJ related configuration directories. Once again, note that these operations are permanent, and one might need backups if these configuration files are needed for future projects or setups.

In conclusion, the complete removal of Jetbrains Toolbox on Ubuntu is a manual process involving the removal of specific directories, launchers and optionally, cache. Being thorough is crucial, as is understanding the underlying folder structures. If the standard commands fail, it usually means the installation location or the launcher files are in an unexpected location. If you happen to find yourself dealing with such scenarios, double-check those locations before executing any `rm -rf` commands. Remember that these commands are powerful and irreversible if executed incorrectly.

For further learning, I highly recommend delving into the linux documentation for `rm`, `find` and `bash`, as they form the foundations for many day-to-day operations within the linux environment. Specific texts like "The Linux Command Line" by William Shotts and "Advanced Programming in the UNIX Environment" by W. Richard Stevens are very useful and can provide a much deeper understanding of these fundamental tools. These resources offer a much more detailed treatment of these concepts than could be covered here and are a great investment for anyone serious about working with linux systems.
