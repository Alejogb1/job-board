---
title: "In which terminal do I run these commands?"
date: "2024-12-15"
id: "in-which-terminal-do-i-run-these-commands"
---

so, you're asking about which terminal to use for running commands, i get it. it's a common newbie question, and honestly, i've been there. when i started out, i remember spending a good hour trying to figure out why my script wasn't working, only to realize i was typing the commands into the wrong window entirely. that was back in the early 2000s, i was fiddling with linux kernels and compiling stuff from source. good times, frustrating times.

the short answer is: it depends on what you’re trying to do and where your code should run. there isn't one single, universal terminal. you need to match the terminal to your target environment. let's break down some common scenarios and the terminals that go with them.

first off, let's talk about the basic concept. a terminal is essentially an interface where you type in text commands, and the computer interprets those commands and executes them. it's how we talk to the operating system, how we tell it to do things. the terminal itself isn't where the code runs but it provides the user the interface to communicate with the operating system which then executes programs for you. think of it as a translator. so, where does your code run and where do you type the translation?.

if you're on a local machine, meaning your own computer, whether it's windows, macos, or linux, you'll usually use a terminal application that's specific to that operating system.

on windows, the most common terminal is powershell. you can also use the command prompt (cmd) but powershell is more powerful and generally preferred by developers. to open powershell you can typically search for it in the start menu. there are other terminals available for windows too, like the windows terminal, which gives you a more modern interface and supports tabs. but powershell is likely what you'll be using the most. powershell is also compatible with core powershell which is cross platform, so you can use powershell on windows, linux and macos and run most of your powershell scripts.

macos, on the other hand, comes with terminal.app, which is a bash terminal by default. if you've never customized it, it’s likely you’re running bash in terminal.app. you can find terminal.app in the applications/utilities folder. you also have other options like iterm2 which is a very powerful terminal emulator, that is quite common among developers. it has tons of features that makes it useful.

linux distributions generally include some version of a terminal application, often xterm, konsole or gnome-terminal. the default terminal emulator can vary between distributions, but most are compatible with bash or zsh. so any terminal emulator will work as long as it uses a compatible shell like bash, zsh or similar. just search in your applications for ‘terminal’ and you will find one available to use.

now, let’s say you’re trying to run code on a remote server. in that case, the terminal you open locally will only act as a window for interacting with that remote system via a protocol called ssh (secure shell). this is especially common when working with cloud services or remote development environments. you’ll open a terminal on your local machine and then use the ssh command to connect to the remote server. once connected, your commands will be executed on that remote machine, not locally.

here's a basic example of connecting via ssh, assuming you have an ssh server running on a machine with the address `remote_server.example.com` and your username is `user`:

```bash
ssh user@remote_server.example.com
```

once you’re connected through the ssh command, anything you type will be executed on the remote computer. you can test this for example by typing `pwd` to see in which directory you are in the remote server. it’s a completely different filesystem. it's not your computer.

another scenario is using containers. docker is a popular containerization tool. you might need to execute commands inside a docker container. in this situation you first need to start a docker container and then, using the `docker exec` command, you can access a shell session within the running container. this is useful for debugging or running specific commands within that isolated environment. for example:

```bash
docker exec -it <container_name> bash
```

in this command, `<container_name>` will be replaced by the name of your docker container. it will open a shell session where you can then use commands as if you are physically inside that container. `bash` in the example command indicates to start a bash shell inside the container. you can use `sh` if you want another simpler shell. the `-it` flag allows for interactive terminal access.

then there are situations when you don’t use a terminal program directly, for example when you use a code editor like vscode. visual studio code has an integrated terminal window. this window usually uses the same terminal available in your operating system. so it could be powershell, bash or similar depending on your configuration and operating system. so when you open vscode and type something in its terminal it’s running in the local machine terminal.

when working with virtual environments, especially in languages like python, it’s important to remember to activate the virtual environment in your terminal session. this modifies the environment to use packages installed in the virtual environment and not your system packages. so, if i want to use my virtual env called `myenv` i will use this command:

```bash
source myenv/bin/activate
```

this command is meant to be run in the specific directory where your virtual environment was created. once activated, you will see your prompt change and usually show the name of the activated virtual environment. after this your python code will run on this virtual environment. once you close the terminal that environment is no longer activated, which is useful to avoid accidental conflicts.

a common mistake when starting out is thinking that a command will run everywhere, irrespective of where the terminal was opened. this is not true, each time you open a terminal, you start a new shell session. each session operates on a specific context, including the specific directory or environment. so you have to be careful. the commands will operate relative to that context and not to every directory and computer on your system.

so, in summary, there isn’t one "right" terminal for all commands. it depends entirely on where you need the commands to be executed: on your local machine, a remote server, or inside a container. you need to be aware of the differences between them, and what environment each terminal session is affecting.

now i remember this one time when i was doing some networking stuff, and i was typing commands into a docker container shell while i was connected via ssh to the server and my system wasn’t responding at all. it took me like 10 minutes to realize i was connected to a third layer of abstraction and i was debugging the wrong computer entirely. it was a really bad moment for me. i felt really stupid, i almost deleted my entire production database. that was a monday morning if i remember it correctly, it wasn't my finest hour. after that i started drawing diagrams to better understand what was running where, and that helped a lot. it was like learning to ride a bike, a really hard bike to ride.

as for resources, instead of random links i would recommend "understanding the linux kernel" by daniel p. bovet and marco cesati for a deeper understanding of how the operating system works under the hood, and "tcp/ip illustrated" by w. richard stevens for understanding network protocols. also "the pragmatic programmer" by andrew hunt and david thomas is a great resource for general software development best practices and helps to create good work habits for developers of any kind.
