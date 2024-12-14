---
title: "Why is Vertigo (mutation testing for Solidity) command not found on Mac OS?"
date: "2024-12-14"
id: "why-is-vertigo-mutation-testing-for-solidity-command-not-found-on-mac-os"
---

alright, so you're having trouble with the `vertigo` command not being found on your macos, huh? i've been down that rabbit hole more times than i care to remember, and it usually boils down to a few common culprits. let's troubleshoot this like we're debugging a tricky smart contract.

first off, `vertigo` isn't a command that magically appears. it's a tool for mutation testing solidity smart contracts, which implies it needs to be installed first. think of it like a library you need to import into your python script, except in this case, it's a tool available on the command line. the most common way to get it is via `npm`, the node package manager. assuming you already have nodejs and npm installed, here's the first thing you should be checking:

**1. installation and global path**

have you actually installed vertigo globally? not every installation is created equal. sometimes we end up installing things locally in a project folder, not realizing that the command line won't find it there.

open your terminal and try this:

```bash
npm list -g --depth=0
```

this command lists all globally installed npm packages, but only at the top level of the dependency tree. look for `@ethereum-ts/vertigo` in that list. if it's not there, then you've got your answer, it's not installed globally. it's like trying to use a function that doesn't exist within a module you are importing. the computer has no idea what you are talking about. the `vertigo` command is missing. 

if it's missing, install it globally with this command:

```bash
npm install -g @ethereum-ts/vertigo
```

the `-g` is crucial, it tells npm to install the package globally, putting the command-line executable in a place where your shell can find it. after installation, try again using:

```bash
vertigo --version
```

that should give you the version of vertigo installed. if you see the version, good. you are in the path of progress. if you get a 'command not found', we need to do further investigation.

**2. path environment variable**

even if you've installed `vertigo` globally, it's still possible your terminal can't find it. this is often because the directory where npm stores global executables is not in your shell's `path` environment variable. the path variable tells your terminal where to look for executable files. think of it as a list of directories where your computer searches when you type a command in the terminal, like when searching for the proper function in a class definition.

to see your current path, use:

```bash
echo $path
```

this will output a colon-separated list of directories. look for a directory that contains npm's global executables. the exact location varies depending on your setup, but common paths are something like:

*   `/usr/local/bin`
*   `/usr/bin`
*   `/usr/local/lib/node_modules/bin`
*   `~/.npm-global/bin`

if the directory where `vertigo` was installed is not in your path, you need to add it. usually, npm tells you where it installs things. when you install something using npm -g.

now, you will probably need to look at how npm's global packages get installed. on linux and mac systems, they are usually installed to `~/.npm-global/bin`, but sometimes that is not the case depending on how you installed npm, if you changed the defaults or any particular configuration you have on your system. so first check that, try:

```bash
npm config get prefix
```

this tells you where npm is installed. so if the `bin` is not at `~/.npm-global/bin`, then it's somewhere else. after knowing that, we proceed to edit the `~/.zshrc` or `~/.bashrc` depending on what shell you are using.

to temporarily add a directory to your path for the current terminal session, use:

```bash
export path=$path:/path/to/your/npm/global/bin
```

replace `/path/to/your/npm/global/bin` with the actual path to npm's global executables, like `/usr/local/bin` or what `npm config get prefix` gave you, plus the `bin` folder. after doing this in the current terminal try using `vertigo --version` to confirm it works. if that worked, it's time to make this change permanent.

to make this permanent for every new terminal session you start, open your shell configuration file. if you are using zsh, it is usually `~/.zshrc`. if using bash, it is usually `~/.bashrc`

```bash
nano ~/.zshrc
# or
nano ~/.bashrc
```

go to the end of the file and add that `export path` command you used before, replacing `/path/to/your/npm/global/bin` with your actual npm path. save the file and close it. to make the changes apply to the current terminal window run:

```bash
source ~/.zshrc
#or
source ~/.bashrc
```

now the path variable should be fixed, and `vertigo` should be discoverable, but there is another very common problem that you could be encountering and this leads to our next section.

**3. permission issues**

sometimes you might install packages globally, but they still can't be executed. this can be due to file permission issues. it's like having the right code in your contract, but not being able to actually deploy it because you haven't signed the deployment transaction.

you can check the permissions using this in your npm global packages bin folder:

```bash
ls -l /path/to/your/npm/global/bin/vertigo
```

look at the output. it will look like something like:

`-rwxr-xr-x 1 user group  123456 Aug 20 10:00 vertigo`

the important part is the `-rwxr-xr-x` string at the beginning. this tells you what users can do with the file. specifically, `rwx` means read, write, and execute. if the execute permissions are not set for 'all', you might have issues running the command.

to add execute permissions, use:

```bash
chmod +x /path/to/your/npm/global/bin/vertigo
```

this command tells your operating system that the `vertigo` file in that folder is an executable and allows users to run it.

**4. nodejs and npm version**

this problem is very uncommon, but it is worth mentioning that sometimes you might have a very old version of nodejs or npm. while this is unusual, it has happened to me in the past (once i went as far as to try using a totally different computer), it's worth checking. outdated versions can sometimes cause weird compatibility problems, and the vertigo package may depend on some newer api.

check your nodejs version:

```bash
node -v
```

and npm version:

```bash
npm -v
```

make sure you have a reasonably recent version. if you do not have a recent version. follow the instructions on the nodejs official page to install a newer version.

**5. the vertigo package itself**

it’s very rare, but it's happened once or twice to me while using alpha versions, there can sometimes be issues with the package itself, but this happens very rarely. most of the time the problem is the path, permissions or that you haven't installed the package, or in the uncommon case the version of node and npm.

you can try to force a re-installation, or a specific version install, you can try those as a last resort, but it rarely is needed.

```bash
npm uninstall -g @ethereum-ts/vertigo
npm install -g @ethereum-ts/vertigo@latest
```

or

```bash
npm install -g @ethereum-ts/vertigo@<specific-version>
```

you can find a list of valid versions in the npm website.

**general debugging tip**

sometimes when you are having problems, just uninstall and install again to force a fresh installation, like doing a git clone from a repository, it may solve the problem by itself.

**books and further reading**

i'd strongly recommend diving into the following resources to better grasp the underlying concepts and tools:

*   "effective modern c++" by scott meyers for the principles of modern c++, although it's not the same as solidity, some principles apply in regards to how dependencies and libraries interact on a project.
*   "understanding operating systems" by ida m. flynn and ann mclver mcwhoe to get a clearer picture of operating system structures and processes, especially regarding the file system and path variables. the file system structure on unix and linux, and even macos, is key to understand how the computer searches for executables.
*   "javascript: the definitive guide" by david flanagan for a deep dive into javascript and node.js ecosystem, as that's the base of how npm works under the hood.

i hope this helps, getting those path variables sorted always feels like getting a tricky state machine right. if i had a dollar for every time i had to deal with path variables, i'd have like, 20 dollars. anyway, let me know if you have any more questions, or if you solve it, i'd also like to know so we can update this knowledge.
