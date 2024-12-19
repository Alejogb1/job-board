---
title: "double pipe bash usage example?"
date: "2024-12-13"
id: "double-pipe-bash-usage-example"
---

Okay so you wanna know about double pipe `||` in bash right Yeah I've been there done that messed it up a few times too Trust me you are not alone lets dive into the nitty gritty of this thing

So `||` in bash it's like the "or" operator but for commands not variables it's all about exit codes basically A command when it runs spits out an exit code zero if it's successful something not zero if it went kaput

The `||` operator checks the exit code of the command on its left If that command fails meaning its exit code is nonzero then and only then does it run the command on its right its a short circuit thing like if the left fails then try the right otherwise dont bother

Think of it like this a chain of backup plans If plan A doesn't work then try plan B and so on And yes multiple `||` are cool its like having plan C and plan D and so on This is extremely important when working with deployment scripts or automation when failures are common

I remember this one time way back probably 2010 maybe I was doing some really messy cloud deployment involving some proprietary command line tool that used to fail randomly It drove me nuts debugging at 3 am and this was before proper cloud logs were a big thing Oh the nightmares We were using some old ubuntu server and the tool kept failing when it was trying to pull container images from the private registry The command looked something like this:

```bash
proprietary-tool pull private-registry.com/my-image:latest
```

And sometimes it would work fine and sometimes not. It would just bail with a non zero exit code and no useful message at all. Classic. So this is where i discovered that the magic of `||` really is. We used to restart it manually and I was like there has to be a better way right and I thought yeah Bash can definitely help.

So I basically used this:

```bash
proprietary-tool pull private-registry.com/my-image:latest || proprietary-tool pull private-registry.com/my-image:latest
```

Yes I tried the same command twice its the easiest way when you dont have control over the error reporting of a command. It was a bit hacky but it worked. The point is if the first pull failed because of whatever network issue then the second one would try again. It did work 70% of the times.

Then of course I used to combine that with `set -e` for good measure because if you combine this with the lack of error handling a bash script can go wild and leave stuff in a really bad state. `set -e` would basically stop the whole script if a command fails and not let it continue which is crucial.

Now you might say "Oh but isn't that inefficient to do the same command twice" And you'd be right. In that case i could have checked if the file exists or something. but in a fast pace development team you often don't have time to do it perfect and that was my case. It is all about tradeoffs and sometimes the most important metric is time to fix.

Let's show you another more useful example Say you're trying to create a directory but you aren't sure if the directory already exists A typical script might try doing this

```bash
mkdir my-directory
```
but what happens if it exists You'd get an error right Not ideal

Now lets see the beauty of `||`:

```bash
mkdir my-directory || echo "Directory already exists no action taken"
```

Here if `mkdir` succeeds it does nothing and the echo command is not executed If `mkdir` fails due to directory existing the `echo` message is printed you get an output a kind of feedback to the user

Its a pretty neat way to handle errors in your script without having to write a lot of `if` statements right?

And lets say you need a fallback mechanism lets say you are trying to install a package via a package manager but that package is not always in all repos here's another classic usage:

```bash
sudo apt-get install my-package || sudo yum install my-package || echo "No such package found using either apt-get or yum sorry"
```

In this example the script tries to install using `apt-get` if it fails then it falls back to `yum` and if that fails too finally it prints the message. Notice that you can chain multiple `||` it's awesome right It's like have an if then else but for the exit code of the commands.

Now I know you are probably itching to know even more about error handling I have had my share of bash script failures and it's not pretty let me tell you.

For really deep dives into shell scripting I would suggest reading the following books: "Classic Shell Scripting" by Arnold Robbins and Nelson H.F. Beebe It covers a lot more than just `||` really a classic if you want to understand shell internals And then there is "Advanced Bash Scripting Guide" by Mendel Cooper which is available for free on line if you want something really in depth It's like the bash bible it's not a novel its really in depth though But it will make you a better script writer for sure. There is even a small section on exit codes you will enjoy.

Also just a side note remember that in bash even comments `#` start at the beginning of the line or after a command so watch out where you put your comments I had this one time where I did a silly mistake by putting a comment at the end of a command line and it became part of the command args it took me half a day to figure out why my script was throwing weird errors haha classic.

So yeah that's the deal with double pipes in bash its a quick and useful shortcut to deal with failures and give you some basic error handling but as all things do not overuse it when things get complex its always better to use more structured tools

Hope this helped. Let me know if you have other questions. I am not afraid to dive deeper. Just remember to avoid the silly comments and debug those errors.
