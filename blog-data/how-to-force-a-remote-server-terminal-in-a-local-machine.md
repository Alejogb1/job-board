---
title: "How to force a remote server terminal in a local machine?"
date: "2024-12-15"
id: "how-to-force-a-remote-server-terminal-in-a-local-machine"
---

hey there, so you're looking to basically hijack a remote server's terminal and make it appear on your local machine? i've been down this road more than once, let me tell you, it's a fun trip. when i started, back in the old days of dial-up, i swear i spent weeks figuring out some of this stuff. think like 14.4kbps modems being the best thing, the good old days where a 10mb file was a serious task to download. not a cloud server in sight. good times. not. anyway, the challenge comes down to getting a command-line interface (cli) on your local machine that's directly reflecting what's happening on the remote server, as if you were physically sitting in front of it. there are a couple of techniques. i am going to give you three approaches that i have used and that i have seen been used. so here's the lowdown:

first off, the most common, and probably the one you'll end up using most of the time, is ssh. you probably already know this. but lets recap the basics of ssh forwarding which we will then use to solve the problem. it's a secure way to connect to a remote server. it lets you execute commands and get responses back in your local terminal. the key part that most people dont' know that it solves exactly what you are asking is the `-t` flag, which forces pseudo-terminal allocation, which is how you get that interactive experience you're looking for, that remote server terminal appearing on your machine. i remember back when i was working on this big data project, i had to access a cluster of servers, and ssh with the `-t` parameter was literally the only thing that kept my sanity. without it, everything was just a hot mess. imagine trying to run some complex processing script and not seeing what the remote server is doing on a live terminal session. complete nightmare.

```bash
ssh -t username@remote_server_ip
```

the `username` part is obviously your username on the remote server, and `remote_server_ip` is the server's ip address or hostname. after running that command, you'll be prompted for your password (or use a key, which i heavily recommend, using passwords to login in to your server is a security risk!). after that, you're straight into a bash session (or whatever shell they are using on that server). you can run any commands, navigate the files, etc. like you would if you were using a computer locally. you can run `top`, `htop`, or any of those system monitoring tools as if you were locally on the server.

ssh with `-t` isn’t just for simple logins. i’ve also used it to remotely execute single commands, which is particularly helpful for scripting. for instance, instead of just dropping into the shell, you could do something like this:

```bash
ssh -t username@remote_server_ip "command to run on the remote server"
```
with this, the remote server executes that command, prints any output back to your terminal, and then the connection closes. the `-t` here is still important because without it, the command might not work correctly. particularly if it’s something that interacts with a terminal, like a text editor or something that expects some form of feedback. i remember once trying to run a interactive cli app that i wrote remotely, and it wouldn't work until i had this `-t` flag set, which forced that interactive session as if i was actually in the machine.

now, lets talk about `screen` or `tmux`, because if we are aiming for the most terminal experience then we have to talk about them, they are another level, more advanced than the standard ssh way. these are terminal multiplexers, they're super useful because they let you have multiple terminal sessions within a single window, and more importantly for what you want they keep the session running even if you disconnect. this is essential if your connection is flaky. you can start a session on the remote server with screen or tmux, do your thing, then disconnect and later come back to the exact same session. when the internet breaks, you don't have to cry when you lose the connection.

first you have to install `screen` or `tmux` on the remote server (if they're not already installed, which they probably will be). for `screen` it is usually something like `apt install screen` on debian based distributions or `yum install screen` in centos/rhel ones. or use whatever package installer the linux distribution of choice has. once they are installed, then you use them.

heres an example of how to start a `screen` session.

```bash
ssh username@remote_server_ip "screen -S my_session"
```

this connects via ssh, and then immediately starts a screen session named 'my\_session'. within this screen session you can run your commands. to detach from the session you press `ctrl+a` then `d`. now, you are back in your local machine with the session still running remotely, to reattach you run the following command:

```bash
ssh username@remote_server_ip "screen -r my_session"
```

this will attach you back to the remote session that you started. `tmux` works in a very similar way, i personally like `tmux` better, you just have to get used to the hotkeys, in general terms is `ctrl+b` instead of `ctrl+a` for `screen`.

screen and tmux are extremely powerful, as you can have several virtual terminals all in the same ssh connection. you can divide the terminal into panels, move things around, etc. i remember when i was writing my master's thesis, i had all of my experiments running on multiple tmux windows in my university's server, it allowed me to disconnect, travel back home and then continue working with my experiments on the spot as if nothing happened, without losing any progress.

now, there are instances where the server is acting funny and none of these methods will work correctly because of permission issues, networking or firewall problems, but in most normal cases, these three options should cover most of what you need to do. sometimes, its not that something does not work, is that the server is misconfigured or has a very strict firewall policy that wont let you connect via ssh. i remember when i was working with a vpn server, that the ssh port was blocked because some dude on the other side of the world was trying to brute force the ssh password with some script. that experience taught me to always set strong passwords and to disable password ssh connections.

there is an alternative solution that is not really a solution that is used for extreme cases or when you need to test an operating system via the terminal. it is used to get a serial console access. this one is very technical, it involves using the serial port of the machine and a terminal emulator like minicom. it goes out of the scope of your question but it is nice to mention. for those of you who want to know more about serial ports and terminals, i recommend “the serial port complete: programming and circuits for rs-232 and rs-485 links and networks” by jan axelson. it really dives deep into the subject. it's a great resource to have around if you get into some low level stuff.

so those are the approaches i have found the most useful, ssh with `-t` for interactive sessions and quick remote execution, screen or tmux for maintaining sessions across disconnects and multiplexing, and minicom with a serial port for more exotic terminal uses.

for resources, i heavily recommend reading “ssh, the secure shell: the definitive guide” by barrett, bysshe, and silverman. it’s a classic, and it really explains the nitty-gritty of ssh. if you are new to linux administration, you should also check “linux system administration handbook” by nemeth, snyder, hein, and whaley. it will give you a good overview of everything that you might encounter on your journey to becoming a great server administrator. these two books in my humble opinion are required reading.

one last thing before i go, i once spent an entire afternoon trying to figure out why my ssh connection kept dropping. turns out the network cable on the server was only half plugged in. i almost had a complete meltdown until i noticed it. yeah. sometimes the solution is just painfully simple, and it has nothing to do with a complicated configuration file or a script going haywire. so, just check the basics first.
