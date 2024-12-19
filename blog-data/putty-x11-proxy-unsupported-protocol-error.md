---
title: "putty x11 proxy unsupported protocol error?"
date: "2024-12-13"
id: "putty-x11-proxy-unsupported-protocol-error"
---

Alright so you're getting the `putty x11 proxy unsupported protocol error` right Been there done that got the t-shirt and the scars to prove it Let me tell you this isn't some arcane magic it's usually pretty straightforward but man oh man it can be frustrating if you don't know what to look for

So first thing first let's unpack what's happening You're trying to use PuTTY to tunnel X11 traffic which basically means you want to display graphical applications running on a remote Linux machine on your local Windows box via some ssh tunneling magic right When PuTTY throws the "unsupported protocol" error it means something in that chain is not happy generally either your settings are off or the server isn't set up to play nice with x11 forwarding

I've seen this pop up in the weirdest situations like way back when I was still rocking Windows XP yes that long ago I was trying to debug a rendering issue on a server across a dodgy VPN the server side was running some old x window server version and I just banged my head against the wall for hours. But don't worry we'll get you sorted

Here's my usual troubleshooting checklist broken down by the typical points of failure from my many painful hours dealing with this thing

**1 Local PuTTY Configuration is Key**

The first place to double check is your PuTTY settings If X11 forwarding isn't turned on in your session config that's your problem I swear it’s always the simple things that catch you out

Go to the PuTTY Configuration window look for `Connection` then `SSH` and then click on `X11` Make sure `Enable X11 forwarding` is ticked Also the `X display location` usually is set to `localhost:0` or sometimes you might need to use `localhost:10` If you're tunneling over a VPN or the local network is set up strangely this sometimes might be the case It’s not generally needed though you’ll usually find that your display is zero and that’s fine.

Sometimes you need to specifically change the X display location to address a case where another service occupies the port which is rather common especially in the university labs. Check which X server you are using on your local machine and change the number in display if needed but usually `localhost:0` works best. Now the X display location setting on putty side this needs to match the DISPLAY environment variable on the server side otherwise its like yelling at your wall hoping it responds.

Here's what I usually do in my putty config for reference. (not code but visual reference)

```
Connection -> SSH -> X11
    [x] Enable X11 forwarding
    X display location: localhost:0
    (unchecked) X authority file for MIT-Magic-Cookie-1
```
You see I'm a minimalist i only tick and configure what I need

**2 The Server Side Setup is Important too**

Okay so PuTTY settings are right next thing to look at is the server itself. X11 forwarding has to be enabled in the sshd configuration. If not PuTTY is trying to do something the server doesn't allow it's basically a recipe for error messages.

Connect to the server with a regular ssh session. I know you're trying to do X11 but let's confirm we can at least get in and the SSH connection is stable. Once connected you gotta have super user permissions to look at the sshd config. Usually it’s in `/etc/ssh/sshd_config`.

First check that x11 forwarding is enabled.

```bash
sudo nano /etc/ssh/sshd_config
```
You’ll be looking for the following line it’s probably commented out or set to `no` and if it is you need to change it to yes

```
X11Forwarding yes
```

If it doesn't exist you can add it to the config file. After editing save the changes usually Ctrl+X and then Y

Now the change you made requires you restart sshd service to take effect

```bash
sudo systemctl restart sshd
```
That line is a lifesaver if the server hasn't been set up for X11 forwarding in the past I spent days trying to understand why my X display is not working before I realized i had to edit this config file you gotta have this set up right otherwise you get absolutely nowhere

**3 Your Local X Server Might Be Throwing a Tantrum**

You probably have an X server installed locally if you are using Xming X410 or vcXsrv which you must have to make this whole thing work if one of these guys is not running on your machine well that’s why it won’t work It’s a bit like trying to watch tv without power it’s just not gonna happen.

I mostly use Xming so I’m more comfortable with troubleshooting using that. If your server isn't receiving any display messages or you can't see them on the local machine it's time to check if your X server is working properly. Try and run the Xming program or the Xserver you are using on your machine it should have a tray icon if its running.

The way to test this is by opening a terminal and typing `xclock`. It displays a very simple clock program usually if this works then you know your xserver is working correctly. So try that first to rule this out.

**4 The DISPLAY variable**

This one's a real troublemaker the DISPLAY environment variable on the remote server has to point to a valid X11 display. It should normally be automatically set up correctly by PuTTY but you need to check that just to be sure.

After making sure the X11 forwarding and that your X server are working you can try to make an ssh connection to the remote machine and check your DISPLAY environment variable

```bash
ssh -Y your_username@remote_host
```
`-Y` means trust X11 forwarding so if you're running a local server that allows connection from other host this would be the way to do it otherwise it would be `-X` which is not secure

Once you are connected to your remote server check the value of the display variable with `echo $DISPLAY`.

Normally it will look something like `localhost:10.0`. If it’s not set or if it’s incorrect then you have found your culprit. You can usually try setting it up manually by typing `export DISPLAY=localhost:10.0` and then try `xclock` but ideally it should be configured automatically by PuTTY. If you need to set it up manually every single time then you probably missed something while configuring your PuTTY X11 options.

You could also check the xauth file used for authentication between the server and client usually on the server side you would use `xauth list` and it will display some keys and host information you can then check if your server can communicate with the client.

**5 Firewalls can make it problematic**

Sometimes those pesky firewalls can cause issues I had this happen at work once we installed a new firewall and it blocked X11 connections across subnets it was a nightmare to find that out you sometimes don’t realize it’s the problem because all the settings seem correct but it’s always the firewall

Make sure your firewall settings on both your local machine and the server aren’t blocking ports used by X11 usually the X server listens on 6000 + display number so if your display is `localhost:10.0` that would be port 6010 for local machine and the same port should be open for the remote machine. There is no code for that just making sure the firewall isn't a kill joy.

I know this is a lot of information and I can imagine this might be a little tedious but you will need to go through all of this if you want to make X11 forwarding work.
So basically if I had to summarise you need to ensure that

1 PuTTY settings for x11 forwarding are correct
2 The server side sshd configuration is correctly set for x11 forwarding
3 Local x server is running and configured to listen on the correct port
4 DISPLAY variable is correctly set on the server
5 Firewall on both server and client are not blocking X11 traffic

If you have checked all these things and you are still having issues then you probably have a very exotic setup.

Finally if you are trying to learn more about the X window system and how it works I would suggest you read the "X Window System: Core and Extension Protocols" which is a book describing in detail how the X server works and its protocols it has all the nerdy details you might want to know about the system. You could also read "UNIX Network Programming Volume 1 The Sockets Networking API" if you want to understand how the network communication and the socket stuff works with all of it. They are classics in their own way and always help to understand stuff in depth. Don’t get lost in too many resources though just try to follow my steps to debug your issue. Also don't make the mistake I did when I started learning about it where I thought I should use telnet for x11 forwarding man did I look stupid. Anyway I hope this clears up your issue have fun with the remote GUI.
