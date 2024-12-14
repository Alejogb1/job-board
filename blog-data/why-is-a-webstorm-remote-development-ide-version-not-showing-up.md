---
title: "Why is a WebStorm remote development "IDE version" not showing up?"
date: "2024-12-14"
id: "why-is-a-webstorm-remote-development-ide-version-not-showing-up"
---

alright, so, you're hitting that classic webstorm remote development snag, where the ide version just ghosts on you. i've been down that rabbit hole more times than i care to recall, and it usually boils down to a few key suspects. let's break it down.

first things first, and i cannot stress this enough, is the ssh connection. i've lost countless hours because of something seemingly trivial in ssh config. make sure the connection itself is solid. can you ssh into the remote machine via the terminal without password prompts? that's ground zero for remote ide setup. no go there, it's not going to work with webstorm. i've been through situations where the ssh config on either my local or remote machines had conflicting configurations, like mismatched keys or restricted protocols. usually the error messages when ssh fails are kind of cryptic which does not really help. in my very first internship i spent two days trying to solve this and turns out the company was using a custom ssh port. rookie error. 

the second thing that usually gets me is the backend ide server itself. webstorm, under the hood, isn't just magically connecting; it fires up a headless ide process on the remote machine. this process needs to be running, and it needs to be running *correctly*. i've had instances where the ide server process got killed, either manually or from a system issue like running out of memory or a random crash. i once spent an entire afternoon trying to figure out why it was failing only to discover that an out of date java version was causing a weird exception with the remote process. you'll need to check this with the command line, if the command hangs, something is likely wrong and you should restart it. you can try to check the remote process directly. i will post the command here:

```bash
ps aux | grep 'jetbrains'
```

if you see something like 'java ...jetbrains...backend' in the results, then your ide server is running, at least we know that one process is running, but if there are multiple processes, some of them can be stale and create conflict. you have to be careful with that. make sure that the process is running as the correct user, the user you are connecting to the server with. if you have weird permissions or the user that runs the process is different than the one you use for connection, this is going to be another big problem.

if the `ps` command doesn't return any jetbrains stuff, you have a different problem you will have to check the logs, but normally webstorm automatically starts the server in the background. usually restarting the ide from the webstorm client is enough to solve this problem.

now, speaking of webstorm's automatic magic, it tries to detect a suitable ide version on the remote, but it might fail if it's not in a standard location or if there are permissions issues. i had this very case with one of my first projects, i was working with a very customized linux distribution that had it's own path for the ide, and it was just not detectable by webstorm. webstorm expects the ide to be in specific locations, it does not try to search the whole file system, which is understandable from a performance perspective, so make sure you have it installed in a path that it usually checks automatically or specify that path in the webstorm client configuration.

sometimes i also see that the firewall on the remote machine interferes with the communication. you'll have to configure the firewall to allow connections on the port that webstorm uses for remote development. sometimes the firewall is too aggressive or there is other kind of security software that can interfere. usually the easiest way to do this is to simply disable the firewall for testing or add a explicit rule. in production, obviously it's better to have a more restrictive rule, it's better to be safe, not sorry. here's a quick way to temporarily disable the firewall in linux using `ufw` assuming you have it installed:

```bash
sudo ufw disable
```

don't forget to re-enable it after you have finished testing:

```bash
sudo ufw enable
```

if you are using another firewall solution consult its documentation.

the other thing that i see quite frequently is that the version of the webstorm client is not compatible with the version of the ide on the server. this usually happens because webstorm is constantly evolving and the format of the communication protocol might change so the server ide will not be compatible with a different client. you always have to make sure you're using the matching client version with the remote server version you are targeting. this should be in the logs, so please check them carefully when things go wrong.

sometimes, just sometimes the problem is simple, and you will laugh about it, like when i forgot to enable the remote development feature on the server side. yes it happened to me, it happens to the best of us. you have to enable the remote development in the ide preferences on the remote server ide instance, it is a checkbox in the preferences, and you have to be sure it is enabled, that's the first thing you have to check if the remote ide is not listed in webstorm's client, but at that point i was so tired from debugging previous remote issues, that i did not even remember to check the basics.

as a bonus and to avoid confusion, please always double check your ip address that you are trying to connect to in the webstorm client, it seems trivial but sometimes people make a mistake in the ip address and that can lead to a lot of wasted time. i have a friend that has a similar ip address in different network environments, and he always gets confused, so he ends up trying to connect to the wrong ip address, a different machine, that does not have the ide, and he usually spends 30 minutes to an hour to figure it out, it is quite funny, but it happens. 

for resources, while i can't drop direct links here, you should look at the official jetbrains documentation, it's surprisingly thorough (i was joking). there's a section dedicated to remote development setup and troubleshooting. also, searching for 'webstorm remote development logs location' can help you find where the ide stores the relevant logs both in client and server side, those logs will be your main weapon, if something goes wrong.

in general, the process of diagnosing this kind of issue is like detective work, you have to check the logs in both sides, understand the network, and pay attention to the small details. good luck, you'll get there.
