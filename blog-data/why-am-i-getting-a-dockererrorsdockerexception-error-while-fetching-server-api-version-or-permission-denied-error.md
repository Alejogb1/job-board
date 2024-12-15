---
title: "Why am I getting a docker.errors.DockerException: Error while fetching server API version or Permission denied error?"
date: "2024-12-15"
id: "why-am-i-getting-a-dockererrorsdockerexception-error-while-fetching-server-api-version-or-permission-denied-error"
---

alright, so you're hitting that fun `docker.errors.DockerException: Error while fetching server API version` or `Permission denied` error with docker, it's a classic. i’ve been there, trust me, more times than i care to count. it usually means docker itself isn't playing nice with how you’re trying to access it, and there's a few likely culprits. i’ll break down what i’ve seen cause this, and what i usually do to fix it, based on my own trials and tribulations, mostly from the early 2010s when docker was just getting its legs.

first up, the “can’t talk to the docker daemon” scenario. this usually throws the `Error while fetching server API version` message. it basically means the docker client (the command line tool or whatever python library you are using) can't connect to the docker daemon (the background process that actually runs containers). think of it like trying to make a phone call, but the phone lines are down. now, this can happen for a few reasons.

one reason which i got hit with back in 2014 when i was playing with containers on my old fedora box was that the docker daemon isn't even running. this is a simple but sometimes overlooked one. on most linux systems you can check this out via:

```bash
sudo systemctl status docker
```
or if you are on mac or windows using docker desktop:
```bash
docker version
```

 if it's not active, you'll likely see a ‘inactive’ or similar message. if that’s the case, starting it is straightforward, again on linux:
```bash
sudo systemctl start docker
```

if you're on a mac or windows machine, restarting the docker desktop app usually does the trick. if you are running docker within a virtual environment as i did back then that can sometimes cause a conflict too, but i won’t dwell too much into that since that isn’t a common case, but it is something to keep in mind if your setup is a bit more involved.

another common reason for this error is that you might be trying to connect using the wrong method. docker typically communicates via a unix socket. normally this socket lives in `/var/run/docker.sock`. sometimes the docker client doesn't know where to find it or might be configured incorrectly. you might have the `DOCKER_HOST` environment variable set to something incorrect, especially if you have been doing some funky setup or switching from local to remote docker environments or some such.

you can check if the environment variable is set by doing:
```bash
echo $DOCKER_HOST
```

if it’s set to anything other than nothing, try unsetting it or setting it to the default unix socket connection:
```bash
unset DOCKER_HOST
```

or if you are using docker context instead:
```bash
docker context use default
```
which should make your commands interact with your local docker installation.

now, let's move onto the `Permission denied` error. this one screams user permissions. specifically, your user might not have the access rights it needs to interact with docker. the docker daemon runs as root and uses that socket to talk to clients. to avoid having to use sudo all the time (and honestly, who wants to type `sudo` every single time they try to run docker) you can add your user to the `docker` group.

this is how i fixed this issue when i was messing around with a debian virtual machine during my studies. if you are on linux and don’t have the docker group you have to create it:
```bash
sudo groupadd docker
```
and then you have to add your user to that group:
```bash
sudo usermod -aG docker $USER
```
then you need to refresh your group information, the easiest way to do that is by logging out and back in or restarting your terminal, and check that you are part of the docker group by:
```bash
groups
```

you should see docker in the list.

if you're dealing with more complex scenarios, like remote docker daemons or setups where you need to authenticate, things get a bit more involved. you might have to delve into things like TLS certificates, or properly setting up environment variables like `DOCKER_TLS_VERIFY` and `DOCKER_CERT_PATH` if you're using TLS authentication. and while i haven't really had issues with that since, for example, the docker machine days back in 2015 i've seen the community struggle with this issue on shared environments or cloud machines quite a bit.

in terms of other resources that may help, i'd recommend checking out "the docker book" by james turnbull. it’s a comprehensive guide that covers a lot of the underlying mechanics of docker and might shed light on some of the less obvious parts of networking, permissions and such. also, the official docker documentation is also your friend. it’s actually pretty well written and up to date. that has been my go to for debugging issues and it can be a life saver when something new and weird happens.

now, i guess the best thing we can do is go through your specific setup to try and figure out what you are dealing with since without more details it is kind of a guessing game, i don’t know if your setup is within a virtual machine or on bare metal, what is your os, what you are actually trying to do. if you can give more context we can go through it more methodically.

finally, a little bit of humor is always needed. what did the docker container say to the other? “i've got to run”. bad jokes aside i hope i've at least pointed you in the direction that’s going to help you get this sorted. debugging these kind of issues can be irritating, but stick with it, you will nail it.
