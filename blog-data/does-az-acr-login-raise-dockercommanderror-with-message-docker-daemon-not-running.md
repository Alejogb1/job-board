---
title: "Does az acr login raise DOCKER_COMMAND_ERROR with message docker daemon not running?"
date: "2024-12-15"
id: "does-az-acr-login-raise-dockercommanderror-with-message-docker-daemon-not-running"
---

Alright so you're hitting that classic Docker daemon not running error when trying to use `az acr login` gotcha I've wrestled with this gremlin more times than I care to admit Let's break it down and get you sorted

First off yeah it absolutely can happen `az acr login` itself doesn't directly interact with the Docker daemon it kind of expects it to be humming along in the background like a well-oiled machine It’s a bit like ordering food and expecting the kitchen to be open if the kitchen aka the Docker daemon is closed you ain't getting your meal aka a successful login

Now the error you're seeing `DOCKER_COMMAND_ERROR with message docker daemon not running` it’s pretty clear and usually means exactly what it says your Docker daemon isn't running simple as that It's the foundation for all things Docker and if it is not on its just as expected that anything that needs the Docker to work is going to fail

I remember one time back in my early docker days this was a real head scratcher I was spinning up a bunch of containers for a prototype and I swear I’d started the Docker Desktop and it just decided to not play nice Turns out it had crashed silently and I had just assumed it was running It's a classic case of "check the basics" something I sometimes still forget to do even after all these years

So how does `az acr login` even get mixed up in all of this well `az acr login` uses the Docker CLI under the hood to handle authentication with Azure Container Registry which means if the Docker daemon is MIA the Docker CLI throws this error back at you and it surfaces through the Azure CLI as a `DOCKER_COMMAND_ERROR` because well it cannot connect to the docker daemon obviously

Let me walk you through some common fixes and some checks we can do because lets face it just saying "turn it on" isn’t always the most helpful when things get complex

**First things first let’s check the Docker Daemon**

Okay I know it seems obvious but really double triple check it's running this looks a bit different depending on your OS so here is a quick checklist for the most common ones

*   **Windows/macOS**: Make sure Docker Desktop is running. Check the system tray icon to make sure it's active. If you can see a whale you are likely good to go If not click on it and launch it

*   **Linux**: You might need to start the Docker daemon using systemd. You'd probably know if it was manually installed by now in this case `sudo systemctl start docker` is your friend after that check its status `sudo systemctl status docker` to be really sure.

*  **Linux (other)** check the docker service running using command `service docker status` if it is not then the command `service docker start` will do the trick and to double check use again `service docker status`

After doing the steps make sure to try your `az acr login` command again if that was the problem that should be good to go hopefully

Now it might be that Docker is running but it still isn't working. This can happen and is really annoying but trust me I went through that. Docker being running and responding means nothing if it's doing so in a wrong way.

**Okay Docker's running but I still get errors what now**

Alright sometimes the daemon is up but it isn't behaving as it should and its the dreaded "it works on my machine" scenario This is where I go and check what is actually happening within docker

*   **Check Docker's logs**: Check Docker logs for any errors or warnings this one is very helpful to see what Docker itself is complaining about

    `docker logs --since 1h`

    This command will show you logs from the past hour for all running docker containers I usually run it when I see something is up with my Docker. It is quite handy to see where is the problem

*   **Docker context issues**: Sometimes the Docker context gets messed up and it is not obvious.

    `docker context ls`

    This will show all available docker contexts I use it often to check whether docker knows where it is working from and how to do the connection. Make sure the current context is correctly configured and your user has permissions to work with the docker.

*  **User permissions on docker**:  It is not uncommon to start Docker as sudo and forget about the proper user and its permissions. Sometimes if you run docker as sudo you need to do everything as sudo. But its best not to go into those types of situations and check user groups and their docker access. I tend to use the command below

    `sudo usermod -aG docker $USER`

    This will add the user to the docker group allowing it to run docker commands without needing sudo every time.
    After doing this you'll probably need to logout and login or run `newgrp docker` to apply changes.

**And a little more complex problem**

Alright so I've seen my share of weird issues and sometimes the problem might not be just the docker daemon being off but actually something more nasty. One of the times the problem was a corrupted docker installation. Now that is some fun I can tell you that. Lets see what you can do:

*   **Restart Docker**
Sometimes it’s just a bad day for the docker. Simply restarting the docker application from your system tray should solve it. It has solved it for me countless times. But if it fails go to the next step
*  **Restart the computer**
I know I know sometimes its just better to restart and see whether it goes away. Computers do have their own logic and restarting does help sometimes
*   **Reinstall Docker**
If all else fails it might be an installation problem and you might need to reinstall docker. Make sure you have your current docker configurations backuped just in case but usually it’s just reinstall and it is good to go. Reinstall is also an option that can solve weird permissions problems

**Code examples to illustrate common problems**

To give you a taste of how this works here are some commands that might be helpful when checking what is happening. You know sometimes an example is much better than a thousand words

First example just trying to connect to the docker daemon and see if its alive

```bash
docker info
```

If this works you should get information about the docker daemon like server version storage drivers and so on. if it fails then it means you have no connection to the docker daemon and you need to check everything. If it is working this is good progress.

Another example I often use to try and see if docker is really alive is try to run a simple container something like

```bash
docker run hello-world
```

This will try to run the hello-world image and see if its working. If it's working it will simply print hello from docker and thats it. If not well it will fail and you will have an error that you can then start troubleshooting. The docker daemon should be started and available for that.

Finally lets try a command to check images currently available on your local machine to see if docker can access them

```bash
docker images
```

If this command works you should see all available images on your local machine. If it fails then something is wrong and you need to go and check the steps above again.

**Resources**

So for in-depth knowledge I strongly recommend the following

*   **"Docker Deep Dive" by Nigel Poulton**: This is a great book for understanding Docker from the ground up it goes really deep into the mechanics of Docker its inner workings and nuances. Its almost a must read for someone working with Docker
*   **The official Docker documentation:** The official Docker site documentation is very helpful for any specific thing or problem. Sometimes I just search there when I'm stuck and there is always something helpful in there.
*   **The Azure Container Registry documentation:** Microsoft also has excellent documentation for their tools and ACR is no exception you should always consult it if you have a problem with it

**Final notes**

I've been in situations where the problem was so obscure that it took days to solve it. So don’t be frustrated. I had problems where my computer time was out of sync which was messing with Docker certificates making it failing to connect with the daemon and I had another where my docker image was so big it was causing a timeout during connection. It’s funny how much we all suffer with Docker but I also had times that everything was working as expected. So keep in mind that sometimes it might be a silly thing and sometimes a more complex one.

I really hope this helps you get your `az acr login` working. Let me know if you hit any more snags and happy coding
