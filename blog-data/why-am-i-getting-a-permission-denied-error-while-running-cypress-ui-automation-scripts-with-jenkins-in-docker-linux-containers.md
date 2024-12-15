---
title: "Why am I getting a Permission denied error while running Cypress UI automation scripts with Jenkins in docker linux containers?"
date: "2024-12-15"
id: "why-am-i-getting-a-permission-denied-error-while-running-cypress-ui-automation-scripts-with-jenkins-in-docker-linux-containers"
---

alright, so you're hitting that classic 'permission denied' error when trying to run cypress ui tests inside a docker container orchestrated by jenkins. yeah, that's a real head-scratcher at first, been there, seen that. let's unpack this. it's almost always about file system permissions, especially when docker, jenkins, and cypress are all playing together. i've debugged this exact thing more times than i care to count, often late at night fueled by too much coffee.

first off, let's get the lay of the land. you've got jenkins, a job scheduler; docker, a containerization tech; and cypress, your ui testing tool. they all run as different users within the linux ecosystem inside your container. the problem often arises when the user running cypress doesn't have sufficient permissions to access the files and directories it needs, usually because of file ownership. jenkins often runs its jobs as a specific user, and that user may not be the same as the one inside the docker container, leading to the error.

when i first ran into this, it was a nightmare. my initial setup was a mess. i had a jenkins job that would spin up a docker container, install cypress, and then try to run the tests. and boom! permission denied. i remember spending hours staring at log files, thinking, 'where did i mess up?' turned out the docker image i was using didn't set things up correctly. the cypress executable, all the node_modules, and the test files were all owned by root, and jenkins was running as a user called 'jenkins'. the jenkins user couldn't touch anything.

the core issue here is that the user inside the docker container that cypress runs under needs to have read, and often write permissions, to the directory where it is running and all sub directories. this can be the directory where your project code is mounted, cypress config files are located, and where the test videos and screenshots get written. we're looking at basic unix user/group permissions concepts here.

let's look at some ways we can sort this out. here are a couple of approaches that i've seen work pretty consistently:

**1. aligning docker user with jenkins user:**

this is a clean solution in most cases. you want to ensure the user running inside the docker container is the same user that jenkins is using. you can do this by modifying your dockerfile to specify the user at build time and use it for all subsequent commands. or change the user when the container starts with command line parameters.

here's what that looks like in a dockerfile. i am assuming you have a 'jenkins' user. if not, make sure you have one, either created at build time or by jenkins itself.

```dockerfile
# ... your docker base image instructions...

# create user, if it doesn't exists
RUN adduser --disabled-password --gecos "" jenkins

# set ownership and permissions
RUN chown -R jenkins:jenkins /app

# change user, and all subsequent commands will be run as this user
USER jenkins

# now do all the rest, install cypress etc
```

after you build this image, when you launch it with jenkins, make sure all commands are executed as 'jenkins' inside the container. all permissions should align, and no more denied permissions (in this case). in some cases, if you build using root, files can still be owned by root. make sure to change ownership using 'chown -R jenkins:jenkins /app' where /app is your app root.

**2. changing ownership at run time:**

if you can't change the dockerfile, for example when using a pre-built one from docker hub, you could change ownership when the container launches, using a docker-entrypoint.sh script.

here's a basic shell script example:

```bash
#!/bin/bash
# docker-entrypoint.sh

# change ownership of your application dir. assuming /app is your folder
chown -R jenkins:jenkins /app

# start your cypress tests
exec "$@"
```

then in your docker run or docker compose command:

```bash
docker run -it --entrypoint /docker-entrypoint.sh <your-image> cypress run
```

this will execute `/docker-entrypoint.sh` first to change permissions and then run the cypress test.

**3. using group permissions:**

you can also use groups to share permissions, this is common in multi-user system environments, although in your situation its not as necessary as you are using a docker container that should be self-contained. but here's the general approach. if both jenkins and the user cypress runs under are members of the same group, you can give the group read write permissions.

```bash
# in dockerfile or before running your tests
RUN addgroup jenkins
RUN adduser  cypress --disabled-password --ingroup jenkins --gecos ""
# make sure to change user in docker file as well using USER cypress
RUN chown -R cypress:jenkins /app
# allow group to read write in app directory
RUN chmod -R 0770 /app
```

and if you were to set a group named 'jenkins' that all jenkins related users share, then you set that group to own the folders and have read write permissions. it makes user administration slightly easier.

remember to use `-r` flag in `chown` and `chmod` to recursively apply these changes to all files inside the target directory. also, `-R` and `-r` flags are the same.

a common gotcha is making sure your volumes are mounted with the correct permissions too. when you use docker volumes, the files are often owned by the user that launched the docker container on the host system. and if your local host user is different than the user inside the container, you will end up with the same permission denied error. i've spent hours trying to fix permission issues inside a container just to realize the host mounting was wrong. so, make sure you understand how docker volumes work and the ownership permissions of the host, to avoid these kind of situations.

these solutions are based on unix user and file permissions fundamentals. if you want a deep dive check out "understanding the linux kernel" by daniel p. bovet and marco cesati, especially the chapter on user and process management. that stuff is gold. also, "operating system concepts" by abraham silberschatz, peter baer galvin, and greg gagne is also a great reference for operating system fundamentals, in general. both those books gave me a solid grounding on how permissions work at a low level. or even the book "unix systems programming communication, concurrency and threads" by kay a. robbins and steven robbins.

remember to adjust the examples to your use case, specially the user names and directory locations. this stuff can be tricky, but with a systematic approach you will get there.

i once spent a whole day debugging a similar problem, turns out the directory was mounted read-only. i thought i was losing it, the solution was literally a one character fix (removing 'ro' from the mount options). it was a moment i’ll never forget, i laughed so hard i cried. it is all about the small details.
