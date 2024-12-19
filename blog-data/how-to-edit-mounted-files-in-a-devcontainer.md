---
title: "How to edit mounted files in a devcontainer?"
date: "2024-12-15"
id: "how-to-edit-mounted-files-in-a-devcontainer"
---

alright, so you're asking how to mess with files inside a devcontainer while it's running, right? i get it, been there. it's like trying to perform surgery on a spaceship while it's in orbit – gotta be precise.

first off, let's clarify what's probably happening. devcontainers, at their core, are docker containers. they usually mount a local directory on your machine into the container. this is how you see your project files inside the container, and how changes get reflected both inside and outside it. the container itself is pretty much a self-contained system, so it’s important to understand that we are not altering the image of the devcontainer, we are altering files inside of the running container instance.

now, the easiest way to edit files is through your ide, assuming that's how you're interacting with the devcontainer. vscode, for instance, has a really nice integrated experience for this. you open a folder or project in vscode, and if it finds a `.devcontainer` directory, it’ll ask you if you want to open it in the devcontainer. once you say yes, vscode handles all the connection and mount magic. any changes you make in vscode's editor are directly saved into the mounted directory inside the container, and they're instantly reflected on your host filesystem and the container environment. the same applies to jetbrains’ tools and others, but the way vscode deals with it is, in my experience, the most stable and smooth.

let's say your problem is a bit more complex, or you want to edit a file using the command line or another tool directly inside the container. in this scenario, you need to jump inside the running container. the easiest way to do this is by running the following command in your terminal:

```bash
docker exec -it <container_name_or_id> bash
```

replace `<container_name_or_id>` with the actual name or id of your devcontainer. you can find this using `docker ps`. now you're essentially inside the container's shell. you can use regular linux commands such as `nano`, `vim` or `emacs` to edit your files. just be mindful that the changes you're making are to the files within the mounted directory which also reflects the same folder outside the container.

let me tell you a story, i remember one time, back when i was messing with docker for the first time for a side project of mine. i messed up a config file inside my container and spent like an entire day not figuring out why the app wasn’t working. i had edited it inside the container using vim and had a typo. when i finally did find the issue, i discovered that i was editing an older version of the file in the container because i didn't realize the files were mounted. boy, did i feel silly. always remember where the files are located! now it makes me think, how many times have i been staring at a problem in code for hours, only to find out it was a typo in a config file or a wrongly imported module? we all have been there, i guess.

now, regarding code examples, you might want to automate the process of making adjustments inside the container with bash scripts or other scripts. this comes in handy when you need to run commands in the container upon creation or rebuilding. you can do this by adding the following command to your `devcontainer.json`:

```json
{
    "postCreateCommand": "chmod a+x /opt/scripts/setup.sh && /opt/scripts/setup.sh",
    "mounts":[
      "source=/some/path/on/host,target=/opt/scripts,type=bind,readonly=true"
      ],
    "workspaceFolder": "/workspace",
}
```

in this example, the `postcreatecommand` property is an instruction to run the `setup.sh` script after the container is created or rebuilt. the script should be inside the mounted volume folder (in this case `/some/path/on/host`) and will be mounted inside the container in `/opt/scripts`.
you can then add the instructions inside of the bash script, for example:

```bash
#!/bin/bash
echo 'setting env variables'
export MY_VAR=my_value
echo 'installing packages'
npm install -g some-package
echo 'setting up git config'
git config user.email "my@email.com"
```

this script will make sure to install some required dependencies or set configuration values that your application needs. in this case it also changes git config and the value of an environment variable. remember that every time you rebuild the container these commands will be run.

another thing that you could do is using the `onCreateCommand`, this command is run only once after the container is built for the very first time. it has the exact same function than the previous example, but it's more useful when you want to set up some kind of database schema or a first time config. to demonstrate, here's a quick example of how you can add a different instruction:

```json
{
  "onCreateCommand": "cd /workspace && ./create_db.sh",
    "mounts":[
      "source=/some/path/on/host,target=/opt/scripts,type=bind,readonly=true"
      ],
   "workspaceFolder": "/workspace",
}
```

where the bash script will contain code to create the database, such as:

```bash
#!/bin/bash
echo 'creating db'
sqlite3 my.db < schema.sql
echo 'db created!'
```

in this case, the script is located inside of the workspace folder.

keep in mind that messing with files inside a devcontainer isn't much different from editing them on your local machine, as long as you’re dealing with mounted folders. the real difference comes with files created or installed inside the container that aren’t part of a mounted folder. these changes will be lost when the container is stopped or removed, so avoid touching those if you want changes to persist between container sessions.

when it comes to resources, instead of just dropping links, i’d suggest checking out books or papers that delve into containerization and development environments. docker's documentation itself is an excellent start. i would suggest a deep look at 'docker deep dive' by nigel poulton for example. there are many books that delve into the architecture of docker, these can help you understand better the underlying mechanics of the environment that devcontainers rely on and improve your understanding on how to deal with the problem you encountered.

the key is to always be aware of where your files actually reside. if you understand that everything is mounted and you understand the relationship between the host filesystem and the container mounted paths you'll have no problem with these. once that mental model clicks, you'll be fine. remember it's all about understanding how the mounting works and how to interact with a container's shell. after that, it's just a matter of practice and getting comfortable with your tools.
