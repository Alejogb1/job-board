---
title: "With GitHub Actions, why is a .git folder gone when the job is running as a container?"
date: "2024-12-15"
id: "with-github-actions-why-is-a-git-folder-gone-when-the-job-is-running-as-a-container"
---

alright, let's tackle this git folder disappearance act within github actions containers. i've bumped into this particular situation more than once, and it always comes down to a fundamental understanding of how containers and git repos interact in the actions runner environment.

it seems like the user is expecting the `.git` folder, with all its history, to be readily available inside a container launched by github actions, but finds that it’s missing. that’s a common point of confusion, and it stems from the way actions executes your workflows in a containerized setting.

the core issue is that github actions doesn't just directly copy your entire repository into a container as is. when a job runs within a container, that container usually starts from a base image. this base image contains the operating system and other software, it does not contain any of the context of your git repository including the `.git` folder. then, github actions strategically mounts the necessary parts of your repository into the container, so your actions can access them.

and this mounting process is where the `.git` folder gets left behind. for performance reasons (mainly avoiding mounting huge repos), only the working directory contents are commonly mounted into the container by default, not the history contained within `.git`. this implies all the files and folders you see in the github repository’s main page or explorer, but excluding the .git folder.

the default behavior is, in most cases, beneficial: it prevents mounting tons of repository history that is usually unnecessary for most actions within a build job. you don't usually need to browse through the whole git history just to build or test your code. but this does mean that if your action needs access to the git history, like say, to grab the most recent commit hash or some information regarding a branch, you’ll need to take extra measures.

now, i've been dealing with this type of situation for quite some time. there was this one time, back when i was working on a release automation for a massive multi-module project, where i was trying to use a custom script that was relying on the git history to figure out what modules actually had changes to create changelogs. i was pulling my hair out wondering why it couldn't find the git history. i spent a lot of time debugging and realized the container didn't have the `.git` folder. that’s when i had to really dive into understanding how github actions does its mounts.

so, let's get technical, what is the solution?

the primary method to get around this limitation is using the `actions/checkout` action with specific configuration. this action will fetch your repository content including the .git directory, to a location in the container. you generally will want to use `fetch-depth: 0` which will fetch the full repo history.

example:

```yaml
- name: checkout repository with full history
  uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

this snippet, placed as a step in your github actions workflow before any actions that need access to the .git folder, will ensure that the entire repository is available within your container. notice the `fetch-depth: 0`. if you don’t set this option, by default it fetches a *shallow* clone of the repository, meaning it will only grab the most recent commit and not the full history.

now, if you are still having issues, it could be another more nuanced situation. let’s say you’re using a custom docker image with your actions workflow. you need to verify how this custom image is defined. sometimes the dockerfile for these custom images is set to create a new volume in the container rather than mounting your git repository’s working directory. thus the .git history will not be there.

if this is the case, you’ll have to modify your docker image configuration to allow github actions to mount your git repository directory. the main thing is understanding your container environment is not a mirror of your local environment, especially when it comes to git repositories and history.

here is a simple dockerfile example of an image that will not have the history:

```dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y git
WORKDIR /app
COPY . .
CMD ["bash"]
```

in the above dockerfile we are copying the files into the container `COPY . .`. this will not create a mount, it will create a copy, thus the .git is never included.

and to further illustrate let’s look at another snippet of a github action workflow. suppose you want to access git commit messages:

```yaml
- name: access git commit messages
  run: |
    git log --pretty=format:"%s" -n 5
```

now if you didn't use the checkout action with the correct fetch depth prior to the above command, the output will be blank, because git has nothing to work with. or more likely it will fail.

so, you have to always make sure you `checkout` using `actions/checkout` prior to using git commands in your workflow. and that's the main thing really to remember here. that github actions does not give you a full git repo by default, you need to request it.

and if you think about it, it makes sense from a design perspective, imagine trying to mount large git repositories for every workflow run, it would be inefficient and wasteful and would make your actions run slower and cost more.

now you might wonder if there are other ways to access the history, like using the github api. while this is possible it's usually more complicated and is rarely necessary. it’s better to just use the `actions/checkout` action, because it was made for this purpose. it handles all the underlying complexities and it’s the most straightforward and reliable solution. remember that github actions is heavily optimized to minimize network traffic, so it is better to fetch locally than request git history via the github api.

one more thing. if you are using submodules, be sure to add `submodules: true` to your checkout action, otherwise the contents of those submodules will not be checked out, this usually leads to the infamous `pathspec <submodule path> did not match any file(s) known to git`.

so if your github action is having issues with submodules and is complaining that it cannot find files in the submodule you might want to do this:

```yaml
- name: checkout repository with full history and submodules
  uses: actions/checkout@v4
  with:
    fetch-depth: 0
    submodules: true
```

for further understanding on git concepts, i’d recommend "pro git" a free online book. for the github actions deep-dives i recommend the official github actions documentation and their blog. also, "effective devops" provides great insights into how to effectively manage and automate your devops practices, which include understanding the underlying technologies.

it's also good to experiment, try a simple github action workflow where you just checkout and list the contents of your workspace and try to see what's there. this will give you a practical understanding of how the actions workflow environment is organized. and try to build it up step by step, instead of trying to understand everything at once.

finally, a little technical joke: why was the git repository so good at solving problems? because it always had the commit-ment to find the solution, hahaha…

i hope this helps with your understanding of why your git history seems to be "disappearing" in your github actions workflows and gives you enough context to go solve it.
