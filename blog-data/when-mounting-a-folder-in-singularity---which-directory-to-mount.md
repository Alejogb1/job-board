---
title: "When mounting a folder in singularity - which directory to mount?"
date: "2024-12-15"
id: "when-mounting-a-folder-in-singularity---which-directory-to-mount"
---

alright, so, mounting folders in singularity, it's a topic i've spent way too many late nights dealing with. let's break it down.

the core issue here is that singularity, or apptainer as it's becoming more known, is all about containerization. it's designed to isolate the environment inside the container from your host system. mounting a folder is how you poke a hole through that isolation, allowing the container to see and interact with specific parts of your host's filesystem.

now, the "which directory" part is where things can get a bit tricky and often leads to head-scratching moments. there's no single, universal best answer because it completely hinges on what you're trying to achieve. but let me give you some scenarios based on my past experiences and the gotchas i’ve encountered.

first off, let's talk about the most common mistake: mounting the *wrong* directory. i recall spending a whole friday night trying to run a data analysis pipeline inside a singularity container. i needed the container to access the massive dataset i had stored in `/mnt/data`. i figured i'd just mount `/mnt` and call it a day.

the problem there was that `/mnt` on my local machine also contained other stuff i absolutely did not want inside the container like various network storage devices and backups. it introduced a lot of complexity, and potentially, security issues. my container was now potentially vulnerable by accident. it was like bringing the whole house to a small apartment you were renting. not ideal, not ideal at all.

so, the lesson learned: mount the most specific directory that contains only what you need, no more, no less. in my case, instead of `/mnt`, it should have just been `/mnt/data`. it simplifies things and keeps the container more isolated.

another situation that came up was when working with source code. let's say i was developing some python application, and i had the source code in `~/projects/my_app`. i could mount the entire `~/projects` directory. but this would also expose my other projects in that directory. or i could only mount `~/projects/my_app`.

the best practice here is that i should create some kind of designated mount point inside my home folder, specifically for the container's purpose. for example, `/home/myusername/container_mounts/my_app`. i then symlink my actual source code there. This means my singularity command would look something like:

```bash
singularity run -B /home/myusername/container_mounts/my_app:/app my_container.sif
```

and then inside my container the code would be available as `/app`. this allows me to maintain a structured approach and not mix source code with other stuff.

ok, lets talk about tmp directories. the temptation is to mount `/tmp` or `/var/tmp` which are the usual culprits. i've done this before and it always ends up messy. you see, when you mount your system’s tmp directories, you are potentially exposing temporary files that might be created by other programs or users. not very controlled.

what i do now is to set up a specific temporary mount point, within my home directory. something like `/home/myusername/container_tmp`. this gives a controlled temporary space that i can use inside the container without affecting others. the command looks something like this:

```bash
singularity run -B /home/myusername/container_tmp:/tmp my_container.sif
```

remember, any changes made inside `/tmp` in the container will be persisted in the host directory `/home/myusername/container_tmp`.

a thing that i tend to do often is to mount configuration files. sometimes my singularity image is stateless, but it needs access to configuration files that i don't want to bake into the image itself. these config files are often in my `$HOME/.config/my_app`. in this case i usually create a subfolder inside the mounting folder that i’m using. so something like `/home/myusername/container_mounts/config`. the mounting command ends up being something like:

```bash
singularity run -B /home/myusername/container_mounts/config:/config my_container.sif
```

and then link the files i need from the `.config` into `/home/myusername/container_mounts/config`. now, inside the container the config files will be inside `/config`.

and i guess i can add that sometimes the problem with mounting can be that we don't really understand the directory structure of the image itself. some images have assumptions on where specific data resides, or where the entry point script expects the data to be. it always helps to spend some time exploring the container's filesystem *before* mounting things into it. `singularity shell` is your best friend in this process. for this case let's say my image expects data in `/mnt`. i can then mount `/home/myusername/container_mounts/data` to `/mnt` in the container.

let’s talk about mounting with read-only access. sometimes you just want the container to have a view of the data without the possibility of modifying the files in the host. that's where the `-ro` flag comes in handy. the command line changes slightly for this case like so:

```bash
singularity run -B /home/myusername/container_mounts/input_data:/data:ro my_container.sif
```

this means that the contents of the host directory `/home/myusername/container_mounts/input_data` will be accessible inside the container in `/data` but the container will be prevented to write to it. i always tell people to use this option whenever possible to prevent any accidental modification of their host’s data.

the last thing i guess i can mention is when you are building your own images, and you need specific data or libraries, you should rather include those inside the image itself. don’t go overboard mounting. mounting is ok when you are dealing with variable data like input or output directories or when you are dealing with configuration files that you don’t want to bake into the image. if it's libraries or static data, try to include them in the image's definition file itself. it leads to more reproducible and self-contained containers.

sometimes i feel i’m just moving files around between folders.

when choosing a directory to mount always consider the following principles: least privilege principle, mount as specific folder as possible and control the temporary directories. doing this will save you time and debugging headaches. there's no magic bullet here, it's just good practice and some experience with the tool.

resources that i found invaluable during my journey with singularity, and that i recommend you read would be the official singularity documentation itself (that has evolved quite a bit over the years into apptainer), and the "container security" book by liz rice which covers containers in a more general sense but has an excellent security overview of the problem itself. i've found that if you know the inner workings of how namespaces and cgroups work, everything becomes much more clear. finally, papers from the systems research community like the ones by sandboxing, and containerisation are extremely helpful to grasp the nuances of how everything works.
