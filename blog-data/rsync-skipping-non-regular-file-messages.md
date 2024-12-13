---
title: "rsync skipping non regular file messages?"
date: "2024-12-13"
id: "rsync-skipping-non-regular-file-messages"
---

Okay so you're seeing rsync throwing "skipping non-regular file" messages right thats a classic I've been there done that got the t-shirt probably a few times

Let me tell you what this means what's likely causing it and how you can get around it its not rocket science but its definitely something that catches folks off guard especially if you're new to rsync or dealing with complex directory structures I've spent far too many nights debugging these scenarios trust me on that

First off lets break down what rsync considers a "regular file" It's essentially your typical file a text file a document a photo a video you know the kind that stores data That's what rsync is primarily designed to sync Now things get interesting when it encounters stuff that isn't that your directories symbolic links named pipes sockets device files these aren't regular files they don't contain data in the same way a text file does these are special system objects that represent different things

When rsync hits these non-regular file types by default it kinda throws its hands up and says nope not touching that hence the "skipping non-regular file" message It's not that rsync can't *do* anything with them it just that without explicit instructions its default behavior is to ignore them for safety reasons think of it as a safety net to prevent potential damage when syncing system configuration files or devices you might not want to directly transfer across systems

Now why are you seeing this specifically well there are a bunch of possibilities

First and foremost you're probably trying to rsync a directory that contains things like directories links sockets all that good stuff and you're doing it with a very basic rsync command something like this

```bash
rsync -avz /source/dir /destination/dir
```

This command while good for most everyday syncs won't cut it here It only targets regular files and ignores others

Another reason that I've seen too often (usually when people are working on dev machines or doing some kind of configuration management) is when you're trying to copy dot files dot files can often be symbolic links to other configurations or scripts These also get caught and skipped

```bash
rsync -avz /source/dir/. /destination/dir
```
This is problematic because the `.` means "current directory" and could be containing symlinks if its user configuration it definitely would be

Sometimes its something else altogether that you overlooked maybe a developer used a named pipe for inter process communication or someone left a device node lying around that's totally possible when you work with linux long enough you will see some crazy stuff that's actually quite hard to reproduce on purpose

Okay so how to fix this the good news is that rsync is not stupid and has a boatload of options that let you deal with this Here are some of the most useful ones

1.  **Preserve symbolic links** the `-l` or `--links` option this makes rsync copy symbolic links as symbolic links instead of treating them like regular files It is very common that users will use this option to avoid creating a copy of the file and instead just linking it and they expect it to be copied as is

```bash
rsync -avzl /source/dir /destination/dir
```

   This tells rsync to preserve symlinks now if you have circular symbolic links you are going to have to be a lot more careful because rsync can get confused by it and you will end up with recursive copies

2. **Copy everything including the device nodes** the `-D` or `--devices` `--specials` options Now *this* is where you have to be very very careful if you're copying device nodes you can really mess up a system that's not prepared to receive those If your not 100% sure do not use that option you can end up copying sockets and pipes that shouldnt be copied to other systems I know some sysadmins who have used this option and ended up wiping whole servers because of this so be wary

   I'll provide the command but please only use it when needed

```bash
rsync -avzDL /source/dir /destination/dir
```

   This includes `-l` to also follow the symbolic links if you have any
   **WARNING: DO NOT COPY DEVICE FILES TO SYSTEM FOLDERS WITHOUT KNOWING WHAT YOU ARE DOING**

3.  **Preserve directories** the `-d` or `--dirs` option this makes rsync copy directories as directory objects and not just their content if you dont add this option it will only copy the contents of the directories not the empty directories themselves so it is important to understand what that option does

```bash
rsync -avzld /source/dir /destination/dir
```
    Notice how the last command includes the links and the directories you can also include the device and special files too using the `-D` flag remember to be careful with it

Now a bit of my personal experience with this issue Years ago during the glorious days of trying to automate server backups with bash scripts I ran into this hard I had a complex server with tons of symlinks device files and named pipes for various services and my initial rsync script (of course made during a late night coffee fueled coding session) didn't include the `-lD` options The first time it ran it totally skipped important parts of the system and I only found out later when trying to restore a backup and everything was broken its a classic situation I had to rewrite everything a bunch of times and made some stupid mistakes as I usually do

This brings me to a very useful point remember that doing backups is only half of the battle and you always have to make sure that the restore also works properly there is an adage in computer security that says if you are not doing restores you are not doing backups.

Also I remember once I had a client who was trying to copy their website using rsync and for some weird reason they were getting errors the client was not tech savvy so they ended up saying their website got a virus (some people are too paranoid) but what actually happened is that they were using a cheap host that had symbolic links instead of directories and their version of rsync was too old and did not support correctly the `-l` parameter so it was skipping the whole site structure, you would think these things do not happen but they do. Sometimes when you go looking for a simple solution it ends up being a lot more complex than you expect. It was a real "fun" day

Now I know you probably wanted something simple like a one-liner solution but I cannot stress enough how important it is to understand *why* these things happen and what rsync is actually doing under the hood its way better than just blindly copy-pasting commands from stackoverflow (sorry not sorry)

If you want a deep dive into how rsync works and its many parameters the rsync manual is your friend (`man rsync`) that should be a given I know I know that documentation can be boring but it is a useful document, I also suggest that you read the *Unix Network Programming Vol 1* and *Advanced Programming in the UNIX environment* these two books are considered the bibles of unix programming and will give you an amazing deep understanding of how systems work, although they do not specifically talk about rsync they teach a lot about why these situations happen in the first place.

Remember be very careful with your data especially with device files but if you do everything correctly this error will be one more of your past annoying adventures, it happens to all of us don't let it discourage you.

Let me know if you have other issues I'm always happy to talk about problems that I solved in the past It makes me feel like I did something useful with my life (just joking) Good luck with it!
