---
title: "rsync skipping non-regular file command explanation?"
date: "2024-12-13"
id: "rsync-skipping-non-regular-file-command-explanation"
---

Okay so you're having an issue with rsync skipping stuff that ain't regular files right I've been there man oh boy have I been there Let me break it down for you based on my years of banging my head against the keyboard trying to get rsync to do what I want

First things first rsync by default it's all about regular files Regular files think your text docs your images your source code the typical stuff that takes up space on a drive When rsync encounters something that is NOT a regular file it basically says "Nope not touching that" and moves on That's the skipping you're seeing

Why does it do this Well mostly for safety and efficiency Imagine trying to rsync a named pipe or a device file those are special things they don't behave like regular files and trying to copy them might lead to all sorts of chaos and also it would be completely pointless since they aren't file contents in the first place Also rsync is built for moving data which is stored in regular files not pointers to system objects and other weird stuff that isn't data

Think of it like this rsync is the diligent librarian it's only interested in books the things with pages and words it's not interested in the weird decorations or the fire extinguisher sitting on the wall just because it's there rsync is a book mover not a building mover

Now let's talk about these non-regular files a bit more common ones you might encounter are

*   **Directories** rsync handles directories but treats them differently than files It creates the directory structure and then copies the files *inside* those directories If a directory is empty it just creates it on the destination
*   **Symbolic Links (symlinks)** These are like shortcuts to files or directories rsync can copy symlinks preserving the link but that's a special option It does not copy the thing it's pointing to unless you tell it to
*   **Named Pipes (FIFOs)** These are for inter-process communication rsync is going to ignore them
*   **Device Files** These represent hardware devices rsync definitely ignores these
*   **Sockets** Another inter-process communication method rsync will move on

So how can you get rsync to deal with these non-regular file types Well it depends on what you want to do and what you need for your job

Here's the thing rsync has a bunch of options that let you tweak its behavior One very important one is `-a` or `--archive` It's a good starting point for most rsync jobs because it combines a lot of settings including `-r` which enables recursion in directories

Here's the first code snippet for you

```bash
rsync -avz /source/directory /destination/directory
```

This command copies everything in `/source/directory` to `/destination/directory` recursively it also tries to preserve permissions ownership timestamps etc and it compresses the data during transfer it's your workhorse rsync command

But `-a` won't copy all the non regular files you want for that you need to explore other rsync options that allow you to copy symlinks and other special files but if you want to force rsync to copy everything and I mean everything including the kitchen sink you will need to add more options to the command

Here is the second example with `--devices` `--specials` and `--copy-links`

```bash
rsync -avz --devices --specials --copy-links /source/directory /destination/directory
```

Here we are adding `--devices` `--specials` which copies device files and special files and `--copy-links` which tells rsync to copy the links rather than skipping them This combo will copy all types of files that rsync would otherwise skip it's a more comprehensive solution although sometimes overkill

Let's say you don't want to copy those symlinks but you want to follow them instead because you want to copy the data they're pointing to rsync gives you that too

```bash
rsync -avzL /source/directory /destination/directory
```

Here we are using `-L` or `--copy-links` to follow symbolic links this way rsync copies the actual file at the end of the symbolic link chain instead of creating a symbolic link in the destination This is useful when you need to have the files at the destination instead of symbolic links to them

I remember this one time back in the day when I was migrating servers for this small startup I was working at I had this huge data directory filled with all sorts of stuff I had assumed that my regular `-avz` command would do the trick I did not test it on a small sample of data I simply went forward and ran it it was a big directory and I had to wait all night when I arrived at work in the morning the data migration was done But when I did a quick check I found that some of the database directories weren't the actual directories I had symbolic links in my data directory to other locations and rsync had not followed those it had copied the links not the actual data I was in a serious rush and didn't think straight I had not read the manual properly and it was a painful mistake to make that day It was a good lesson in rsync options and how they can have unexpected behavior if not read about properly

One thing to note though when using `--devices` and `--specials` make sure you know what you're doing Device files and pipes aren't like regular files so moving them blindly could cause all sorts of weirdness you might get some error that will look like this rsync: failed to set times on "/destination/directory/dev_device": Operation not permitted if you copy device files to places where you don't have permissions to modify permissions on those specific types of files be careful and always test

And one more thing sometimes you might be thinking why rsync is not deleting files in the destination that you have deleted in the source here's the deal rsync by default does not delete files in the destination even if they have been deleted in the source if you want rsync to do that you need to add the `--delete` option to your command This option makes rsync be a mirror so to say it removes files from the destination if they are not present in the source be careful with this one too

And remember that rsync has a million options I'm not even kidding There's a whole world of possibilities hidden in those flags So the key is to figure out exactly what you want to accomplish and read the man page to find the right combination This is where I shamelessly tell you that the rsync man page is your best friend it's verbose it can be overwhelming but it is THE book you need to consult to really understand the tool

One more suggestion if you are serious about using rsync you might also look into the original research paper by Tridgell and Paul in the paper “The rsync algorithm” they go into deep detail about the inner workings of the algorithm and other resources that you might find interesting on the topic of data migration

As a final note rsync is a powerful tool but it's not a magic bullet always read the documentation before running it on your production system especially if you're working with non regular files or anything sensitive and this should be pretty obvious but test your rsync commands on test data first It's better to mess up with test data than with important stuff

Hope this helps you out man Let me know if you have other rsync questions or questions about other tech stuff and I'll try my best to help based on my personal experiences Just don't ask me for relationship advice because I'll just recommend using rsync to clone yourself and you might not be able to get rid of them after that you will have a second you that is an exact copy of you and I'm not kidding that's my experience
