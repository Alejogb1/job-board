---
title: "How to trigger `inotifywait` action when starting waiting for changes?"
date: "2024-12-14"
id: "how-to-trigger-inotifywait-action-when-starting-waiting-for-changes"
---

so, you're asking how to get `inotifywait` to actually *do* something immediately when you start watching, instead of just waiting for the next change? yeah, i've been there, that's a classic head-scratcher. it feels like you're staring at a blank screen hoping it magically fills in, but it just sits there, patient as a rock. i've spent a good amount of time banging my head against this particular wall, so let me share what i’ve learned over the years.

basically, `inotifywait` is designed to notify you of file system *changes*. it doesn't automatically trigger on the current state when you start the process. this is, fundamentally, the way it was made and expected to work. it's not a bug, it's a feature, but it can feel like a pain if you’re looking for a particular kind of behaviour, especially when you need to act on existing files or directories.

my first experience with this was back in the early 2010s, when i was setting up a simple automated build system on a linux server. the idea was to have inotifywatch trigger a script when source files were updated. however, i wanted to also process the existing ones during the first run, not just the modifications afterwards. this is exactly the problem you're experiencing, and i remember the frustration. the initial implementation only worked after the first manual change of the watched files, which made the whole automation kinda pointless. i quickly realised `inotifywait` is not what you would call "an all-in-one watch and process" tool and i needed to handle this initial state more explicitly.

the core of the issue here is that `inotifywait` is an event-driven tool. it watches the events that the kernel exposes via the inotify api. and until there's an event it'll just block and wait.

so, here’s the trick: you need to create the initial events yourself. the simplest solution that i found that works across systems is to use a `find` command combined with `touch`. the key idea here is to 'touch' every file and directory you are about to watch. ‘touch’ is often used for changing file timestamps but also triggers the modified event, which inotifywait will pick up.

here's how this looks in code, assuming you are working with a folder called `watched_folder`:

```bash
find watched_folder -print0 | while IFS= read -r -d $'\0' file; do
  touch "$file"
done

inotifywait -m -r -e modify watched_folder |
while IFS= read -r event; do
  echo "$event"
  # put your processing code here
done

```

let's dissect that code block.

the `find watched_folder -print0` part lists all files and directories within `watched_folder`, and crucially, separates them with null characters (`\0`), which is safe for files with spaces, newlines or special characters in their names, preventing weird surprises.

the loop `while IFS= read -r -d $'\0' file; do touch "$file"; done` then reads this list, one by one, and applies the `touch` command. this updates the modification time, which triggers an inotify event. the `-r` flag prevents escape interpretation, and the `-d $'\0'` ensures correct handling of file names containing spaces.

after that, `inotifywait -m -r -e modify watched_folder` starts watching recursively for modification events within the folder, and the loop after the pipe extracts the event output one by one.

the next example shows how to trigger events only on directories when a directory is created or deleted:

```bash
find watched_folder -type d -print0 | while IFS= read -r -d $'\0' dir; do
  touch "$dir"
done
inotifywait -m -r -e create,delete watched_folder |
while IFS= read -r event; do
  echo "$event"
  # add your processing logic here
done
```

this one is pretty similar. the important part is that the find command searches only for directories `find watched_folder -type d -print0` and the events we are waiting for are `create` and `delete`, meaning inotifywait is going to fire on creating a new directory or removing one. this is handy if your processing only needs directory-level updates, for example when indexing and caching folders.

one thing i learned the hard way over the years is you need to be very specific about what events you listen to. for example, sometimes, you want to trigger only when content is actually changed, and a simple `touch` may be not enough. another case i faced was trying to trigger on a file being completely replaced or created from scratch. to detect this behaviour `modify` may not be enough and you need to use other events like `move` and `create`.

here is another example showing this with `move` and `create` as the watched events:

```bash
inotifywait -m -r -e create,moved_to watched_folder |
while IFS= read -r event; do
    echo "$event"
    # process file move or create events
done

```

this listens for `create` and `moved_to` events, this means, this will trigger only on new files created or files moved to the folder. in this case the ‘touch’ isn't needed because we want to detect the presence of new files or moved ones, not just modifications.

when dealing with more complex scenarios, and i've definitely dealt with many, you need to be aware of several subtle problems. sometimes the ‘touch’ command could not trigger changes on some particular types of file systems that have very fine-grained timestamps. this is not very common but it’s something to keep in mind.

also, be mindful of performance. if your watched directory has thousands of files and directories, the initial `find` and `touch` operation may take some time. it's usually fine, but in some corner cases it may cause your system to slow down, or use more resources during that first touch loop, for these cases you can add extra filters to the find command. i recall an old server that nearly melted because i tried touching the whole root folder instead of the folder it was expected to watch… it was a long night fixing it.

for further reading, i'd recommend checking out the manual page for `inotifywait` (`man inotifywait` in your terminal). it has a lot of hidden gems you may not be aware. the linux kernel documentation also explains the inotify subsystem in deep detail, although its a complex topic. a good book would be "advanced programming in the unix environment" by w. richard stevens and stephen a. rago, even if it is an old book, it covers the principles of operating systems that are still valid, and covers also the concepts behind the linux kernel event handling. if you have access to a university library, check it out.

so, that’s pretty much it. hopefully, this will help you with your `inotifywait` adventures. happy coding. and as a bonus, did you hear about the programmer who quit his job? he didn't get arrays. (a little programming joke to end).
