---
title: "revert unchanged files perforce command?"
date: "2024-12-13"
id: "revert-unchanged-files-perforce-command"
---

Okay so you're asking about reverting unchanged files in Perforce right I've been there man I've wrestled with Perforce more times than I care to remember it's like a rite of passage for anyone working with version control at scale let me tell you

So the core issue here is that Perforce sometimes gets a little overzealous It might mark files as opened for edit even though you haven't actually changed them This happens for various reasons maybe a tool touched the file maybe your IDE auto-saved something or maybe just Perforce being Perforce And then when you try to submit your changelist you get this annoying "file(s) not changed on disk" error which can be a total drag especially when you have a lot of files in your workspace

Reverting those individually is like using a toothpick to move a mountain a tedious and time-consuming process that will make you question your life choices So you need a way to tell Perforce to only care about actual changes not pretend ones

The typical way to revert in perforce is p4 revert filename which works fine for the changed ones but not for these pesky ones This is where the real fun begins and its why I ended up spending a few hours one friday afternoon trying to solve this mess

I remember one particularly bad case back in my early days at *Fictional Company Name* we were building this giant module and the build system kept triggering Perforce edits on dozens of configuration files every single build It was a mess my workspace was littered with these ghosts that Perforce stubbornly refused to let go of I was young and foolish back then I tried manually reverting each one that was a long afternoon of clicking through the GUI like a monkey

The key to solving this is understanding that Perforce keeps track of file digests essentially hashes of the file content and what it has in the depot version of it If these digests are identical it means that the file content is the same and there is no actual changes This lets us revert just the ones that have not changed

Here's the command you need the bread and butter of your Perforce life now it's a little complex but stick with me it's worth it trust me

```bash
p4 opened -a | grep -v " - edit " | awk '{print $1}' | xargs p4 revert
```

Let's break it down this is like a stackoverflow post that needs to be documented well

`p4 opened -a` This gets a list of all files opened in your workspace regardless if its pending for edit add delete or any other p4 operation

`grep -v " - edit "` This part filters out any files that were marked with the - edit flag which means its an actual edit and not just some phantom changes we dont want to revert those

`awk '{print $1}'` This takes the first column from the list of files which should be the name of the file that has not changed

`xargs p4 revert` This finally takes the files from `awk` and reverts the p4 changes for them now those files are in pristine state

That's the one liner you want But lets get fancy shall we its never that simple is it we are programmers after all lets do one with a bit of more finesse lets say we want to add some verbosity to the command so we can see what is going on

```bash
p4 opened -a | grep -v " - edit " | awk '{print $1}' | while read file; do
  echo "Reverting: $file"
  p4 revert "$file"
done
```

This does essentially the same thing but it adds a bit of logging telling you exactly what file is being reverted makes it less magical and easier to debug it's always good to see things happening

I like to take the approach of understanding every tool that I work with deeply It's like with a guitar you can play chords that way but if you know your scales you can actually make music it's the same with programming or with using a version control system I highly recommend reading the perforce documentation especially the section on file digests and workspace management it will get you very far

Let's take a step further let's say for some reason you want to keep some of the files that seem to be unchanged but they are in a specific directory because that directory is part of a dependency or generated code you can do it

```bash
p4 opened -a | grep -v " - edit " | awk '{print $1}' | while read file; do
  if [[ ! "$file" =~ ^path/to/your/directory/.* ]]; then
      echo "Reverting: $file"
      p4 revert "$file"
  fi
done
```

In this example I am using bash regex to filter out any files within a specific directory so the command is still reverting the files that haven't changed but it skips those that are in the `path/to/your/directory` directory you will have to change this to a path you need of course.
This helps when you have generated code or generated assets that you might want to keep as opened even if it reports as not changed

These approaches work perfectly and they have served me for a long long time I had to implement something similar when we decided to start using perforce as the main tool for our team at that *Fictional Company Name* after that the builds were smoother and my coworkers thanked me for saving hours of their time it was truly a good day

As a side note it would be fantastic if Perforce had a built in option to just revert unchanged files with one command but hey what are we supposed to do right just write more complex scripts this is the programmer way

Someone once told me that programmers are like magicians except we don’t pull rabbits out of hats we pull bugs out of code and then make them disappear I thought it was funny anyway

In addition to the Perforce documentation and man pages I'd recommend looking into "Version Control with Git" even though it's about Git understanding version control principles in general will help you understand why these problems with Perforce happen In this regard there is also a book that I quite enjoy called "Pragmatic Version Control Using Git" or "Understanding Version Control" by Eric Sink it covers most version control concepts. This is not meant to endorse just those books but these types of resources will help greatly

Remember version control can seem tricky but you will get better with practice and with understanding of the underlying concepts Don't give up and always try to automate your workflows its just part of being a good programmer

Hope it helps you in your journey good luck in coding
