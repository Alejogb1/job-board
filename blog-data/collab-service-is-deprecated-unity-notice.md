---
title: "collab service is deprecated unity notice?"
date: "2024-12-13"
id: "collab-service-is-deprecated-unity-notice"
---

 so you're getting that "collab service is deprecated unity" notice right Been there done that got the t-shirt or rather the git commit history to prove it. Let me tell you this wasn't always the case with unity collab it was actually kinda useful back in the day like when i started using it around unity 2017 or so before they started shifting gears towards unity version control. Honestly at the time it was a lifesaver especially when I was working on solo game jams. I remember specifically that one time it was a 48-hour game jam and three people were using the same collab project and none of us really knew how version control worked yet besides the basic idea. Think about it three people simultaneously working on scenes and scripts some of us probably never even pushed commits and the horror that this caused. It's a wonder we even finished a game back then a pretty bad one at that i might add. The point is that collab even with all its faults was at least available and integrated into the editor.

Now Unity wants you to move on to Unity Version Control or basically their wrapper around Plastic SCM. I get why they're doing it it’s not bad when you get it working but it’s definitely a migration pain. I've had to help a bunch of folks move their projects over so trust me I've seen it all. That "collab service is deprecated" notice is basically Unity's way of saying "hey we're done with this old thing please move on". The notice itself doesn't tell you much it just flags it as a big no-no. You'll probably see the deprecation notification in the editor console window maybe also when you try to use the collab tab in the editor. It's a polite version of a flashing red light saying "stop".

Here is the deal with the collab it was actually built on top of something called Perforce Helix Core or at least that's how it looked under the hood. I messed around a bit with the Perforce CLI before i even touched collab just to see how it all worked. Collab was kind of like a managed version of that. But Unity's new Unity Version Control system seems to be trying to compete directly with established platforms like GitHub and GitLab. They have been promoting it for some time now and its not a bad platform i would admit but the transition itself can be cumbersome. They're basically pushing to have their own version control solution fully integrated into the Unity ecosystem and dropping collab like a hot potato that nobody actually wants.

So what do you do? Well first of all don’t panic. You're not losing your project you're just losing a way to manage it. Second it is highly advised that you actually read the migration guidelines Unity provides. I know most of us (me included) skip the manuals but for something like this reading is your friend. It's mostly about changing the version control settings in your project and setting up Unity Version Control. It will also involve creating a new repository. It seems very straightforward and maybe that is for smaller projects but the issues start to surface once the project becomes large. If you have multiple people on your team and if each individual has different git preferences or if you have multiple branches then it gets even more complicated and you would need to consider the whole branch structure again. Speaking of that here's a little snippet showing how to handle a single branch setup in git using basic commands assuming you have git already installed on your machine of course

```bash
# Initialize git
git init

# add project files to the working directory
git add .

# commit the first version of files
git commit -m "Initial commit"

# add a remote url
git remote add origin <your_remote_repo_url>

# push the main branch to the remote repository
git push -u origin main
```

That's a simplified version for new projects. For existing projects with collab history you'll likely have to do some more complicated git operations. It's also probably a good idea to learn a thing or two about git branching strategies its something that could save you a lot of headache. There's even that gitflow model that's somewhat popular you might want to look at it. I mean the whole idea of git branching is to make sure that you don't ruin the main branch by integrating half baked features or some buggy code. So i guess that makes git the adult version of collabs.

Now for some issues you will find when migrating from the old collab system. First issue is the actual migration of course. Sometimes if your project is large and the collab history is too big you can actually experience errors or unexpected behaviors. That's one of the primary reasons why i moved to a different version control system early on. I mean a single Unity project can easily grow to a couple of GBs in size with all the assets especially if you're using a lot of textures and audio. So if you want to track all that in the version control system then you should be aware of the sizes and the performance. Second issue is the "learning curve" which always comes with new tools. I mean some devs are still using svn in 2024 which is crazy to me.

The third issue and something i personally encountered is related to file locking. It might sound weird but the Unity Version Control uses file locking mechanism which is something you might not expect from a tool. But it makes sense if you think of the way Unity works. For instance when a developer is working on a scene other developers can't modify it. But that might mean that people are blocked unnecessarily. So always check if your files are locked before you get a surprise when pushing to a remote repo. Also if the Unity version control plugin in the editor glitches and does not release file locks you can always use the CLI tool to remove them. Here is an example showing how to unlock files using the CLI

```bash
# connect to the workspace
cm workspace cmworkspace
# Get the list of the workspace files
cm status
# Unlock a specific file
cm unco <filename>
# Unlocking the workspace
cm unlock --all
```

It seems easy to handle in the CLI i know but it can become a hassle if you're not comfortable with command line interfaces. That reminds me once i was debugging a weird file locking bug for a week and it was simply a problem with a plugin not working correctly and locking all files. I spent so much time going back and forth with Unity support that i could have learned a new programming language instead. I guess that's how it goes when using new tools.

Speaking of tools there is that new integrated version control window in Unity which is but i honestly prefer to use the command line tools and some visual git management software. It’s just faster for me and it allows for better control. You know sometimes using only the visual tool makes you think that you can only do what is provided through the graphical interface but that's definitely not the case. You can do so much more with the command line and it's also a lot faster and efficient. But if you're new to git then of course visual tools are the way to go. There are good ones like SourceTree and GitKraken that can help you visualize your commits and branches. I think there is also a github desktop app if you prefer that one. And honestly it doesn't matter which one you use as long as you know what you're doing. And no i am not going to recommend any specific one this isn't a sponsored post.

Now for a more techie bit here is how to pull the latest changes from the main branch using a CLI tool

```bash
# Fetch the remote repository and branches
git fetch

# Change to main branch
git checkout main

# Pull latest changes
git pull
```

That’s basic stuff. You'll be doing those commands a lot. And you better know the differences between fetch pull and push. But the point is that this whole Unity version control switch thing is not about a better way to work it's more about Unity trying to build their own ecosystem. I understand they want to control everything but for some of us it's just more annoying than helpful. I mean there are plenty of other good git providers out there so it’s not as if you were forced to use their solution. But anyway the collab service is deprecated so that's the end of it.

If you are asking for resources I would recommend reading the "Pro Git" book you can find it online for free. Also the official git documentation is really helpful. But honestly the best way to learn git is to just use it and make mistakes. You'll eventually get the hang of it and realize that its not that complicated once you internalize its branching model. And if you ever get stuck try looking on Stack Overflow there is literally everything there. Just make sure you're asking good questions because if you're asking bad ones people will not be that nice and you'll get downvoted to oblivion. We can be a bit harsh here I admit. But overall we're helpful when we're not annoyed. I hope this helps you move on from Unity Collab to the new system. It's a bit of a hassle i know but you'll get through it. Good luck and have fun debugging. I'm out.
