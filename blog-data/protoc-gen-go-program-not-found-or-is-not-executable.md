---
title: "protoc-gen-go program not found or is not executable?"
date: "2024-12-13"
id: "protoc-gen-go-program-not-found-or-is-not-executable"
---

 I see the question protoc-gen-go program not found or is not executable Been there done that so many times it's basically my coding origin story Feels like I've spent half my life debugging this specific issue lets unpack this I bet its probably a pathing or install problem it usually is

So here's the deal and trust me I've been in the trenches with this more than I care to admit This error the "protoc-gen-go program not found or is not executable" it screams loud and clear that the protoc compiler which handles your proto files doesnt know where to find the `protoc-gen-go` plugin This plugin is vital it's what translates your protocol buffer definition files .proto into Go code

Basically when you run the `protoc` command with the `--go_out` flag you're telling it hey use this go plugin to generate go code and if it can't find that plugin well you get this lovely error message Lets go through possible causes and how to nail down each one of them you know basic systematic debugging

First things first lets address the obvious Have you actually installed `protoc-gen-go` I know its basic but its usually where the issue hides Sometimes in your haste you might overlook the most important step I used to do this a lot when I started and spent 2 days trying to figure out why my proto files would just not compile Turns out the plugin was sitting there in my downloads folder not installed I mean come on you know that pain

So if you are not sure just try this command in your terminal to install the plugin or re-install it if you think you did it correctly before it should always get the latest version unless you specifically require some other version

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

Now lets move onto the next part the path you see when protoc runs it doesn't look in every nook and cranny of your filesystem it checks a few places determined by your OS environment variables or its configuration The most typical culprit here is your `PATH` environment variable it dictates the directories your OS will check to find executable files like `protoc-gen-go`

So if its installed you need to make sure the directory where `protoc-gen-go` lives is actually in your `PATH` For me this was painful it always felt like a wild goose chase especially in the early days You see one time I installed it but somehow missed adding my go binaries path to the PATH var and it took me some time to realize how dumb I was I learned my lesson that day you better believe it

So lets find the install location which varies depending on how you did the install or go version or platform or whatever

If you're using go tools you should be able to find it in your `$GOPATH/bin` or `$GOBIN` directory depending on your go setup this was the main issue 9 out of 10 times when I had this bug that folder was not in the PATH it's almost a rite of passage for go developers I guess

You can confirm its there with something like this

```bash
ls $GOPATH/bin/protoc-gen-go
```

or maybe this

```bash
ls $GOBIN/protoc-gen-go
```

or if you use other ways like `go install bin` probably it went to your `/usr/local/bin` or other system paths This depends on how you configured your go environment

Now the tricky part you need to add this directory to your `PATH` environment variable it varies by OS If you have zsh or bash and you have a .zshrc or .bashrc file the following command should do the trick to append it

```bash
export PATH="$PATH:$(go env GOPATH)/bin"
```

you should run this after you install it and after changing the file you might need to `source` the file so that the environment changes apply or simply reopen your terminal session

now a little side note sometimes if you are working with docker or any type of container environment the pathing can be a bit different since you have to configure the container separately to use external tools it's always a pain debugging those kinds of problems you can be almost sure something is wrong when a program runs in your environment but not on docker it's probably a path thing

A word of warning though dont just go adding random folders to your path as this can cause issues down the road I saw someone do this once and their machine became unusable

Now lets cover a very sneaky cause file permissions Sometimes the executable file `protoc-gen-go` is installed but lacks execute permissions and it should be executable to work It's like it's there but not ready to do its job this happened to me once for an entirely different program and I spent ages chasing this problem it was not fun but an important lesson in debugging process. The most probable cause was me not using `sudo` correctly or other permission issues

So lets check the permissions using this

```bash
ls -l $(which protoc-gen-go)
```

The output should show something like `-rwxr-xr-x` it means the file is executable if its something like `-rw-r-r--` well it's not then you need to use `chmod` to make it executable like this

```bash
chmod +x $(which protoc-gen-go)
```

This is a common one and it can be a real head scratcher because everything looks like its installed correctly

Now here is where the really obscure stuff happens If everything up to this point is in order it can be some subtle issues Sometimes if you have multiple versions of the go toolchain or the `protobuf` toolchain it can cause conflicts and its a mess to unravel that is why you should try to always stay on the latest version for development or at least a stable version

It can be a clash of versions where one version requires a certain version of the other tools or plugins and it's a whole problem with compatibility issues I remember one time I was using an old version of `protoc` and the latest version of `protoc-gen-go` and I had so many errors that at one point I was sure the computer itself was about to explode I wish I was kidding but that is how frustrating it was This stuff can happen to anyone even if you are as experienced as me

Always check what versions you are using of everything try to upgrade them or downgrade them to see if the issue is a version problem In my experience that always solved it it's always the version I say almost always

So to summarize:

1.  **Verify Installation:** Make absolutely sure `protoc-gen-go` is installed using the go tools
2.  **Check Your PATH:** Ensure that the directory where `protoc-gen-go` resides is in your `PATH`
3.  **Verify Permissions:** Make sure `protoc-gen-go` has execute permissions
4.  **Version Clashes:** Double check for conflicts in the toolchain version you are using if the issue persists

Now I know that was a wall of text but I hope it helps someone out there who is also suffering the protoc-gen-go blues you know sometimes all we need is someone else telling us we are not dumb and that it can happen to anyone but seriously you need to check the path you have my word on it its almost always that

As for useful resources I would avoid random blog posts most of the time you can get a lot of bad information that is not tested The official Google protobuf docs and the go protobuf repository are your friends check the README of each project it has a lot of useful information also I strongly suggest reading "Programming in Go" it has a nice section on how go toolchain works that helped me a lot back when I was beginning my journey as a go programmer

Hope this was helpful! Now go fix your issues!
