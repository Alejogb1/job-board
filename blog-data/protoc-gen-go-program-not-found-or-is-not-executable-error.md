---
title: "protoc-gen-go program not found or is not executable error?"
date: "2024-12-13"
id: "protoc-gen-go-program-not-found-or-is-not-executable-error"
---

Okay so you're hitting that classic "protoc-gen-go not found" error right I feel your pain I've stared at that screen more times than I care to admit This isn't some obscure problem its practically a right of passage for anyone wrestling with protobufs and Go

Alright lets break this down from the ground up and I'm gonna sprinkle in some of my past battles with this monster so you know you're not alone First off you need to understand whats actually happening here `protoc-gen-go` isn't some magical incantation it's a plugin a specific program that the `protoc` compiler which is the main protocol buffer compiler calls to actually generate Go code from your `.proto` files Without it `protoc` is like a car without an engine it can't go anywhere

I remember once back when I was a junior dev I spent a whole afternoon debugging a CI/CD pipeline only to realize I had completely overlooked this step I was so focused on the gRPC side I didn't even think about how the code gets generated in the first place facepalm moment for sure

So the error you see "protoc-gen-go program not found or is not executable" is exactly that it means the `protoc` compiler can't find the executable for the Go plugin or the file it finds isn't in an executable form. It’s a binary basically. Now lets get our hands dirty.

The first most likely cause is that you simply havent installed the plugin yet. Yes I know it sounds basic but we all been there forgetting to install something essential. The way to do it is usually using `go install` which downloads and installs it. This usually works well.

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```
That line should install the latest stable version of `protoc-gen-go` if you've installed Go before you should already know that this usually installs the binaries to `$GOPATH/bin` or `$GOBIN` depending how you configured your Go installation. Now if this alone solves your issue congratulations its time for a beer you earned it.

But wait. This is where it starts getting tricky. The `protoc` compiler needs to know where to find that plugin you just installed. It does this by searching the directories listed in your `$PATH` environment variable. Now here is a little life pro-tip if you're working with more than one different tool and more than one project it's useful to start creating specific env files so you don't get confused with variables that should only be specific to an environment. This leads us to the next step to add the installation directory to the PATH if its not there.

Let's confirm that the `protoc-gen-go` binary is actually located where we expect to find it. Open a terminal and write this command:

```bash
which protoc-gen-go
```

If that gives you an output like `/home/<user>/go/bin/protoc-gen-go` well great we got the path where the binary is if it gives nothing that means that its still not in the path so we need to set it up. if it's there we can just proceed with making sure that the PATH has the necessary path in the environment. You can set the `$PATH` variable using `export` but you may also want to change the configuration files like `.bashrc` or `.zshrc` to permanently add the path for example:

```bash
export PATH="$PATH:/home/<user>/go/bin"
```

Make sure you replace `/home/<user>/go/bin` with the actual location where `protoc-gen-go` was installed or any path the `which` command gave you. After this restart your terminal because the `export` command only affects the current session. If you want to add it to a configuration file you can either use your favorite text editor to add `export PATH="$PATH:/home/<user>/go/bin"` to the end of the `.bashrc` or `.zshrc` and save the file.

Now here comes the other gotcha. Sometimes even if `protoc-gen-go` is in your `$PATH` it might still throw this error. The reason? File permissions. If the `protoc-gen-go` executable isn't marked as executable, `protoc` won't be able to use it. You can fix this with:

```bash
chmod +x /home/<user>/go/bin/protoc-gen-go
```
And again replace the path `/home/<user>/go/bin/protoc-gen-go` with the path you got from the `which` command. It basically grants execution permission to the file and solves that.

Back in my early days I actually managed to download the source code of `protoc-gen-go` manually and compile it. I did everything manually but I actually did not install it in the correct path and because I was a noob I actually did not realize that I needed to add it to the `$PATH` that was a rough day.

Also make sure that you actually installed both `protoc` compiler and `protoc-gen-go` with versions that are compatible. Sometimes having different versions of the tool and the plugin will give you a very unexpected error. In general try to use always the latest version of everything to make sure you dont have any issue with compatibility although sometimes a specific legacy project might require to have an older version of some tool. If this is the case then the best course of action is always use a docker image and use the versions of the tools you want in that specific container. That is a best practice.

It is worth mentioning that the protobuf github repo is a goldmine of information you should read the [official documentation](https://github.com/protocolbuffers/protobuf).

Now for the resources if you are doing anything with protocol buffers you need to familiarize with these concepts. I would strongly recommend reading "Programming Google Protocol Buffers" by Nanik Taneja. It's a solid book and explains all of these nuances in detail. You should also look at the Google's official documentation the one I linked before. I also highly recommend to try out all the tutorials in their official site.

So in short check your installation with `go install` check your `$PATH` variable with `echo $PATH` and `which protoc-gen-go` make sure `protoc-gen-go` is executable with `chmod +x` and make sure you have compatible versions and you should be set. If that does not work then there might be some other issue with the setup that is not specific to `protoc-gen-go` this kind of issue is the reason I have a love-hate relationship with technology. Its annoying but you always learn something new. Also when you deal with complex systems always be sure to read the logs carefully. If you are using makefiles or other kind of build system you also might want to debug these systems as well as sometimes the error can come from there. Also a little advice try to build a minimal reproducible example this will help others and yourself to understand the issue more clearly and be able to better debug it if you still have issues. I hope this helps.
