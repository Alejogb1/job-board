---
title: "Why is a Redhat UBI8: buildah fails with Error: Unable to find a match: e2fsprogs?"
date: "2024-12-15"
id: "why-is-a-redhat-ubi8-buildah-fails-with-error-unable-to-find-a-match-e2fsprogs"
---

alright, let’s break down this redhat ubi8 and buildah `e2fsprogs` error. i’ve banged my head against similar walls more times than i care to count, so hopefully, i can save you some grief.

the core of the problem is that buildah, when it’s creating an image, needs to install packages. one of those packages, `e2fsprogs`, provides essential filesystem utilities, specifically for ext2/3/4 filesystems, which are pretty common. the ubi8 base image, being minimal, doesn’t include it by default and relies on a configured package manager, which is usually `dnf`. if buildah can't find `e2fsprogs` it's because the configured package sources aren't set up to supply it, or the package manager is not yet configured in your build instructions.

i remember this all too well when i was working on a containerized data pipeline for a previous employer a few years back. we were using buildah because it gave us better control over the image build process than docker, and i wanted that extra control. i had this elaborate build script where i was layering multiple ubi-based images, and i hit this same error with `e2fsprogs` while trying to create an application image that required it because it used ext4 filesystem. it took me a good half-day to track it down, since the error message was, shall we say, economical.

first, let’s verify that your `buildah` environment is actually seeing the `ubi8` repositories. you probably know this already, but it is good to check. it could be a simple issue where the repositories are not enabled or correctly configured, so we need to make sure that’s not a problem. the `ubi8` base image should come with a basic set of repos enabled by default, but sometimes, things happen, especially with custom build environments.

a common problem arises with the way you are attempting to configure the required repositories. often users forget to install the tooling for package management itself. make sure that you are calling a package manager like `dnf` before you try to use `dnf` to install `e2fsprogs`.

here is an example of how you could configure dnf repos and install the package with buildah:

```bash
buildah bud -f - <<EOF
FROM registry.access.redhat.com/ubi8/ubi:latest
RUN microdnf install dnf -y
RUN dnf install e2fsprogs -y
RUN echo "e2fsprogs installed" > /success.txt
EOF
```

in the above example, `microdnf` is the minimal package manager that comes with the base image. we use that to install `dnf` which then we use to install `e2fsprogs`. this may seem like an additional step, but often it is the most stable path forward to a solution, it avoids a lot of problems.

another reason, and i’ve seen this one too many times, is that the `buildah` environment is not set up to talk to the redhat repositories. sometimes, if you are behind a corporate firewall or if you are not using the redhat registry directly, the necessary certificates or registry configurations might be missing. you need to ensure your `buildah` environment has proper access to the redhat content delivery network (cdn) or where you are pulling your images. without that, `dnf`, or any other package manager, won't be able to find `e2fsprogs`.

let’s say you are using a custom registry mirror, your `dnf` configurations may be pointing to wrong repositories. make sure that `/etc/yum.repos.d` folder has the appropriate configurations and that your repositories are working and available, and are not causing a chicken-egg problem because the tools to connect to the registry are not installed yet and that is what causes this circular problem. you would need to debug this.

here is a basic example on how to debug your current situation and verify that the configurations are correct:

```bash
buildah bud -f - <<EOF
FROM registry.access.redhat.com/ubi8/ubi:latest
RUN microdnf install dnf -y
RUN dnf update -y
RUN ls -la /etc/yum.repos.d/
RUN cat /etc/yum.repos.d/*.repo
RUN dnf search e2fsprogs
EOF
```

in the above example, we are updating the available packages for dnf, showing the `/etc/yum.repos.d` folder to see if our configuration is working, then we are showing all of the repo configurations and finally we are checking if dnf can find `e2fsprogs` in its current configuration.

finally, there is the odd case where some other package that depends on `e2fsprogs` is the real source of the error and you do not know that. the solution would be exactly the same, as `dnf` would require `e2fsprogs` to install the required package.

one last piece of advice: if you are doing complex layering, make sure each layer has a very minimal set of commands. complex layers are harder to debug and if a layer fails, it is much harder to go through each command to find what is causing it. think of this process like lego blocks, every block should be very simple and do one thing.

here is an example of simple layered image to be clearer:

```bash
buildah bud -f - <<EOF
FROM registry.access.redhat.com/ubi8/ubi:latest as builder
RUN microdnf install dnf -y
RUN dnf update -y
RUN dnf install e2fsprogs -y
FROM scratch
COPY --from=builder /usr /usr
COPY --from=builder /etc /etc
COPY --from=builder /bin /bin
COPY --from=builder /sbin /sbin
COPY --from=builder /lib /lib
COPY --from=builder /lib64 /lib64
RUN ls -la /bin
EOF
```

in this example we use multi-stage builds to first create an image that has our required tooling and then we copy the files to a `scratch` image. this is a very clean way of creating small images. this approach, while a bit more work initially, it usually is a time saver in the long run as it allows for more streamlined builds. also the last layer does not need to run as root since it's already self-contained. this gives better control over what is included in the final image and allows for more secure builds.

as for resources, i would not recommend random blog posts. instead you should check out the red hat documentation, which is generally pretty solid, specifically the information on ubi images, and also the man pages for `buildah` and `dnf` and of course, the red hat container registry itself, it contains all the information and tooling needed to debug problems like this. and if you are really into deep dives, some of the coreos papers are a gold mine of background information.

and if you are really stuck, i’ve found that sometimes stepping away, having a cup of coffee, then coming back and reading the error message again can help, especially if you have been staring at the screen for too long. or you can try asking a rubber duck. they do not judge you. they just stare and you figure it out.
