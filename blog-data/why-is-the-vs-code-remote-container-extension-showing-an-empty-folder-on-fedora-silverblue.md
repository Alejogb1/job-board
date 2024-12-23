---
title: "Why is the VS Code remote container extension showing an empty folder on Fedora Silverblue?"
date: "2024-12-23"
id: "why-is-the-vs-code-remote-container-extension-showing-an-empty-folder-on-fedora-silverblue"
---

, let's tackle this particular head-scratcher. It's happened to me before, actually, back when I was setting up a development environment for a rather intricate distributed system on a Fedora Silverblue box. The issue of an empty folder when using the VS Code remote container extension is definitely not a fun one, especially given the promises of reproducibility that Silverblue and containers are meant to provide. It's almost like a glitch in the matrix, so let’s break down what’s likely going on and how we can resolve it.

Essentially, the problem stems from how VS Code's remote container extension and Fedora Silverblue interact, specifically with regards to file system mounts and permissions within the container. Silverblue, being an immutable operating system, differs fundamentally from traditional Linux distributions. This immutability is great for system stability but introduces some interesting caveats regarding how containers access your host system's files.

When you instruct VS Code to connect to a remote container, it utilizes Docker or Podman (depending on your setup). It attempts to mount the directory containing your source code into the container, so the code inside the container can function within your project’s context. In a standard Linux system, this is straightforward. However, Silverblue uses a layered file system managed by rpm-ostree. This means the traditional user home directory, `/home/<your_username>`, is not actually the base of the file system your container sees directly. It’s rather a symbolic overlay.

The most common cause of the "empty folder" is that the container either lacks permission to access the host file system where your project directory exists, or the container is looking for the directory at the base of the host file system, which doesn’t typically include the path to your user's home directory. Silverblue, in its quest for maximum security, tightens up file access. This can easily break the standard container mounting mechanism. The mount point may exist, but it might be inaccessible, causing VS Code to interpret it as an empty directory.

Here’s a dive into a few ways we can fix this, accompanied by some working code snippets.

**Solution 1: Specifying the Correct Mount Point**

Often the container does see *some* files, but not *your* files. This usually points to the mount directory being wrong. We need to ensure the container mounts the correct folder from the host. To illustrate this, let's assume your project is in `~/dev/myproject`. In your `.devcontainer/devcontainer.json` file, ensure you're setting the `workspaceFolder` and `mounts` correctly. A common mistake is to assume that `workspaceFolder` is set to the correct host file path. It is an *internal* path, but also needs to correspond to the mount configuration. The snippet below shows how the mount path should explicitly be set to the actual full path of the folder on the host, and we make sure that `workspaceFolder` inside the container is set correctly, too:

```json
{
    "name": "My Dev Container",
    "build": {
        "dockerfile": "Dockerfile"
    },
    "workspaceFolder": "/workspace",
    "mounts": [
        "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
    ],
    "remoteUser": "vscode"
}
```

In the above configuration:

*   `workspaceFolder`: defines the directory within the container where our project is mounted.
*   `mounts`: specifies which host directory to map to what path within the container. The important part here is `source=${localWorkspaceFolder}`. This tells VS Code to mount the local folder the user has opened in VS Code and map it to `/workspace` in the container. `type=bind` ensures that the changes are instantly visible, and `consistency=cached` helps with performance. Notice that I do not assume the user's home path, but let VS Code use its own variables to refer to the local folder. This is much better than hardcoding.

**Solution 2: Handling Permissions**

Sometimes, it is not about the location of your mount point, but rather about access permissions to it. The issue is that the user running the container process may not have access to the folder you intend to mount. This is especially true if you create new folders on Silverblue, outside of your standard `~/` directory structure (which is usually fine since that folder is already set up with the right permissions). You might try a configuration like the following in your `devcontainer.json`, which uses the `runArgs` option:

```json
{
  "name": "My Dev Container with user",
  "build": {
      "dockerfile": "Dockerfile"
  },
  "workspaceFolder": "/workspace",
    "mounts": [
      "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
    ],
    "runArgs": [
        "--user",
        "${localEnv:USER}"
    ],
    "remoteUser": "vscode"
}
```

In the example above, the `runArgs` field includes `--user ${localEnv:USER}`. This tells Docker or Podman to run the container as the same user who initiated the container creation, with `USER` being an environment variable. In this case, VS Code will resolve the environment variable from the context of your local machine, which will, in effect, tell the container to run as the same user who initiated the VS Code session. This will provide the user running inside the container with the same access rights as your local machine’s user. `remoteUser: vscode` ensures that all code that VS Code itself executes inside the container, is done using the vscode user.

**Solution 3: Using `podman` instead of `docker`**

Often it is not a permission or pathing problem, but the fact that `docker` itself does not have enough permission to access the folder you are trying to mount. Silverblue has strong focus on using `podman` as it is much more secure. If you are using Docker, consider using `podman` instead. It does not require a daemon to be run as root, and usually respects the user's permissions. The first step is to simply uninstall `docker` and install `podman`:

```bash
sudo rpm-ostree uninstall docker
sudo rpm-ostree install podman
```

Once you have `podman` installed, make sure that the VS Code setting "Dev > Containers: Docker Path" in VS Code is pointed to `/usr/bin/podman` and also that `dev.containers.dockerPath` setting is set to the same value in your settings.json file.

```json
{
"dev.containers.dockerPath": "/usr/bin/podman"
}
```

**Going Deeper: Resources**

For a more in-depth understanding of the intricacies involved here, I’d recommend looking into the following resources:

1.  **"Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati:** This book is a deep dive into the Linux kernel, which is beneficial to understand how file systems, mounts, and permissions operate at the core.
2.  **"Docker Deep Dive" by Nigel Poulton:** This book is an excellent guide to Docker, explaining its internals and the implications of using containers, including details on how they interact with the host file system and user permissions.
3.  **The official Fedora Silverblue documentation:** The documentation provides crucial insights into the file system layout and the unique challenges and benefits of working with an immutable operating system. It details the rpm-ostree system in depth.
4.  **The VS Code Remote Containers documentation:** This resource is indispensable for understanding how VS Code interacts with containers, including mounting options, user settings, and configuration parameters.

**Practical Takeaways**

My experience has shown that the empty folder problem on Silverblue with VS Code is most commonly related to incorrect mount paths or insufficient permissions. Specifying explicit mount paths and user contexts, as demonstrated in the code examples above, usually resolves the issue. Furthermore, switching to `podman` can bypass some security issues related to `docker`. Remember, the immutability of Silverblue is designed for system security, but this can sometimes lead to unexpected behavior when interacting with technologies designed for more traditional operating systems. These approaches are all based on real debugging experiences I've had, and it's crucial to remember that the devil is always in the details. Start with these configurations and refine as needed to get your dev environment up and running. Don't get discouraged, it’s usually just a minor tweak.
