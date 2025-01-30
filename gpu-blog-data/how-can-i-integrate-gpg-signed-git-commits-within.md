---
title: "How can I integrate GPG-signed Git commits within a VSCode dev container (WSL2 Ubuntu v20)?"
date: "2025-01-30"
id: "how-can-i-integrate-gpg-signed-git-commits-within"
---
Git commit signing using GPG adds an important layer of authenticity and integrity to a project’s commit history, verifying that a commit originates from the stated author and has not been tampered with in transit. Successfully integrating this practice within a VSCode dev container environment, especially one leveraging WSL2 and Ubuntu v20, requires careful attention to several configuration points across the host OS, WSL2 instance, and the dev container itself. My experience building and maintaining CI pipelines using containers confirms the critical need for consistent and secure code contributions.

First, I’ll discuss the underlying requirements for successful GPG key management and how they interplay between the various layers of this environment. Next, I will walk through three practical examples demonstrating GPG integration in this specific context, covering the most critical areas.

GPG signing relies on a key pair: a private key used to sign commits, and a public key that others use to verify them.  The private key *must* remain secure. The challenge in this scenario lies in ensuring the dev container, which is essentially an isolated environment, has access to the necessary signing tools and the private key, and that the git configuration within the container correctly utilizes these tools. A common mistake is relying on the host OS's GPG configuration, leading to inconsistencies and potential authentication issues when working within the container.

Specifically, we face a challenge because the GPG agent, which manages the private key, traditionally resides on the host operating system (Windows in this case, via WSL2). The VSCode dev container, however, operates within the Linux subsystem, effectively as a separate operating system instance with its own user and file system. The private key must, therefore, be accessible to the container's git process to enable signing. Direct sharing of the private key is generally discouraged for security reasons. Instead, we'll need a mechanism to communicate with a running GPG agent.

Let’s break down the process with concrete examples.

**Example 1: Setting up the GPG agent and keys on the WSL2 instance**

The initial setup focuses on correctly configuring the GPG agent within the WSL2 Ubuntu environment, ensuring it's running, and import keys. This step usually happens once when setting up the development environment. I’ve observed that skipping this step can cause hard-to-trace signing failures within the dev container.
```bash
# Example 1: WSL2 Ubuntu Terminal
# Ensure GPG is installed
sudo apt update && sudo apt install gnupg

# Generate a new key or import one from the host OS if it's already present.

# (If generating, follow the prompts. Save the key id for later use.)

# Example import process from host (assuming your keys are in .gnupg on windows)
# First copy the .gnupg directory from windows to your WSL instance
# cd ~
# mkdir .gnupg && cd .gnupg
# cp /mnt/c/Users/<YourUserName>/.gnupg/* .
# then import keys
# gpg --import <Your key files here>
# Alternatively if you already have keys set up on your WSL2 instance you don't need to do the above
# Verify your key is visible
gpg --list-secret-keys --keyid-format long

# Configure gpg agent
echo "use-agent" >> ~/.gnupg/gpg.conf
echo "enable-ssh-support" >> ~/.gnupg/gpg.conf

# Start the gpg agent
gpg-connect-agent /bye
```
In this example, after installing `gnupg`, you would either create a new key pair (following the interactive prompts) or import your existing one. After the keys are set up, `gpg-connect-agent /bye` starts the agent in the background and configuration steps are taken to make it work as expected. This command avoids the need for explicitly starting `gpg-agent`.  The command `gpg --list-secret-keys --keyid-format long` will show the keys available and provides the key ID used for further setup. The output of that command will include a long string for the key id, such as `sec   rsa4096/YOUR_KEY_ID_HERE 2023-10-27 [SC]`. You will need this key id for the later steps.

**Example 2: Configuring the dev container to use the WSL2 GPG agent**

Once the keys and agent are running in WSL2, the next step is to ensure the dev container can communicate with this agent. This involves mounting the relevant sockets used by GPG. This is defined in the `.devcontainer/devcontainer.json` file for the dev container.
```json
// .devcontainer/devcontainer.json
{
"name": "GPG Signing Example",
"build": {
        "dockerfile": "Dockerfile"
},
  "mounts": [
		"source=/run/user/$(id -u)/gnupg,target=/run/user/$(id -u)/gnupg,type=bind,consistency=delegated",
      "source=/tmp/gpg-agent,target=/tmp/gpg-agent,type=bind,consistency=delegated"
	],
"containerEnv": {
		"GPG_TTY": "/dev/pts/0",
	"GPG_AGENT_INFO": "/run/user/$(id -u)/gnupg/S.gpg-agent:$GPG_TTY"

},
"postCreateCommand": "git config --global user.signingkey YOUR_KEY_ID_HERE && git config --global commit.gpgsign true && echo 'export GPG_TTY=/dev/pts/0' >> ~/.bashrc"
}
```

Here, we are using the `mounts` property in the `devcontainer.json` file to mount the GPG agent socket from the host WSL2 environment to the dev container. The `containerEnv` section sets the `GPG_TTY` and `GPG_AGENT_INFO` variables. These ensure that the container’s git process knows where to find the running agent to handle signing requests.  Finally, the `postCreateCommand` sets the key to be used for signing along with setting git config to sign commits by default, this command sets up the git environment inside the container so that the user does not need to configure that themselves on every session. Note that `YOUR_KEY_ID_HERE` should be replaced with the specific key id found in the prior step. The `echo 'export GPG_TTY=/dev/pts/0' >> ~/.bashrc` command ensures the setting persists across shell sessions. Using this configuration, once the dev container is launched, git will be able to communicate with the gpg agent. This addresses the core communication challenge.

**Example 3: Testing GPG Commit Signing within the Container**

After setting up the agent, keys, and container configuration, we need to confirm the signing works correctly by making a commit.
```bash
# Example 3: Inside the VSCode dev container
# Initialize a git repository (if not already present)
git init

# Create a file to commit
touch test_file.txt
echo "Test file" > test_file.txt

# Stage and Commit
git add .
git commit -m "Test commit with GPG signing"

# Verify that the commit is signed
git log --show-signature
```

This simple example initializes a git repository (if it does not exist), creates a test file, and commits it. Crucially, after the commit, `git log --show-signature` will display details about the commit, including whether it was signed by a GPG key. If the integration is successful, you'll see the "Good signature" output along with your key information. If not, an error is reported, and a review of the steps above is needed. If the signature is “BAD” it usually means that the key was not trusted, or that the user id in git did not match the key's user id.

These three examples, in my experience, address the most prevalent challenges encountered during GPG integration in this particular development environment.

For further study on this topic, I recommend consulting the official Git documentation on GPG signing, the VSCode documentation on dev containers and their configuration options and the GnuPG manual pages for in-depth information.  The Arch Linux wiki also offers valuable, detailed instructions on GPG agent and key configuration, which often apply to Ubuntu-based systems due to their shared roots.  These resources provide a comprehensive understanding that builds upon the specific implementation presented here, assisting in building a strong and secure development practice. A strong understanding of these concepts is vital in a multi-developer environment.
