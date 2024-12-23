---
title: "How can I configure containerd in an offline kubespray environment using an older version?"
date: "2024-12-23"
id: "how-can-i-configure-containerd-in-an-offline-kubespray-environment-using-an-older-version"
---

Okay, let's tackle this. It's a situation I've encountered more than once, particularly in those tightly controlled or air-gapped environments where pulling directly from public repositories just isn't an option. Setting up containerd within an offline kubespray deployment, especially when using an older version, requires careful planning and a bit of manual wrangling. It's not as straightforward as a standard online install, but it's definitely manageable. Let me walk you through the steps and some of the challenges I've faced.

The core issue here revolves around dependency management and ensuring that all the necessary components of containerd, along with its container images, are available locally. Kubespray, in its default configuration, expects to fetch these artifacts from remote sources. We need to circumvent this by pre-populating the required files and adjusting the kubespray configuration accordingly.

First, let's talk about acquiring the necessary components. For containerd, this typically means the containerd binary itself, configuration files (like `config.toml`), and the necessary container images used by kubernetes, particularly pause images and possibly other foundational images used by the container runtime interface (cri). For an older version, finding these directly might require some additional research. Older releases of containerd can usually be found on their official GitHub release pages. For example, if you were aiming for, say, containerd version 1.4.4, you would go to the containerd github release and download the appropriate binary. These releases often also include checksum files to verify the integrity of the downloads. The first piece of advice I can offer is always start with the official release archive. This should also include the sample config files. Then for your image requirements, I'd recommend exploring the kubernetes releases page or a trusted container registry, like docker hub, and downloading any image you require manually (using `docker save` if you have a local docker install to facilitate this) or using a tool such as `skopeo` to download them directly to disk. Make sure these files are available in a location reachable by kubespray on all nodes that will use containerd. I usually set up a local file server for this purpose, such as a simple http server using python or a similar option. This simplifies distribution across multiple nodes.

Now, let's discuss the kubespray configuration. Kubespray relies on a variable configuration that is defined in `inventory/group_vars/all.yml`, or a similar configuration file based on the specific installation. This is where you'll define the parameters for containerd, specifically how it accesses its resources. We need to modify this file to point to the local resources instead of trying to download them from the internet.

Here’s a snippet that illustrates the kind of changes I’ve made:

```yaml
container_manager: containerd
containerd_version: "1.4.4" # Or your specific version
containerd_use_systemd_cgroup: true
containerd_package_path: "/opt/local-repos/containerd-1.4.4.tar.gz" # Path to your containerd archive

containerd_config_path: "/opt/local-repos/config.toml" #path to your config file

# Disable remote image pull (we'll import images manually)
containerd_image_pull_policy: "Never"
containerd_custom_images:
  - image: k8s.gcr.io/pause:3.2
    local_path: "/opt/local-repos/pause_3_2.tar"  # Path to locally saved image tar file
  # Add more custom images as needed
```

Here, `containerd_version` specifies the exact version you intend to use. `containerd_package_path` and `containerd_config_path` now point to your local file system instead of relying on a remote download. Crucially, we set `containerd_image_pull_policy` to "Never" to prevent containerd from attempting to download images online. Then, we can define `containerd_custom_images` to include paths for locally stored container image tar files.

The next challenge we face is that kubespray needs to ensure these files are copied to the target node locations before the containerd installation occurs. This often involves modifying the kubespray playbooks or using ansible commands to pre-deploy these files. Consider the following snippet, that might need to be modified within your ansible playbooks:

```yaml
  - name: Copy containerd package
    copy:
      src: "{{ containerd_package_path }}"
      dest: "/tmp/"
      owner: root
      group: root
      mode: 0644
    when: containerd_package_path is defined

  - name: Copy containerd config
    copy:
      src: "{{ containerd_config_path }}"
      dest: "/tmp/"
      owner: root
      group: root
      mode: 0644
    when: containerd_config_path is defined

  - name: Copy local images
    copy:
      src: "{{ item.local_path }}"
      dest: "/tmp/"
      owner: root
      group: root
      mode: 0644
    loop: "{{ containerd_custom_images | default([]) }}"
    when: containerd_custom_images | default([]) | length > 0
```

This code snippet copies your containerd archive, configuration file, and pre-downloaded images to the `/tmp` directory on each target node. Depending on your specific ansible setup, the dest location may vary. This will require that this code snippet or something similar be run *before* containerd installation occurs.

Once the files are in place, you’ll need to adjust the kubespray playbooks to unpack and use those resources. The specific tasks within kubespray that handle containerd installation will have to be modified to read the binaries and images from their local locations, instead of attempting to download them. Here is a sample of the type of change that you would be looking for within kubespray’s ansible tasks:

```yaml
- name: Extract containerd
  unarchive:
    src: "/tmp/{{ containerd_package_path | basename }}"
    dest: "/usr/local/"
    owner: root
    group: root
    remote_src: yes # Important, because the file has been copied to remote
  when: containerd_package_path is defined

- name: install config
  copy:
      src: "/tmp/{{ containerd_config_path | basename }}"
      dest: "/etc/containerd/config.toml"
      owner: root
      group: root
      mode: 0644
      remote_src: yes
  when: containerd_config_path is defined

- name: Load images to containerd
  command: /usr/local/bin/ctr image import /tmp/{{ item.local_path | basename }}
  loop: "{{ containerd_custom_images | default([]) }}"
  when: containerd_custom_images | default([]) | length > 0

```

In this snippet, we are extracting containerd directly from `/tmp` and then importing any required images into containerd directly using the containerd cli binary (`ctr`). This step is essential because containerd has no other mechanism to retrieve them once we've configured the `containerd_image_pull_policy` to "Never". Notice the use of `remote_src: yes`, this is important because we moved the files in a previous task.

Remember to verify that the versions of containerd and the kubernetes components are compatible. Older versions of containerd might not fully support the latest kubernetes features. Refer to the official kubernetes and containerd documentation for compatibility matrices. The documentation for both projects on their respective websites and github pages are crucial for this process. Additionally, The Cloud Native Computing Foundation’s website and associated blogs are good starting points. Also the book "Kubernetes in Action" by Marko Lukša can be useful for understanding container runtimes within a Kubernetes context.

By carefully pre-populating the resources, modifying the kubespray configuration, and adjusting the installation process, you can successfully deploy containerd in an offline kubespray environment. It requires a meticulous approach, but it’s a manageable challenge with the right planning and understanding of the required steps. The key takeaway is to be thorough in preparing your local repository of files and to precisely control how those resources are accessed by kubespray. This ensures that the installation doesn't rely on remote sources and aligns with the restrictions of the offline environment.
