---
title: "How can Singularity effectively replicate Docker's layer caching for a shared base image?"
date: "2024-12-23"
id: "how-can-singularity-effectively-replicate-dockers-layer-caching-for-a-shared-base-image"
---

Let’s dive into this, shall we? I recall a particularly intricate project a few years back, involving a large-scale scientific simulation pipeline. We initially deployed everything using Docker, enjoyed its excellent layer caching, which significantly sped up our build times and deployments. Then, due to security and infrastructure requirements, we transitioned to Singularity. The challenge, of course, was replicating that Docker-like layer caching performance with Singularity. It's a different beast, but not insurmountable.

The heart of the issue is that Singularity, by design, does not natively build images using layered file system operations like Docker does. Docker's `Dockerfile` approach creates a series of changes to a base image, building up in layers, each layer corresponding to a different command. These layers are immutable and cached. Subsequent builds that reuse prior stages benefit greatly. Singularity, conversely, primarily focuses on producing single-file images (sifs), often from existing Docker images or directly from filesystems, which naturally hinders leveraging this caching paradigm directly. However, there are strategies to approximate similar performance benefits.

The primary method to accomplish something akin to layer caching involves utilizing *modular image creation* with Singularity, leveraging the *overlay* functionality. Instead of rebuilding the entire image each time, we can separate our changes into multiple filesystems (or directories) that are then mounted on top of a base image during runtime. This is achieved in two stages: building reusable base images and applying overlay layers.

Let's break this down with a code example. We'll start by creating a reusable base image containing the core dependencies of our application.

```python
# Example 1: Building a base image
import os
import subprocess

def create_base_image(base_name, base_dir):
    """Creates a base singularity image from a directory."""
    cmd = ["singularity", "build", f"{base_name}.sif", base_dir]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error building base image: {stderr.decode()}")
        return False
    print(f"Base image {base_name}.sif created successfully")
    return True


if __name__ == "__main__":
    base_dir = "base_env" # Assume we have a directory with core libraries and packages
    if not os.path.exists(base_dir):
      os.makedirs(base_dir)
    
    with open(os.path.join(base_dir, 'environment.txt'), 'w') as f:
         f.write("numpy==1.23.5\n")
         f.write("scipy==1.10.0\n")
    
    print("Creating base image directory")

    cmd = ["python3", "-m", "venv", "venv"]
    process = subprocess.Popen(cmd, cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error creating venv: {stderr.decode()}")
        exit()

    print("Installing packages to venv")

    cmd = ["./venv/bin/pip", "install", "-r", "environment.txt"]
    process = subprocess.Popen(cmd, cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error installing packages: {stderr.decode()}")
        exit()


    create_base_image("base_image", base_dir)
```

In this Python script, we create a directory named `base_env`, create a virtual environment there, then install our core dependencies. We then use the `singularity build` command to construct a `base_image.sif`. This `base_image.sif` is our equivalent of a Docker base layer; it’s a singular file and can be reused across different builds.

Now, let's move on to overlay layers. Instead of baking in all the specific application components, we'll create separate directories representing distinct "layers" of functionality. This avoids needing to rebuild the entire image when these components change. Here’s the code for that:

```python
# Example 2: Building an overlay layer

import subprocess
import os
import shutil


def create_overlay_layer(base_image_path, overlay_name, overlay_dir):

    # Make overlay directory if it doesnt exists
    if not os.path.exists(overlay_dir):
      os.makedirs(overlay_dir)

    # Add a file to overlay_dir
    with open(os.path.join(overlay_dir, 'application.py'), 'w') as f:
        f.write('print("Overlay applied")\n')

    # Create an overlay container, copying only the overlay directory
    cmd = ["singularity", "build", "--fakeroot", "--overlay", overlay_dir, f"{overlay_name}.sif", base_image_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


    if process.returncode != 0:
      print(f"Error building overlay: {stderr.decode()}")
      return False

    print(f"Overlay {overlay_name}.sif created")
    return True


if __name__ == '__main__':
    base_image_path = "base_image.sif" # Assuming this exist from the previous example
    overlay_dir = "app_code"
    create_overlay_layer(base_image_path, "app_overlay", overlay_dir)


```

Here, we're creating an `app_overlay.sif`. Importantly, `overlay_dir` contains only our application-specific code (`application.py`). This overlay file, built using the `singularity build --overlay` command, doesn't contain a complete image. Instead, it holds changes (our application code) which can be mounted onto our base image at runtime. The `--fakeroot` argument is necessary to perform the build without needing root permissions, useful in shared environments.

Finally, let's look at how we’d run this application with the base image and overlay:

```python
# Example 3: Running with overlay
import subprocess
import os


def run_container(base_image_path, overlay_path, command):
    """Runs a singularity container with overlay."""
    cmd = ["singularity", "exec",  "--overlay", overlay_path, base_image_path, "python3", command]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
      print(f"Error running container: {stderr.decode()}")
      return False

    print(stdout.decode())
    return True



if __name__ == '__main__':
    base_image_path = "base_image.sif"
    overlay_path = "app_overlay.sif"
    if os.path.exists(base_image_path) and os.path.exists(overlay_path):
         run_container(base_image_path, overlay_path, "/app_code/application.py")
```

In this script, we utilize `singularity exec` with the `--overlay` flag, mounting `app_overlay.sif` on top of `base_image.sif` before executing `python3 /app_code/application.py`. Because we mounted the overlay layer using `--overlay`, the python script we wrote there is available at that path. This means you would execute the python script in the context of the combined filesystem. If `application.py` was modified and we rebuild `app_overlay.sif` the changes will be reflected immediately, while the base image is untouched. This setup mirrors the layered approach used by Docker for build efficiency and flexibility.

To delve deeper into these techniques, I would recommend exploring the Singularity documentation directly at their website, specifically the sections concerning image building and the overlay feature. The book "High-Performance Computing in Python" by Melissa E. O'Neill provides a great foundation for understanding underlying filesystem concepts. Additionally, examining the specifics of the Singularity system call API will offer insights into the implementation details and nuances of container behavior. While these resources are less directly targeted at layer caching, understanding these core pieces will enable you to implement more sophisticated solutions, including strategies to leverage tools like buildah which can produce OCI compliant images which can then be converted to sif using `singularity build` and also facilitate incremental image builds.

In conclusion, while Singularity's native image creation differs from Docker's, we can replicate much of the performance benefit of layer caching. By judiciously structuring your projects into base images and overlay layers, you create reusable components which accelerate both build times and allow for much more rapid iterations of your software. The approach requires some planning and a fundamental shift in mindset from a single monolithic image towards a modular filesystem approach, but the performance and flexibility it provides are well worth the effort.
