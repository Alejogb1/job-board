---
title: "Does containerd support a volume-from equivalent to Docker's?"
date: "2024-12-23"
id: "does-containerd-support-a-volume-from-equivalent-to-dockers"
---

Let’s tackle this with a bit of context, shall we? I recall a particularly frustrating project a couple of years back. We were migrating our container orchestration from a predominantly docker-based setup to one leveraging containerd more directly, aiming for increased flexibility and a finer level of control. One of the stumbling blocks was indeed this very issue: how to replicate the behavior of docker’s `volume-from` using containerd. The short answer is: containerd doesn’t offer a direct equivalent of `volume-from` out of the box. This doesn’t mean the functionality is impossible; it just requires a different approach, one rooted in a more granular understanding of container storage and management.

When you use `volume-from` in Docker, you're essentially telling the Docker daemon to share the volumes of an existing container with the new one. These volumes could be either named volumes or anonymous volumes. Containerd operates at a lower level, focusing on the underlying mechanics of running containers, whereas Docker provides a more user-friendly, higher-level interface. Consequently, containerd doesn’t abstract away these complexities to the same degree. Containerd doesn’t enforce concepts like the association of volume mounts at container level as docker does. Instead it relies on the underlying storage mechanism and its configuration.

So, how do we achieve this volume-sharing capability with containerd? The answer lies in configuring container mounts directly, targeting the underlying storage locations. It’s not as elegant as docker’s syntax, but it provides significantly more control and transparency.

The key here is understanding that volume mounts are essentially mounting host paths or other storage entities into a container's namespace. Containerd deals with these mount configurations directly. Instead of using named or anonymous volumes, you explicitly define the source and destination paths. This source can be a directory on the host, a dedicated mount point, or even the mount namespace of another running container. The latter, although complex, provides the closest equivalent to docker's `volume-from`.

Let’s break it down with a few examples that illustrate how to configure these mounts using containerd’s client library, often in a programming context like go (as that's how most of us interact with containerd).

**Example 1: Mounting a host directory:**

This is the simplest case, and arguably the most common use of volumes. Let's say we need to mount a local directory `/data` on the host to `/appdata` inside a container. In containerd, we would achieve this during container creation, using the `mount` spec as part of container options.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
    "github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/oci"
)

func main() {
    ctx := namespaces.WithNamespace(context.Background(), "my-ns")
    client, err := containerd.New("/run/containerd/containerd.sock")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

	image, err := client.Pull(ctx, "docker.io/library/nginx:latest", containerd.WithPullUnpack)
    if err != nil {
        log.Fatal(err)
    }

	container, err := client.NewContainer(
		ctx,
		"my-nginx-container",
		containerd.WithImage(image),
		containerd.WithNewSnapshot("nginx-snap", image),
        containerd.WithNewSpec(oci.WithMounts([]oci.Mount{
            {
                Type:   "bind",
                Source: "/data",
                Target: "/appdata",
				Options: []string{"rbind", "rw"},
            },
        })),
	)
	if err != nil {
		log.Fatal(err)
	}
    defer container.Delete(ctx, containerd.WithSnapshotCleanup)

    task, err := container.NewTask(ctx, containerd.WithTaskIO(os.Stdout, os.Stderr, os.Stdin))
    if err != nil{
        log.Fatal(err)
    }

    defer task.Delete(ctx)

	err = task.Start(ctx)
    if err != nil{
        log.Fatal(err)
    }
	
	statusC, err := task.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}

	<-statusC

	fmt.Println("Container exited.")

}

```

In this Go code snippet, you can see that we’re not specifying a `volume-from` anywhere. Instead, within `oci.WithMounts`, we define a bind mount using the `Source` path on host `/data` and linking it to `/appdata` inside our container. The `options` field is standard mount options in linux which are not related to container features. This approach provides direct control over the storage mount and is similar to mounting a host-based volume in docker manually.

**Example 2: Sharing a read-only volume from a named volume (emulating named volume sharing):**

This is a little more complex since we need to first create a directory where to host the data, let's say, `/namedvolume`, then add it as a mount. There’s no volume concept in containerd the same as in Docker. In containerd, we're creating a shared mount on the host. This example demonstrates a shared read only volume

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
    "github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/oci"
)

func main() {

    ctx := namespaces.WithNamespace(context.Background(), "my-ns")

    client, err := containerd.New("/run/containerd/containerd.sock")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

	image, err := client.Pull(ctx, "docker.io/library/busybox:latest", containerd.WithPullUnpack)
    if err != nil {
        log.Fatal(err)
    }
    
	// Create the shared directory.
	err = os.MkdirAll("/namedvolume", 0755)
	if err != nil {
		log.Fatal(err)
	}
	// Create a test file in the shared volume
	err = os.WriteFile("/namedvolume/test.txt", []byte("This is a shared file."), 0644)
	if err != nil {
		log.Fatal(err)
	}


    container1, err := client.NewContainer(
		ctx,
		"container1",
		containerd.WithImage(image),
		containerd.WithNewSnapshot("snap1", image),
        containerd.WithNewSpec(oci.WithMounts([]oci.Mount{
            {
                Type:   "bind",
                Source: "/namedvolume",
                Target: "/volume",
				Options: []string{"rbind", "ro"},
            },
        })),
	)
	if err != nil {
		log.Fatal(err)
	}

    defer container1.Delete(ctx, containerd.WithSnapshotCleanup)

	task1, err := container1.NewTask(ctx, containerd.WithTaskIO(os.Stdout, os.Stderr, os.Stdin))

	if err != nil {
		log.Fatal(err)
	}
	defer task1.Delete(ctx)

	err = task1.Start(ctx)
	if err != nil {
		log.Fatal(err)
	}

	statusC1, err := task1.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}

	<-statusC1


     container2, err := client.NewContainer(
		ctx,
		"container2",
		containerd.WithImage(image),
		containerd.WithNewSnapshot("snap2", image),
        containerd.WithNewSpec(oci.WithMounts([]oci.Mount{
            {
                Type:   "bind",
                Source: "/namedvolume",
                Target: "/volume",
				Options: []string{"rbind", "ro"},
            },
        })),
	)
	if err != nil {
		log.Fatal(err)
	}
    defer container2.Delete(ctx, containerd.WithSnapshotCleanup)

	task2, err := container2.NewTask(ctx, containerd.WithTaskIO(os.Stdout, os.Stderr, os.Stdin))

	if err != nil {
		log.Fatal(err)
	}

	defer task2.Delete(ctx)

	err = task2.Start(ctx)
	if err != nil {
		log.Fatal(err)
	}

	statusC2, err := task2.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}

	<-statusC2


	fmt.Println("Containers exited.")

}
```
Here, both `container1` and `container2` are given a bind mount pointing to the same location on the host `/namedvolume` this achieves the same effect of shared volumes that Docker provides. Notice the `ro` option indicates that all mounts to this volume are read only.

**Example 3: Sharing the mount namespace of another container (advanced use):**

This is the closest we get to direct emulation of Docker's `volume-from`. It involves retrieving the mount namespace of an existing container and using that as a volume source for new containers. This method is rarely used and its complexity makes it less practical than other methods. This is complex as it requires inspecting the underlying container to obtain the mount namespace and that is done through its pid. This requires careful coordination and should be used sparingly. It's included here only for comprehensive coverage and is quite advanced.

```go

package main

import (
    "context"
	"fmt"
	"log"
	"os"
    "github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/oci"
    "github.com/containerd/containerd/runtime/v2/runc"
)
func main() {
    ctx := namespaces.WithNamespace(context.Background(), "my-ns")

    client, err := containerd.New("/run/containerd/containerd.sock")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

	image, err := client.Pull(ctx, "docker.io/library/busybox:latest", containerd.WithPullUnpack)
    if err != nil {
        log.Fatal(err)
    }


    container1, err := client.NewContainer(
		ctx,
		"container1",
		containerd.WithImage(image),
		containerd.WithNewSnapshot("snap1", image),
        containerd.WithNewSpec(oci.WithMounts([]oci.Mount{
            {
                Type:   "bind",
                Source: "/tmp",
                Target: "/share",
				Options: []string{"rbind", "rw"},
            },
        })),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer container1.Delete(ctx, containerd.WithSnapshotCleanup)

	task1, err := container1.NewTask(ctx, containerd.WithTaskIO(os.Stdout, os.Stderr, os.Stdin))
	if err != nil {
		log.Fatal(err)
	}
	defer task1.Delete(ctx)

	err = task1.Start(ctx)
	if err != nil {
		log.Fatal(err)
	}

	statusC1, err := task1.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}

	<-statusC1

	taskinfo,err := task1.Info(ctx)
    if err != nil{
        log.Fatal(err)
    }

	
    pid := taskinfo.Pid

	mountNsPath := fmt.Sprintf("/proc/%d/ns/mnt", pid)



	container2, err := client.NewContainer(
		ctx,
		"container2",
		containerd.WithImage(image),
		containerd.WithNewSnapshot("snap2", image),
         containerd.WithNewSpec(oci.WithMounts([]oci.Mount{
            {
                Type:   "bind",
                Source: mountNsPath,
                Target: "/mnt",
				Options: []string{"rbind", "rw"},

            },
        })),
	)
	if err != nil {
		log.Fatal(err)
	}
    defer container2.Delete(ctx, containerd.WithSnapshotCleanup)

	task2, err := container2.NewTask(ctx, containerd.WithTaskIO(os.Stdout, os.Stderr, os.Stdin))
	if err != nil {
		log.Fatal(err)
	}
	defer task2.Delete(ctx)

	err = task2.Start(ctx)
	if err != nil {
		log.Fatal(err)
	}
	statusC2, err := task2.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}

    <- statusC2
    fmt.Println("Containers exited.")
}
```

In this example, after running `container1`, we obtain its pid and build the path to the mount namespace, `/proc/<pid>/ns/mnt`. Then, we use this path as a source for a bind mount in `container2`, effectively sharing the mount namespace. This allows `container2` to see the mounts of `container1`.  Again, this method is quite complex and requires intimate knowledge of underlying linux namespace mechanisms.

To delve deeper into container storage and related concepts, I would strongly suggest looking at *Linux Kernel Development* by Robert Love for understanding the underlying mechanisms of namespaces and mounts. Also, *Programming in Go* by Mark Summerfield will help understand more deeply the interaction of the client library with containerd. And, of course, the official containerd documentation is invaluable.

In summary, while containerd doesn’t directly replicate docker’s `volume-from`, it provides the necessary mechanisms to achieve similar results with explicit mount configurations. The key takeaway is to understand the underlying concepts of mounts, namespaces, and container storage to leverage containerd effectively.
