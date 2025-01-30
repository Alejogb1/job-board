---
title: "How can I stop containers with names containing a specific string using podman?"
date: "2025-01-30"
id: "how-can-i-stop-containers-with-names-containing"
---
The efficient management of containerized applications often requires targeted actions, such as stopping containers based on specific naming conventions. Using `podman` directly, a straightforward approach involves a combination of filtering and command execution. I’ve frequently encountered scenarios where dozens of containers are running, making manual selection impractical. The key here is leveraging `podman ps`’s format and filter options alongside `podman stop`.

The core problem is identifying the containers we want to stop, given that we only know a portion of their names. Podman offers powerful filtering mechanisms with `podman ps`, which can display a wide array of information beyond the basic `CONTAINER ID` and `IMAGE` columns. Furthermore, we can specify a custom format to extract just the container name. By piping the output of this formatted command to `xargs`, which takes that output as arguments to another command, `podman stop` in this case, we can achieve the desired outcome.

Let's start with a basic implementation and progressively improve it. My initial efforts focused on a straightforward pipeline, filtering using the `--filter` option directly:

```bash
podman ps --filter "name=my-prefix*" --format "{{.Names}}" | xargs podman stop
```

This example uses `podman ps` to list all containers. The `--filter "name=my-prefix*"` portion instructs `podman ps` to only display containers whose names begin with `my-prefix`. The `--format "{{.Names}}"` specifies that we only want to see the container names in the output. Finally, the piped `xargs podman stop` takes the container names as arguments to the `podman stop` command, effectively stopping them. This worked well for my initial use case, but it had several limitations that became apparent later on in more complex projects.

Specifically, this approach relies on substring matching and might accidentally capture containers with unintended prefixes or suffixes. It also doesn't handle errors very gracefully. For example, if a container has already stopped, `podman stop` would throw an error that, while technically harmless, pollutes the terminal output.

To address these limitations, I have since shifted to a more robust approach. One improvement involves using regular expressions. Podman supports regex for the filter option, and while they can become complex, they offer a very powerful way to describe patterns. Here is an example showing this change:

```bash
podman ps --filter "name=^my-prefix-.*$" --format "{{.Names}}" | xargs podman stop
```

This refined version uses the regex `^my-prefix-.*$` with the filter option. The `^` anchors the regex to the beginning of the string, while the `$` anchors it to the end. Therefore, this effectively requires the container name to start with “my-prefix-” and have any characters afterward. The use of a more specific regex pattern reduces the chance of inadvertently stopping containers whose names simply happen to contain a matching substring. This reduces the scope of operation of podman stop to only those containers which have names that conform to the exact name pattern which is described by the regex.

However, I was not yet completely satisfied. While we had now improved the matching of the containers, we are still printing out errors from `podman stop` if any of those containers are already stopped. This is visually disruptive and not informative to the user. I have worked around this by introducing an additional filtering step using `grep` which checks whether the container is running and only operates on those which are in a running state.

Here's the final example that I have used effectively:

```bash
podman ps --format "{{.Names}}\t{{.State}}" | grep "my-prefix.*Running" | awk '{print $1}' | xargs podman stop
```

This version starts by getting both the name and state of all containers. Then `grep` filters the output for only those containers which have the specific prefix *and* are currently running. Then `awk` is used to extract just the name which is the first field in the space separated output and passed as argument to `podman stop` through `xargs`.

Specifically, this pipeline operates in stages: First, `podman ps --format "{{.Names}}\t{{.State}}"` lists all containers along with their states, separating these two columns by a tab character.  Second, `grep "my-prefix.*Running"` filters these results, selecting lines where the name includes "my-prefix" followed by any characters, and the state matches exactly "Running". Third, `awk '{print $1}'` extracts the container name from the filtered lines, which corresponds to the first field. Finally, the output of `awk` is piped to `xargs podman stop`, which stops only the filtered containers. This significantly reduces noise in the command output. In practice, this approach yields a more reliable and user-friendly experience. This approach is also quite flexible. You can change `Running` to `Exited` to selectively operate on exited containers. You can also remove the `grep` portion to perform actions on *all* containers matching the name prefix regardless of state.

In terms of resource recommendations for further understanding, I suggest exploring the official Podman documentation, particularly the sections relating to `podman ps` and its filtering options, as well as the `format` command, and `xargs`. Additionally, familiarity with regular expression syntax is very helpful, with online tutorials easily discoverable. Practical experimentation using dummy containers and different naming conventions is also very useful and helps with understanding the behavior of these commands in the real world.
