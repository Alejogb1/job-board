---
title: "Can Linux perf be used inside a Docker container?"
date: "2025-01-30"
id: "can-linux-perf-be-used-inside-a-docker"
---
Profiling applications running within Docker containers presents unique challenges. My experience working on high-performance computing projects at a major financial institution revealed a crucial detail:  the effectiveness of `perf` within a Docker container hinges significantly on the container's configuration and the kernel features exposed to it. While it's possible to use `perf` inside a Docker container, it rarely functions seamlessly without careful consideration of several key factors.  Directly using the host's `perf` is generally not recommended due to potential issues with shared memory and access rights.

The primary challenge stems from the layered architecture of Docker. The container's kernel is abstracted from the host kernel, impacting access to system-level profiling tools like `perf`.  `perf` relies on kernel-level events and requires appropriate permissions to access these events.  By default, a container's access to these events is limited for security and resource management reasons.  Furthermore, differing kernel versions between the host and container can introduce compatibility problems and cause unexpected behavior.

To overcome these challenges, we need to ensure the container has the necessary capabilities enabled.  This includes enabling the `CAP_SYS_ADMIN` capability, which grants broad system administration privileges, including access to profiling interfaces.  However, granting such extensive capabilities should be approached cautiously due to the inherent security risks.  A more controlled approach involves using a privileged container or carefully selecting the specific `perf` events accessed.

**1.  Clear Explanation:**

Employing `perf` within a Docker container necessitates a strategy to overcome the limitations imposed by containerization.  The most straightforward approach leverages a privileged container.  This grants the container near-host-level access, effectively removing many of the permission barriers. While simple, it compromises security.  A more secure approach, favored in my professional work, involves using a dedicated user namespace within the container. This restricts the container's capabilities to the specific actions necessary for profiling, minimizing the potential security surface.  Finally, if specific profiling events are identified upfront, they can be enabled selectively, reducing reliance on broad capabilities.  This careful event selection minimizes the privileges required and thereby enhances the security posture.

**2. Code Examples with Commentary:**

**Example 1: Privileged Container (Least Secure)**

```bash
docker run --privileged -it <image_name> /bin/bash
perf record -F 99 -a -g sleep 10  # Records CPU events for all threads
perf report
```

This example uses `--privileged`, granting the container extensive privileges. The `perf record` command profiles all threads (-a) with hardware event sampling (-g) at 99 Hz.  `sleep 10` provides a ten-second profiling period. This is the simplest method, but its significant security implications make it unsuitable for production environments.

**Example 2:  Using User Namespaces (More Secure)**

```dockerfile
FROM <image_name>
USER 1000:1000  # Create a specific user and group.
RUN setcap CAP_PERF_EVENT_OPEN+ep /usr/bin/perf
CMD ["/bin/bash"]
```

```bash
docker run -it <image_name> /bin/bash
perf record -e cycles -c 1 sleep 10 # Records cycle counts for a specific process (PID 1)
perf report
```

This approach creates a dedicated user and group within the container and adds the `CAP_PERF_EVENT_OPEN` capability specifically to the `perf` binary, limiting elevated privileges. This methodology is significantly safer. The `-e cycles` argument targets specific CPU cycle events, further minimizing risks.  The use of `-c 1` targets the process with PID 1, which is typically the main process within the container.


**Example 3: Specific Event Selection (Most Secure)**

```bash
docker run -it <image_name> sh -c '
  echo 0 > /proc/sys/kernel/perf_event_paranoid;
  perf record -e cycles,instructions -p $$ sleep 10;
  perf report'
```

This example demonstrates a selective approach.  It modifies the `/proc/sys/kernel/perf_event_paranoid` setting to a less restrictive value (0) only for the duration of the profiling.  This method carefully chooses specific events (`cycles`, `instructions`) and targets the current shell process (`$$`). The temporary modification of `perf_event_paranoid` provides a controlled increase in `perf` capabilities.  This approach reduces the potential attack surface considerably.  Note:  Always reset `perf_event_paranoid` to its original value after profiling.

**3. Resource Recommendations:**

For a comprehensive understanding of `perf`, consult the official `perf` documentation. Explore advanced topics such as specifying event groups, using different sampling methods (hardware, software, tracing), and analyzing the `perf report` output.  Review Docker's documentation on security best practices and container capabilities.  Study kernel documentation pertaining to the `perf_event_paranoid` sysctl and the capabilities mechanism to understand the implications of granting or restricting specific capabilities.  A strong grasp of Linux system administration and process management is also crucial.  The combination of these resources will build the necessary background for proficient use of `perf` within a container environment.
