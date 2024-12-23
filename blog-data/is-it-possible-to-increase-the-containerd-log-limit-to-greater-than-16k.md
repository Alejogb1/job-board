---
title: "Is it possible to increase the Containerd log limit to greater than 16K?"
date: "2024-12-23"
id: "is-it-possible-to-increase-the-containerd-log-limit-to-greater-than-16k"
---

Let's delve into the intricacies of Containerd logging limits, a topic that’s tripped up more than a few of us over the years. I recall one particular incident at a previous company—we were running a particularly verbose application in a high-throughput environment, and those default 16K log limits became a very real constraint, very quickly. The answer to whether you can increase the container log limit beyond that standard 16K within Containerd isn't a simple yes or no. It's more nuanced and dependent on how you configure the container runtime environment.

The 16K limit you're referring to isn’t inherently a hard constraint baked into the core of Containerd itself. Rather, it's usually a default setting imposed by the logging driver in use, commonly the `json-file` driver, which is often the default choice. This driver buffers container output and persists it to disk as JSON files, and that buffer has a default size. The restriction isn’t on the *amount* of logging the container generates, but on the *size of a single log message* it can handle before that message gets truncated or split. This is to prevent excessively large log lines from consuming too much memory during the buffering process or filling the log file with just one overly long line.

The core understanding here is that Containerd is a container runtime, and it delegates the actual logging and output processing to various log drivers. Therefore, increasing the limit isn't about modifying Containerd's core code but rather configuring the chosen log driver properly.

Let's break this down further into practical approaches. The first and probably the most straightforward solution is to switch to a log driver that doesn't impose the same 16k limit or has configurable limits. `fluentd` or `syslog` drivers are excellent examples, these are capable of handling larger log outputs, and they can route those logs to external storage or monitoring systems. Here’s a basic snippet demonstrating how you’d configure a container with the `fluentd` logging driver when starting a container using the `ctr` client:

```bash
ctr run \
  --log-driver fluentd \
  --log-opt fluentd-address=tcp://fluentd-server:24224 \
  docker.io/library/nginx:latest my-nginx
```

In this case, I'm using a simple nginx image as an example. This command tells Containerd to run the `nginx` image and route all its logs through a `fluentd` instance located at `tcp://fluentd-server:24224`. Now, the log limits are effectively governed by fluentd itself rather than Containerd’s default json-file driver. This approach offers flexibility as fluentd allows for additional log processing, routing and other sophisticated capabilities.

If migrating to a different log driver entirely isn't feasible in your scenario and you're bound to use the `json-file` driver because of your environment or other constraints, there's another avenue, although one that comes with certain considerations. You can adjust the `max-size` and `max-file` options within the `json-file` driver's configuration. While these options don't directly increase the *single message limit* (still around 16k), they do provide control over the log rotation and file size, indirectly influencing the overall logging behavior. Let me demonstrate a configuration change via a containerd configuration file to show how this is done:

```toml
[plugins."io.containerd.runtime.v1.linux".options]
  ...
  [plugins."io.containerd.runtime.v1.linux".options.container_log_path]
    type = "json-file"
    options = {
       "max-size" = "50m",
       "max-file" = "5"
    }
```

Here we are configuring the `json-file` log driver through the `containerd` config file usually found at `/etc/containerd/config.toml` (location varies depending on your distribution.) In this setup, I’ve set the `max-size` to 50 megabytes and `max-file` to 5, which rotates the log files. Note this setting, again, doesn’t increase the size of a single message, but prevents log files from growing too large and offers a measure of indirect control. This ensures, in this example, that we only keep up to 5 rotated logs of 50 megabytes each in size.

Finally, a third approach involves using a sidecar container logging pattern. Instead of relying on the primary container's logs, a separate logging container runs alongside, which intercepts the main application's logs, processes them and forwards them as needed. This pattern provides maximum flexibility, allows you to use any logging tools, and handles the log messages of any size while bypassing the limits associated with the `json-file` driver. Here’s a basic docker-compose example of how this would be structured:

```yaml
version: '3.8'
services:
  app:
    image: my-app:latest
    logging:
      driver: none
  logger:
    image: fluent/fluentd:v1.16
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
    ports:
      - '24224:24224'
```

In this snippet, the primary application `app` has its logging driver set to none. Instead the `logger` sidecar container runs an instance of fluentd, allowing the application container `app` to output to standard output or its own files, while fluentd is configured to ingest those logs. The `fluentd.conf` file would need to be configured based on your needs, but the overall principle is the same: bypass the container runtime's default logging and let the sidecar container handle logging in a dedicated fashion.

It’s important to understand that each approach comes with tradeoffs. Using `fluentd` or similar log drivers adds a dependency on external infrastructure but gains better scalability and feature richness. Tweaking the `json-file` driver provides quick local changes, but still maintains the constraints with the single message size. Sidecar patterns offer the most flexibility but add complexity to your deployment. I’ve seen all of these approaches in practice, each used effectively in different scenarios.

To solidify your understanding further, I strongly suggest exploring resources on logging best practices within the container ecosystem. Specifically, dive into the documentation for log drivers like `fluentd`, `syslog`, and any alternatives you may consider. The official containerd documentation provides an overview of supported log drivers, but for deep dives, the actual documentation for the log drivers themselves is paramount. For practical implementations, the book "Kubernetes in Action" by Marko Luksa offers an extensive exploration of container logging in Kubernetes environments, which are also based on the concepts of the underlying container runtime and logging drivers. Additionally, I recommend looking into papers or articles related to log aggregation and analysis, like those available from the OpenTelemetry project, as these will provide insight into designing your log management system.

In conclusion, while the common perception might be a hard 16K limit, the situation is more about understanding the architecture and the log driver configurations within the container runtime. With proper configuration, you can absolutely handle larger log outputs than the default would allow, and it really boils down to the specific needs and resources of your project. It's about moving beyond the default and proactively choosing the log strategy that best suits your application and environment.
