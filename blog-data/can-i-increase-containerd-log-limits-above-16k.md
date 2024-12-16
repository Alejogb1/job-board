---
title: "Can I increase containerd log limits above 16K?"
date: "2024-12-16"
id: "can-i-increase-containerd-log-limits-above-16k"
---

Okay, let's tackle this. You're hitting a common pain point with containerd’s default log handling, and I've certainly been there myself, particularly when troubleshooting verbose applications. The short answer, thankfully, is yes, you absolutely can increase those 16K log limits. It’s not a setting you’ll find exposed directly in a single config file, though, so let’s dive into the mechanics of it.

That default 16k limit you’re experiencing isn’t a static, hardcoded constraint in containerd proper, but rather an artifact of the `cri-containerd` plugin and the way it interfaces with the underlying log-handling mechanisms of your operating system. Specifically, it's tied to how the container runtime interacts with the `fifo` or socket, if one is configured, through which application logs are streamed.

The fundamental mechanism involves reading from a stream, usually the container's stdout/stderr, into a buffer, then flushing that buffer to persistent storage or another output. The 16K limit is the size of this intermediate buffer, and that limit is imposed within the containerd's log collection logic itself when using `cri-containerd`, rather than something controlled at the OS-level or even within containerd’s core.

The good news? We have options. While there is no simple flag to just bump up that 16k buffer size, we can achieve the desired outcome using several strategies. It’s worth mentioning, before jumping into code, that changing these settings can have resource implications. Larger buffers mean more memory consumed per container, so it's essential to monitor your system after making such adjustments. This might mean adjusting your infrastructure specs or scaling your resources. My experience debugging a particularly noisy microservice led me to quickly discover the importance of careful monitoring when manipulating log buffer size, and I’d suggest you always keep a keen eye on resource utilization after implementing changes like this.

Now, onto solutions:

**1. Utilizing an external logging driver (recommended for most production deployments):**

The most robust approach is to move away from the default `cri` logging entirely and delegate log collection to an external logging driver. These drivers handle the stream from the container and forward the data, typically to a central logging platform, bypassing the `cri-containerd` buffer limitation. This method avoids modifications to containerd and its plugin and is highly flexible for customization and scaling. The popular choices include `fluentd`, `fluent-bit`, and `promtail`, each offering configurable buffering, batching and other advanced features.

I’ve implemented this in countless environments and found it to be highly reliable. The benefit here isn’t just about bypassing the 16k limit; it's about having a comprehensive logging solution that integrates smoothly with modern observability practices. Instead of looking at raw logs through `kubectl`, you’re working with structured data, which simplifies analysis and allows for sophisticated alerting.

Here is a snippet showcasing basic configuration using `fluent-bit` as an example, assuming a systemd context, or, one where docker, the common container runtime, is the default log driver:

```
[SERVICE]
    flush        1
    log_level    info
    daemon       off
    parsers_file  parsers.conf

[INPUT]
    name        systemd
    Tag         containerd.*
    Systemd_Filter _SYSTEMD_UNIT=cri-containerd.service
    Read_From_Head   On

[FILTER]
    Name    kubernetes
    Match   containerd.*
    Kube_URL  https://kubernetes.default.svc:443
    Kube_CA_File  /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File  /var/run/secrets/kubernetes.io/serviceaccount/token
    Kube_Tag_Prefix  containerd.var.log.pods.*
    Kube_Tag_Suffix .log
    Merge_Log On
    Merge_Log_Key   log
    Use_K8S_Parser On


[OUTPUT]
    Name        stdout
    Match       *
```

This `fluent-bit` configuration collects logs from `cri-containerd.service` via systemd and outputs the structured logs to standard output. In a real deployment, you’d replace the stdout output with a more persistent backend like Elasticsearch, Loki, or Splunk. A key thing to note is that the filter directs us to `cri-containerd.service` to capture the logs at the container runtime level, bypassing the 16k limit of the built-in `cri` plugin mechanism.

**2. Custom Log Driver Plugin for `cri-containerd` (Advanced, less common):**

It's technically feasible, although significantly more involved, to create your own `cri-containerd` log driver plugin. This plugin would be responsible for capturing logs from the container’s streams and processing them according to your specific requirements, including custom buffering. This approach offers complete control but requires a deeper understanding of the `cri-containerd` API, golang, and how the logging subsystem interacts with the container runtime. It’s not an approach I would suggest for most users because of the overhead of maintaining this type of custom software; I only mention it for completeness.

For those wanting to dive down this route, the documentation and examples provided by containerd’s team are the best start. A good place to look is in the `pkg/cri/streaming/log` and `runtime/v1/plugin/v1` directories of the `containerd/containerd` repository on github, where the core implementations reside.

The code, in go, might resemble this skeleton in a very basic sense (note that a fully implemented version is significantly more complex):

```go
package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
    log "github.com/containerd/containerd/log"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/plugins/cri/streaming"
	"github.com/containerd/containerd/runtime/v1/task"
    "github.com/containerd/containerd/sandbox"
)

const pluginID = "my-custom-log-driver"

type logWriter struct {
	output io.Writer
    buffer []byte
    bufferSize int
}
func (lw *logWriter) Write(p []byte) (int, error) {
    n := len(p)
    if len(lw.buffer) + n > lw.bufferSize {
        flushed, err := lw.output.Write(lw.buffer)
        if err != nil {
            return 0, err
        }

        if flushed != len(lw.buffer) {
            // Handle partial writes which can happen in concurrent access.
             log.G(context.Background()).Warnf("partial flush to log writer")
        }

        lw.buffer = nil

    }

    lw.buffer = append(lw.buffer, p...)
    return n, nil
}

func (lw *logWriter) Close() error {
    if len(lw.buffer) > 0 {
        _, err := lw.output.Write(lw.buffer)
        return err
    }
    return nil
}
type logDriver struct {
    bufferSize int
}

func (ld *logDriver) Open(ctx context.Context, stream *streaming.LogStream) (io.WriteCloser, error) {
   log.G(ctx).Infof("Opening custom log driver for %s", stream.TaskID)
    return &logWriter{
            output:   os.Stdout,
            bufferSize: ld.bufferSize,
        }, nil
}
func (ld *logDriver) Name() string {
    return pluginID
}
func NewLogDriverPlugin() func(ctx context.Context, config interface{}) (interface{}, error){
     return func(ctx context.Context, config interface{}) (interface{}, error){
        return &logDriver{bufferSize: 65535}, nil // 64k buffer size
     }
}

func main() {

    plugin.Register(&plugin.Registration{
		Type:   plugin.LogPlugin,
		ID:     pluginID,
		Requires: []plugin.Type{plugin.RuntimePlugin, plugin.SandboxControllerPlugin},
		Init: NewLogDriverPlugin(),
	})

    select { }
}
```

This snippet illustrates a custom log driver, configured with a 64k buffer and writes to stdout. This is only a skeleton and not intended for real-world use directly. A production driver would require handling timestamps, logging levels and other requirements. It demonstrates the architecture and required `containerd` framework interactions for implementing a custom solution.

**3. Using larger `fifo` socket (Less reliable, not recommended):**

Another option I've used sparingly, with mixed results, is configuring a larger `fifo` socket for container logging. By default, the `cri-containerd` plugin uses a named pipe or a Unix domain socket to relay logs from the container to the containerd daemon. By increasing the buffer size of this socket, you can, to a degree, increase the amount of log data that is buffered before being flushed to the containerd logging infrastructure.

However, this approach is not reliable because it is susceptible to lost logs should the socket buffers overflow. Furthermore, the socket buffer size is heavily influenced by system-level settings, requiring both modifications to containerd configuration as well as to OS configuration parameters, which makes deployments brittle and hard to maintain. I mention this only for completeness, as it's rarely a viable solution in anything beyond a development or test setting. This is because this is a system-wide operation that impacts not only containerd and its containers, but any service utilizing these types of operating system primitives.

While it’s not a code modification to containerd itself, here's a snippet showing how one might configure containerd to use a socket for log management, which then, can have the socket size potentially configured at the OS level (usually through `sysctl`):

```toml
# /etc/containerd/config.toml

[plugins."io.containerd.grpc.v1.cri".containerd.log]
  path = "/run/containerd/s.log"
  type = "socket"
```

Then you would use `sysctl` to configure the underlying socket size.

**Concluding Thoughts:**

While increasing the 16k log buffer limit in containerd might seem like a simple adjustment on the surface, it’s not a knob you can easily tweak directly. The default buffer size is a consequence of how `cri-containerd` handles log streams, and as we explored, the best strategy is to use an external logging driver, such as `fluent-bit` or `fluentd`. These tools provide robust solutions for forwarding logs, while also offering additional features, like filtering, aggregation, and integration with analysis platforms. A custom log driver is also a solution, albeit one that introduces maintenance challenges. While the socket approach is the quickest, it's the least reliable.

For further reading, I'd recommend exploring the documentation for `fluent-bit`, `fluentd` (or other logging forwarders you choose) and the `containerd` architecture diagrams, particularly the areas relating to the `cri-containerd` plugin. The official containerd documentation is quite comprehensive and can be found on the `containerd.io` website. Lastly, a good book to consider is "Kubernetes in Action" by Marko Luksa, which provides an in-depth look at how container runtimes and logging are interconnected.
