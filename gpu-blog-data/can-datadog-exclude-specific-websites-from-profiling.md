---
title: "Can Datadog exclude specific websites from profiling?"
date: "2025-01-30"
id: "can-datadog-exclude-specific-websites-from-profiling"
---
Datadog APM's website profiling capabilities, while powerful, do not offer a direct mechanism to exclude specific websites based solely on their domain name or URL.  My experience implementing and troubleshooting Datadog APM across numerous large-scale applications confirms this limitation.  Excluding specific websites requires a more nuanced approach leveraging Datadog's tagging and filtering functionalities, often in conjunction with custom instrumentation.  This necessitates a deeper understanding of how Datadog's tracing and profiling operate.

**1. Understanding Datadog's Profiling Mechanics**

Datadog APM profiles applications by instrumenting the code, typically through agents and libraries specific to the runtime environment (e.g., Java, Python, Node.js). These agents capture performance metrics at various levels, including function call times, memory allocation, and garbage collection statistics.  Crucially, this profiling happens *within* the application context.  Datadog receives and processes this telemetry data, presenting it in a user-friendly interface. The key takeaway here is that the filtering happens post-profiling, at the data aggregation and visualization level, not at the source.  Datadog doesn't inherently know, at the profiling stage, that a specific request originates from, say, `example.com`. It observes execution time and resource consumption within *your* application.

Therefore, directly blocking website-specific profiling is impossible without modifying the application's logic. The solution necessitates carefully tagging requests within your application and then filtering these tagged spans in Datadog.

**2. Code Examples and Explanations**

The following examples demonstrate how to achieve effective website exclusion using different programming languages and Datadog's tracing API. Remember to adapt these examples to your specific application architecture and tracing library.  These examples assume familiarity with Datadog's tracing libraries for respective languages and basic understanding of asynchronous programming where relevant.

**Example 1: Python with the ddtrace library**

```python
from ddtrace import tracer, patch_all
from urllib.parse import urlparse

patch_all()

@tracer.wrap()
def process_request(request):
    parsed_url = urlparse(request.url)
    netloc = parsed_url.netloc
    span = tracer.current_span()
    if netloc not in ["example.com", "exclude.net"]:
        span.set_tag("exclude", "false")
        # ... Your request processing logic ...
    else:
        span.set_tag("exclude", "true")
        # ... Minimal processing, logging perhaps, but no profiling ...
    return response
```

Here, we use the `ddtrace` library to instrument the `process_request` function. We parse the request URL and add a custom tag `exclude`. The Datadog dashboards can then be filtered based on this tag (`exclude:false`), effectively excluding spans with `exclude:true`. The minimal processing within the `else` block is crucial to avoid triggering significant profiling data for excluded websites.


**Example 2: Node.js with the `@dd/trace` library**

```javascript
const { tracer } = require('@dd/trace');
const url = require('url');

function processRequest(request, response) {
  const parsedUrl = new URL(request.url);
  const hostname = parsedUrl.hostname;
  const span = tracer.scope().active();
  if (!['example.com', 'exclude.net'].includes(hostname)) {
    span.setTag('exclude', 'false');
    // ... your request processing logic ...
  } else {
    span.setTag('exclude', 'true');
    // ... minimal processing ...
    response.end('Request excluded.');
  }
}
```

Similar to the Python example, this Node.js snippet uses the `@dd/trace` library to add the `exclude` tag based on the hostname.  The logic again prioritizes minimal processing for excluded websites to minimize their contribution to the profiler's data.


**Example 3: Java with the dd-trace-java library**

```java
import io.opentelemetry.trace.Span;
import io.opentelemetry.trace.Tracer;
import java.net.URL;
// ... other imports

public class RequestProcessor {
    private final Tracer tracer; // obtained from your Datadog initialization

    public void processRequest(HttpServletRequest request) {
        Span span = tracer.spanBuilder("processRequest").startSpan();
        try {
            URL url = new URL(request.getRequestURL().toString());
            String host = url.getHost();
            if (!List.of("example.com", "exclude.net").contains(host)) {
                span.setAttribute("exclude", false);
                // ... your request processing logic ...
            } else {
                span.setAttribute("exclude", true);
                // ... minimal processing ...
            }
        } catch (Exception e) {
            // Handle exceptions
        } finally {
            span.end();
        }
    }
}
```

This Java example uses the OpenTelemetry API (often integrated with Datadog) to achieve similar tagging and filtering.  The core principle remains the same:  applying custom tags to distinguish between included and excluded requests for subsequent filtering in the Datadog UI.



**3. Resource Recommendations**

For a deeper dive, consult the official Datadog documentation on APM, specifically the sections on custom instrumentation and tracing.  Review the API documentation for your specific language's Datadog tracing library.  Understanding the concepts of spans, tags, and metrics within the context of distributed tracing is essential for effective implementation of these solutions.  Furthermore, explore Datadog's documentation on creating custom dashboards and leveraging their filtering capabilities to efficiently analyze the profiled data.  Pay close attention to the limitations of tag cardinality to avoid performance issues.

In summary, while Datadog APM doesn't directly support website exclusion from profiling,  strategically using custom tags and filtering within your application code combined with effective use of Datadog's dashboarding functionalities provide a robust and practical workaround.  This approach ensures that only relevant data contributes to performance analysis, improving the accuracy and efficiency of your monitoring efforts. Remember that careful design and implementation of tagging is crucial to maintain the effectiveness and avoid performance overhead introduced by the tagging process itself.
