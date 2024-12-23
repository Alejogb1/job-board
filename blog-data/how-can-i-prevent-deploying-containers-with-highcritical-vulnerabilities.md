---
title: "How can I prevent deploying containers with high/critical vulnerabilities?"
date: "2024-12-23"
id: "how-can-i-prevent-deploying-containers-with-highcritical-vulnerabilities"
---

Alright,  It's a problem I've certainly seen my fair share of, both in my own projects and across teams I've worked with. Deploying vulnerable containers isn't just a theoretical risk; it can become a very real and very costly operational headache. The core issue stems from a disconnect – often between development practices and security protocols. It's not enough to merely *scan* for vulnerabilities; you need to bake security into your workflow, from image creation through deployment.

My experience has taught me that a multi-layered approach is the only way to effectively mitigate these risks. Simply relying on one check at the very end is like locking your front door and leaving all the windows wide open. You need to be proactive.

First, you need a robust image building strategy. This is where a lot of the battle is won or lost. The temptation to grab pre-built, readily available images from public registries is understandable, but often, these are like ticking time bombs. They might contain outdated dependencies, known exploits, or even deliberately malicious packages. Instead, I’d always advocate for creating your own base images. Start with a minimal, trusted operating system image (like a slim version of alpine or debian), and then meticulously add only the libraries and tools that you genuinely need.

Here's a very basic Dockerfile example, to demonstrate the principle. This is intentionally simplistic, but it highlights building with a minimal mindset:

```dockerfile
FROM alpine:3.18

RUN apk update && apk add --no-cache python3 py3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "your_app.py"]
```

Notice the `--no-cache` options. These reduce image size and avoid potential issues with cached packages. Also, explicitly specifying `alpine:3.18` ensures we’re using a specific, known version of a trusted base image.

Next, once you've got your base images sorted, you need an effective image scanning solution that integrates with your CI/CD pipeline. This isn’t just about identifying vulnerabilities, it's about setting up a policy framework to govern what’s acceptable. I've seen teams struggle with overwhelming vulnerability reports that are never addressed. The goal isn’t to find all possible issues, but to find and *fix* the critical and high-severity ones. This means you need automated tools, combined with clearly defined criteria for what triggers a build failure.

Here’s a sample bit of shell code that shows a simplified version of that integration, assuming you're using a scanner with a command-line interface (like trivy or anchore):

```bash
#!/bin/bash

IMAGE_NAME="your-app-image:latest"
SCAN_RESULTS=$(trivy image "$IMAGE_NAME" --format json)
VULNERABILITY_COUNT=$(echo "$SCAN_RESULTS" | jq '.Results[].Vulnerabilities | length')
HIGHEST_SEVERITY=$(echo "$SCAN_RESULTS" | jq -r '.Results[].Vulnerabilities[].Severity' | sort -r | head -n 1)

if [[ "$VULNERABILITY_COUNT" -gt 0 ]]; then
  echo "Vulnerabilities found: $VULNERABILITY_COUNT"
  echo "Highest severity: $HIGHEST_SEVERITY"

  if [[ "$HIGHEST_SEVERITY" == "CRITICAL" || "$HIGHEST_SEVERITY" == "HIGH" ]]; then
     echo "Critical or High severity vulnerabilities detected. Failing build."
     exit 1
  fi

  echo "No critical or high vulnerabilities detected. Proceeding..."
else
  echo "No vulnerabilities found. Proceeding..."
fi

```

This script executes a vulnerability scan on the generated image. It then parses the results and checks if any vulnerabilities with high or critical severities are found. If so, the script terminates with an error, preventing the image from being deployed. While this example is simple, real implementations would likely be more complex, involving integrations into your CI/CD platform.

Finally, it's crucial to establish a clear and documented process for addressing identified vulnerabilities. This isn't just the security team's responsibility; it's a shared effort across development and operations. Regularly update your base images and dependencies, and implement a system for reporting and patching vulnerabilities as they are discovered. Ignoring low severity vulnerabilities can be an acceptable trade-off in many cases, as long as you continuously monitor them.

Beyond specific tooling, it's crucial to focus on practices. For example, minimizing your attack surface is key, which often means reducing the size of your images by removing unnecessary packages and libraries (multi-stage builds within the Dockerfile can be used to accomplish this). Also, don't store sensitive information such as credentials directly within the container image. Use environment variables or secrets management systems instead. And of course, ensure you’re following the principle of least privilege, both within the container and in your overall container orchestration.

To give another concrete example, here’s a slightly more complex Dockerfile demonstrating a multi-stage build concept. It helps separate the build environment from the final image, leading to a smaller, more secure final artifact:

```dockerfile
# Build Stage
FROM golang:1.21 as builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o my-app main.go

# Final Stage
FROM alpine:3.18
WORKDIR /app
COPY --from=builder /app/my-app ./
EXPOSE 8080
CMD ["./my-app"]

```

This approach uses an initial `builder` stage with the full golang toolchain. The final image only copies the compiled binary and the minimal runtime dependencies, significantly reducing the attack surface.

In summary, preventing vulnerable container deployments is about a comprehensive strategy, encompassing secure base images, automated vulnerability scanning, a clear remediation process, and a security-conscious development culture. You need to be proactive, not reactive. Don't wait until something is broken.

For further reading, I highly recommend exploring the *CIS Benchmarks for Docker* and the *OWASP Top Ten for Containers*. Also, the *DevSecOps Handbook* by Shannon Lietz is an excellent resource for understanding the cultural and procedural aspects. Finally, *Effective DevOps* by Jennifer Davis and Ryn Daniels provides practical guidance on building robust and automated pipelines. These resources will give you a solid foundation to build upon. This isn’t an overnight fix, but a journey, and consistent effort yields meaningful results.
