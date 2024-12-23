---
title: "How do I address python3.7 Docker vulnerabilities reported by docker scan?"
date: "2024-12-23"
id: "how-do-i-address-python37-docker-vulnerabilities-reported-by-docker-scan"
---

Right then, let's talk about those pesky vulnerabilities docker scan throws your way, particularly when you're dealing with older python environments like 3.7 within docker containers. I've certainly been down that rabbit hole more times than I care to recall, so I can offer some insights based on firsthand experience and the paths I’ve taken to resolve them. It's rarely a single magical command, but a strategic combination of approaches that work best.

First, it’s crucial to understand *why* docker scan is flagging these vulnerabilities. It's not just being pedantic; these are genuine security risks. Python 3.7, having reached its end-of-life, doesn’t receive security patches anymore. Consequently, any vulnerabilities discovered after its support cycle ended are never addressed in the upstream python distribution. This means your containers could be exposed to known exploits, especially if you’re using older base images that haven't been updated in a while. That's the core issue.

Here’s my practical approach, broken down into logical steps:

**1. The Obvious – Upgrade Python & Base Image:**

The most straightforward, and frankly the best, long-term solution is to upgrade to a supported python version. This often means migrating to 3.10, 3.11, or 3.12, depending on what’s stable at the time. Upgrading isn’t just about getting rid of the scanner’s warnings; it’s fundamental to maintaining a secure application. In many instances, I've found that using the latest alpine-based python base images is a solid start. They're minimal, reducing the overall attack surface and often include up-to-date package versions by default. The Dockerfile change typically looks something like this:

```dockerfile
FROM python:3.12-alpine3.19

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "your_app.py"]
```

*   **Explanation:** We've switched from a potentially outdated base image to `python:3.12-alpine3.19`. Alpine is lightweight, and `3.12` is an actively maintained python version. We install our requirements, copy the project, and then execute the application. Note that the specific alpine version (`3.19` in this example) should be kept up to date with the latest available. It's crucial to run `docker scan` again to confirm the vulnerabilities have been addressed.

**2. Pinning Dependencies with Extreme Care (When Upgrading is Difficult):**

Sometimes a full python upgrade is a substantial undertaking, perhaps due to legacy dependencies or extensive refactoring needs. In these scenarios, you need to be very meticulous about the dependencies you include in your project's `requirements.txt` file. I've learned from some projects that we are unable to change the python version due to dependency restrictions, this is a very difficult position to be in.

The key is to use exact versions whenever possible. Avoid using wildcard version specifiers such as `package_name>=2.0`. This introduces uncertainty because your build might include new, vulnerable versions later without you realizing it. The recommendation is to specify something like `package_name==2.5.1`. Further, after pinning down the versions, actively track the releases and changelogs of each dependency to ensure any security patches are installed. This includes not just python libraries but other system-level packages that might be included in your Docker image.

Consider this `requirements.txt` excerpt:

```text
Flask==2.2.5
requests==2.28.1
urllib3==1.26.18
gunicorn==20.1.0
```

*   **Explanation:** Every package is explicitly versioned. This strategy limits the risk of surprise updates introducing security holes. However, this is not a long-term strategy. Remember that these specific versions will also become obsolete over time. The long-term solution will always be an upgraded version. I had a project like this and the team had to monitor the CVE reports for every dependency pinned, it was a difficult, repetitive and error-prone process.

**3. Scanning & Minimization – A Post-Build Approach (When Pinning Isn’t Enough):**

After taking the measures outlined above, *always* re-scan your images. Sometimes, even after pinning all the versions, docker scan might report vulnerabilities in system packages installed within the base image itself. If you're utilizing a base image that includes a lot of pre-installed utilities, consider if you *really* need them. I've used this pattern many times to reduce the vulnerability footprint. You might even look at creating your own base image using the python slim images that include only the minimal requirements.

Another technique is to use multi-stage docker builds. This method allows you to separate build tools and intermediate dependencies from the final application container, keeping it lean. You compile your application in one stage and only copy the resulting executable and necessary files into your deployment image. This will help reduce the overall size and potential security concerns. Here's a slightly more complex example with a multi-stage Dockerfile:

```dockerfile
# Stage 1: Builder
FROM python:3.12-alpine3.19 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m compileall . # Optional bytecode compilation

# Stage 2: Final Image
FROM python:3.12-slim-buster
WORKDIR /app
COPY --from=builder /build .
CMD ["python", "your_app.py"]
```

*   **Explanation:** The first stage uses a full alpine-based image to build and compile. The second uses a smaller, more focused, `slim-buster` variant and copies only the necessary components, reducing vulnerabilities and overall image size. The bytecode compilation is optional but it helps reduce start up times. This stage is often referred to as a "distroless" image pattern.

**Further Reading and Resources:**

For a deeper understanding, I strongly recommend these resources:

*   **“Security Engineering” by Ross Anderson:** This book provides a fundamental understanding of security principles, which is invaluable in understanding why these vulnerabilities are critical. The broader security context will make it clear why upgrading and minimization is so vital.

*   **The Docker Security Documentation:** The official Docker documentation is the most authoritative source for details about image hardening, security best practices, and how to interpret output from `docker scan`. It details how the tool works, what is considered a vulnerability and mitigation strategies.

*   **CVE database (https://cve.mitre.org/):** Familiarize yourself with the Common Vulnerabilities and Exposures (CVE) system. Understanding what specific vulnerabilities are being reported allows you to make informed decisions on mitigating them.

**Final Words:**

Addressing docker scan results isn’t a one-time task. It requires continuous monitoring and a proactive approach. Always keep your python versions and dependencies up to date, minimize your container footprint, and regularly scan your images for vulnerabilities. The short-term solutions I've discussed are often necessary but should be seen as stopgaps to allow a migration to modern, supported ecosystems. Don’t treat docker scan as an annoyance; consider it a valuable tool for building more robust and secure systems. These actions are vital for ensuring your applications are safe from exploitation.
