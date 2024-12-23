---
title: "How can a simple container pipeline be implemented using GitHub Actions or webhooks?"
date: "2024-12-23"
id: "how-can-a-simple-container-pipeline-be-implemented-using-github-actions-or-webhooks"
---

, let’s tackle container pipeline implementations, a topic I’ve navigated countless times across various projects. Thinking back, I vividly remember a particularly frustrating deployment process at a startup where we were pushing manually built docker images. It was messy, error-prone, and consumed precious developer time. Implementing a proper pipeline, whether with GitHub Actions or webhooks, became critical, and it fundamentally altered how we approached deployments. So, let's break down how to construct a straightforward container pipeline leveraging these two methods.

First, it's important to grasp the core concept: we want to automate the process of building, testing, and deploying containerized applications. That automation should trigger from code changes – usually a push to a repository. The pipeline needs to encapsulate these steps: source code retrieval, container image construction, some kind of testing (ideally unit and integration), and finally, pushing the built image to a container registry, which could be Docker Hub, GitHub Container Registry (ghcr.io), or others. We’ll consider both GitHub Actions and webhook approaches, each with its own advantages.

Let's explore GitHub Actions first. The beauty of actions is their tight integration with GitHub repositories, eliminating the need for complex configuration outside of the repository itself. You define the pipeline in a yaml file, typically within the `.github/workflows` directory of your repository. This file specifies the events that trigger the workflow (e.g., a push to the main branch), the jobs to execute (e.g., build, test, deploy), and the steps within those jobs.

Here's a skeletal example to get you started:

```yaml
name: Container Pipeline with GitHub Actions

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.run_id }}

  test:
   runs-on: ubuntu-latest
   needs: build-and-push
   steps:
    - name: Checkout code
      uses: actions/checkout@v4
    - name: Run tests
      run: |
        echo "This is a placeholder for your tests"
        # your actual test commands here
```

In this workflow, the `on: push` event specifies that this pipeline is activated when code is pushed to the `main` branch. The `build-and-push` job uses the `docker/build-push-action`, which streamlines building, tagging, and pushing the docker image to the GitHub Container Registry (ghcr.io). The `test` job is a placeholder, but crucial – you should place the commands to execute your testing suites here and it depends on the build-and-push stage completing successfully. The `github.actor` and `GITHUB_TOKEN` are context variables provided by GitHub Actions, providing authentication. The tag `${{ github.run_id }}` will ensure each build has a unique tag. This is a very basic version, of course, but this sets the foundation.

Now, let's pivot to webhooks. Webhooks offer a more decoupled approach where a code change in GitHub triggers an http request to a predefined URL. This is useful when you have a more complex build process outside of GitHub's own runners, perhaps needing specialized hardware or an entirely different orchestration environment.

For example, you might have a build server that listens for these webhook calls. On receiving one, the server would typically execute the build, test, and deployment phases. The advantage here is the flexibility it provides, but it introduces greater administrative overhead because you're responsible for the webhook listener.

Here's a python snippet to illustrate how you might handle a webhook on your side. This is deliberately minimal. A production version would require proper security measures:

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        if data['ref'] != 'refs/heads/main':
          self.send_response(200)
          self.end_headers()
          return # Ignore pushes to non-main branches
        #This would normally be handled by an asynchronous task system
        try:
            subprocess.run(["./build_and_deploy.sh"], check = True)
            self.send_response(200)
        except subprocess.CalledProcessError as e:
            self.send_response(500)
        self.end_headers()


def run(server_class=HTTPServer, handler_class=WebhookHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting webserver on port {port}")
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      pass
    finally:
      httpd.server_close()

if __name__ == "__main__":
    run()
```

And, as a complement to this server-side python code, a simple `build_and_deploy.sh` script to perform the building:

```bash
#!/bin/bash
set -e # Exit on error

echo "Building the application"
docker build -t my-app .

echo "Running unit tests (placeholder)"
# Add your actual testing commands here

echo "Pushing Docker Image (placeholder - use docker login and a proper registry)"
# docker login && docker push ...
```

In this basic python server example, we extract the push details. If the push is to the `main` branch, we execute the `build_and_deploy.sh` script to build a docker image, and run a placeholder testing step before a placeholder push step. In practice, this would need to be much more sophisticated and incorporate secrets management, a proper build process, robust error handling and more.

The approach to choosing between GitHub actions or webhooks really depends on your project requirements and operational constraints. GitHub Actions are simpler to set up within the ecosystem, while webhooks provide more flexibility but require more manual configuration. For a relatively small team that doesn’t require complex workflows, GitHub actions offers a very elegant solution.

For further study, consider "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation" by Jez Humble and David Farley; it’s a foundational text. Another solid choice is "Docker in Action" by Jeff Nickoloff and Stephen Kuenzli if you wish to delve deeper into containerization. For a deep dive into GitHub Actions specifics, consult the official GitHub Actions documentation, it's continually updated and contains many practical examples.

My experience implementing these pipelines has reinforced that the chosen method matters less than implementing one. The key is to automate your container workflow, reduce the number of manual steps and provide robust and consistent deployments. This is an iterative process. Start simple and expand as your requirements evolve. You will find that the reduction in manual effort and deployment errors will be immediately apparent.
