---
title: "How can I use sidecar deployments to load UI artifacts to Nginx?"
date: "2024-12-23"
id: "how-can-i-use-sidecar-deployments-to-load-ui-artifacts-to-nginx"
---

Alright, let's talk about sidecar deployments and serving UI artifacts with Nginx; it’s a topic I’ve navigated quite a bit over the years, especially during my time working on a heavily containerized microservices architecture where we wanted to keep concerns neatly separated. It’s not just about getting things running, but about doing it efficiently, robustly, and in a way that scales with your needs.

The fundamental problem we’re tackling here is how to get your frontend, let's say a react or angular app's static build files, into the hands of your users when that frontend is designed to be stateless and separate from the backend serving API requests. In a traditional monolithic application, these assets would simply live in the same filesystem, but in a microservice setup, you're likely containerizing each component, and keeping them separate is a key goal. A sidecar deployment, in this context, refers to having a companion container alongside your primary application container—in our scenario, Nginx would be the primary and a container housing the UI build artifacts is our sidecar.

The value proposition of this pattern is significant. Firstly, it promotes modularity. The Nginx container is focused solely on serving static files and acting as a reverse proxy, while the UI artifacts container houses only the build output of your frontend application. This separation of concerns makes both containers easier to manage, test, and scale. Secondly, it helps in decoupling deployments. We don’t need to rebuild the entire Nginx image every time we update UI assets. Instead, we can swap out the UI artifact container or update it independently. Thirdly, using sidecar patterns effectively mitigates common problems such as inconsistent builds or issues arising from deploying UI artifacts directly into an Nginx image itself, which often leads to issues with CI/CD pipelines.

Now, let's get down to specifics on how to achieve this. I’ve typically used one of two primary approaches. The first and more straightforward method is to use a shared volume. You have both the Nginx container and the artifact container mounted to the same persistent volume. The artifact container would then populate this volume with the necessary build files. The Nginx container, configured to look at that shared volume for the UI files, will then serve them.

Here's a simplified example demonstrating this using docker-compose, as it is often used to test this type of configuration locally. Note, this is a local example for illustrative purposes. In production, volumes would be managed differently, often using persistent volume claims in orchestration tools like Kubernetes:

```yaml
version: "3.8"
services:
  nginx:
    image: nginx:latest
    ports:
      - "8080:80"
    volumes:
      - ui_volume:/usr/share/nginx/html
    depends_on:
      - ui-artifacts
  ui-artifacts:
    image: alpine/git
    volumes:
      - ui_volume:/app/build
    command: sh -c "git clone <your-git-repo> /app && cd /app && npm install && npm run build && cp -r build/* /app/build"

volumes:
  ui_volume:
```

In this snippet, we have two services: *nginx* and *ui-artifacts*. The `ui-artifacts` service clones your UI repository, installs the necessary dependencies, and executes the build process, placing the resulting static files in `/app/build`. Both the `nginx` service and `ui-artifacts` service mount the shared `ui_volume`, which synchronizes the build output to `/usr/share/nginx/html`, the standard path where nginx looks for files. This approach is simple, easy to understand, and avoids having to recreate images on every ui update. It also uses the alpine/git image which is lightweight to perform the clone and build, which keeps your overall setup lightweight.

The second approach, and one I tend to favor for environments where shared volumes aren't ideal or where a clear separation between "write" and "read" is paramount, involves the artifact container acting as a short-lived build container that extracts the files to a shared emptyDir volume only. The files are then copied to a read-only, statically-mounted emptyDir volume shared with nginx using an init container. The init container essentially pre-populates the read-only volume at startup and then shuts down. This method offers improved isolation and makes it impossible for the application container to inadvertently modify the static files being served. This would look similar to the following Kubernetes manifest snippet:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ui-deployment
spec:
  selector:
    matchLabels:
      app: ui
  template:
    metadata:
      labels:
        app: ui
    spec:
      initContainers:
      - name: init-ui
        image: alpine/git
        volumeMounts:
        - name: ui-content-writable
          mountPath: /ui-artifacts
        command: ["sh", "-c"]
        args: ["git clone <your-git-repo> /ui-artifacts/app && cd /ui-artifacts/app && npm install && npm run build && cp -r build/* /ui-artifacts"]

      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
        volumeMounts:
        - name: ui-content-readonly
          mountPath: /usr/share/nginx/html
        livenessProbe:
            httpGet:
              path: /
              port: 80
            initialDelaySeconds: 3
            periodSeconds: 3

      - name: copy-ui
        image: alpine/git
        volumeMounts:
          - name: ui-content-writable
            mountPath: /ui-artifacts
          - name: ui-content-readonly
            mountPath: /dest
        command: ["/bin/sh", "-c"]
        args: ["cp -r /ui-artifacts/* /dest/"]

      volumes:
      - name: ui-content-writable
        emptyDir: {}
      - name: ui-content-readonly
        emptyDir: {}
```

This example provides a basic configuration using two `emptyDir` volumes. The `init-ui` initContainer populates a *writable* emptyDir volume with the output of the build process from the UI artifacts source code. Then, the `copy-ui` container copies from the *writable* volume to the *read-only* volume. The `nginx` container serves its static assets from this *read-only* volume. While more verbose, it is arguably a safer deployment pattern.

The last approach I would suggest using, and what I leaned on most heavily for larger systems, is to utilise a pre-built container. Here, I’m referencing a dedicated UI artifact container built during CI and simply deployed as a sidecar. The build process remains the same, but the result is a ready-to-go image that can be spun up and mounted in the same way as the *read-only* volume example, except that the image’s content is read-only and part of the image filesystem itself, not a volume. This reduces the reliance on init containers or volume copying and makes rollouts significantly more reliable. The deployment becomes straightforward: the Nginx container and your pre-built artifact container are deployed together, with the artifact container's static files mounted in a read-only fashion for the Nginx container to serve. For brevity I will refrain from posting a full deployment example here as it is similar to the second approach, but the important takeaway is that the artifact is now a standalone deployable component.

To dive deeper into these concepts, I'd strongly recommend consulting "Kubernetes in Action" by Marko Lukša. It provides a comprehensive understanding of Kubernetes and container deployment patterns, which is invaluable for deploying applications in this manner. Additionally, “Docker Deep Dive” by Nigel Poulton is excellent for solidifying your understanding of containers and how they work. Also, the cloud native computing foundation's (cncf) documentation is an excellent resource to research deployment patterns.

In closing, sidecar deployments for serving UI artifacts with Nginx offer a powerful and flexible approach to structuring modern, containerized applications. It's not a one-size-fits-all solution, and choosing the appropriate strategy, such as shared volumes, initial containers, or pre-built artifact containers, will depend on the constraints of your specific environment, the desired degree of isolation, and overall management overhead. Through practical experience, it's something you'll refine as you gain a better understanding of your own needs.
