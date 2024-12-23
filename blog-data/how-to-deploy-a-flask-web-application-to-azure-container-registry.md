---
title: "How to deploy a Flask web application to Azure Container Registry?"
date: "2024-12-23"
id: "how-to-deploy-a-flask-web-application-to-azure-container-registry"
---

Okay, let's tackle this. Deploying a Flask application to Azure Container Registry (ACR) is a process I've gone through countless times, and while it might seem complex initially, breaking it down into steps makes it quite manageable. It's crucial to understand that we're essentially packaging our application into a Docker container and then pushing that container to a registry which Azure Container Instances or Kubernetes can then use. Here's how I've approached this, with some concrete examples from prior projects.

First, we need to containerize the Flask application. That involves creating a Dockerfile. The Dockerfile will specify the base image, copy our application code, install necessary dependencies, and define the command to run when the container starts. I remember one project where we had a rather tricky dependency conflict, and we had to carefully pin specific library versions in the `requirements.txt` file to ensure the build process was reproducible.

Here's a straightforward example of a Dockerfile for a basic Flask application:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to handle debug mode in flask
ENV FLASK_APP app.py
ENV FLASK_ENV production

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
```

In this Dockerfile, we begin with a lean Python 3.9 base image. The `WORKDIR` instruction sets the working directory inside the container. We then copy the `requirements.txt` and install dependencies, followed by copying the rest of the application files. Finally, we expose port 5000, which is the default port for Flask, and we specify the command to start the Flask application. Setting `FLASK_ENV` to production will prevent debug mode which is important for a production environment.

Next, we need a `requirements.txt` file that lists all the Python packages our Flask app needs. For this example, let’s assume our application requires Flask:

```
flask
```

With these two files in place, we can build our Docker image. In the same directory as your Dockerfile, run the following command in your terminal:

```bash
docker build -t my-flask-app .
```

This command tells Docker to build an image named `my-flask-app` using the Dockerfile in the current directory. The `.` at the end specifies the build context. Now we have a Docker image ready to go, and we can test locally to make sure everything works.

To run your application locally, use:

```bash
docker run -p 5000:5000 my-flask-app
```

This command runs the `my-flask-app` image and maps port 5000 on your host machine to port 5000 inside the container. This allows you to access the application in your browser by navigating to `http://localhost:5000`.

Now, let's push this image to Azure Container Registry. First, make sure you have an ACR instance created within your Azure subscription. Assuming you've already set up the ACR instance named `myacr`, you'll need to log in using the Azure CLI:

```bash
az acr login --name myacr
```

This will prompt you to authenticate. Once logged in, we need to tag the image with the fully qualified ACR image name and then push the tagged image. This is important; ACR needs an image name that includes your registry details.

```bash
docker tag my-flask-app myacr.azurecr.io/my-flask-app:v1
docker push myacr.azurecr.io/my-flask-app:v1
```

In these commands, we’re tagging the local `my-flask-app` image with the registry-specific name `myacr.azurecr.io/my-flask-app:v1`. The `:v1` part represents the tag – this allows you to manage different versions of your container images. Then, we push the tagged image to ACR.

Let’s consider a slightly more involved scenario where the Flask app also requires a database connection, say Postgres. In this case, your `requirements.txt` would include `psycopg2-binary` (or `psycopg2`), and the Dockerfile would need an additional configuration to ensure the necessary database driver is available. You might also need environment variables to manage connection strings.

Here's an updated Dockerfile example, assuming you need some environment variables. Note that securely managing secrets like database passwords is crucial, but let's stick with simple environment variables for illustrative purposes for now:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Environment variables for database connection
ENV DATABASE_URL="postgresql://user:password@host:5432/dbname"
ENV FLASK_APP app.py
ENV FLASK_ENV production

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]

```

With that more complex example, we also need to adjust `requirements.txt`:

```
flask
psycopg2-binary
```

Now the build, tag and push process is the same, just with the newly changed files and requirements.txt.

```bash
docker build -t my-flask-app .
docker tag my-flask-app myacr.azurecr.io/my-flask-app:v2
docker push myacr.azurecr.io/my-flask-app:v2
```

The updated image named `my-flask-app:v2` now contains support for your Postgres database connection and is pushed to ACR as well. It’s important to note that versioning with tags is best practice and allows you to revert to earlier images if necessary.

Finally, to consume the image from the ACR, you'd typically use it in either Azure Container Instances (ACI) or Kubernetes. You would configure either to pull the image `myacr.azurecr.io/my-flask-app:v1` (or `:v2`) from the registry and run it. The configuration needed for this varies depending on the platform you choose, but the core process always involves referencing the registry name and image tag.

For a deeper dive, I'd recommend looking into books like "Docker in Action" by Jeff Nickoloff and "Kubernetes in Action" by Marko Lukša. Also, the official Docker and Kubernetes documentation are indispensable resources. For specifics on Azure, Microsoft’s official documentation on Azure Container Registry and Azure Container Instances are excellent starting points. Understanding containerization principles, building efficient Docker images, and correctly leveraging the capabilities of your chosen deployment platform are essential for a successful deployment.

This whole process, while detailed, becomes second nature with practice. Each project provides unique nuances and learnings, and that continuous iterative improvement is what pushes your tech skills forward. Always prioritize clear, concise code, efficient container images, and robust deployment strategies.
