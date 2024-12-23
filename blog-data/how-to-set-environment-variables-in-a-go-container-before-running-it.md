---
title: "How to set environment variables in a Go container before running it?"
date: "2024-12-23"
id: "how-to-set-environment-variables-in-a-go-container-before-running-it"
---

Alright, let's tackle environment variables in go containers. This is a topic I’ve bumped into numerous times over the years, often during deployments where nuanced configurations were crucial. The goal, as always, is to ensure our go applications behave as expected across different environments, without recompiling or altering code.

From my experience, there are several robust approaches to injecting environment variables into a go container before it launches, and I've learned firsthand that understanding these methods is crucial for smooth deployments, particularly in microservices architectures where each service often requires unique environmental settings.

The most common, and arguably the simplest method, is using the `-e` flag with the `docker run` command. I remember one particular project where we had multiple staging environments, each with different database connection strings and api keys. Using docker run with individual -e parameters for each environment became quite cumbersome, but it's an effective starting point. Here's how that looks:

```bash
docker run -d \
    -e "DATABASE_URL=postgres://user:password@host:port/database" \
    -e "API_KEY=your_secret_key" \
    -e "DEBUG_MODE=true" \
    your-go-image:latest
```

In this example, `DATABASE_URL`, `API_KEY`, and `DEBUG_MODE` are being passed into the container’s environment. The downside here is that as you start scaling the number of environment variables, this command becomes lengthy and more error prone to manage. Also, it lacks reusability. Copying this full command with modifications between deployments is not exactly best practice. I've seen this lead to subtle misconfigurations causing unexpected behavior in deployed environments.

To address this, I often turn to environment files. These are just plain text files, typically named `.env`, where you can define your variables in a key-value format. This keeps your docker run command cleaner and offers a single source for your environmental configurations. Consider this a more evolved approach compared to inline `-e` variables. Here’s what an example `.env` file might look like:

```
DATABASE_URL=postgres://user:password@host:port/database
API_KEY=your_secret_key
DEBUG_MODE=true
LOG_LEVEL=info
```

You then load these using docker's `--env-file` flag:

```bash
docker run -d --env-file .env your-go-image:latest
```

This approach provides better manageability and makes it simpler to adjust your environment configurations between deployments. I've found this method extremely useful when managing configurations across different docker-compose files too, and have generally found it to lead to more maintainable infrastructure. You’ll see many teams use this in their everyday deployments. However, a problem still persists: secrets management. You don’t want to commit sensitive data directly into an env file in your code repository.

The next approach, and one that I’ve found incredibly beneficial, uses docker secrets. Docker secrets allow for secure handling of sensitive information without exposing them directly in your image or command line. Secrets are stored by Docker and are injected as files within the container, usually at `/run/secrets/`, and are then read by your application. I’ve implemented this in several projects requiring database credentials and api keys, as I mentioned earlier. This was incredibly helpful for mitigating the risk of leaking secrets.

To use secrets, you first need to create the secret within the Docker environment. For instance, let's create a secret named `database_password` containing the actual password.

```bash
echo "your_real_database_password" | docker secret create database_password -
```

Now, we'll modify our docker run command, adding the `--secret` flag. Note that we are not passing the value directly, just the secret name.

```bash
docker run -d --secret database_password \
    your-go-image:latest
```

Your go application would need to be altered to read the password from `/run/secrets/database_password`. The following go snippet illustrates how you’d accomplish that.

```go
package main

import (
	"fmt"
    "os"
    "io/ioutil"
	"log"
	"strings"
)

func main() {
	// Load from /run/secrets
    dbPassword, err := loadSecret("/run/secrets/database_password")
	if err != nil {
		log.Printf("Error loading database password from secret: %v", err)
        dbPassword = os.Getenv("DATABASE_PASSWORD")  // fallback to environment variable
        if dbPassword == "" {
            log.Fatal("DATABASE_PASSWORD not provided")
        }
    }

    // Fallback to environment variables, if needed
	apiKey := os.Getenv("API_KEY")
	debugMode := os.Getenv("DEBUG_MODE")
    logLevel := os.Getenv("LOG_LEVEL")


	fmt.Printf("Database password: %s\n", strings.Repeat("*", len(dbPassword))) // Print masked password
	fmt.Printf("Api key: %s\n", apiKey)
	fmt.Printf("Debug mode: %s\n", debugMode)
    fmt.Printf("Log level: %s\n", logLevel)

}

// Loads secret from file
func loadSecret(path string) (string, error) {
    content, err := ioutil.ReadFile(path)
    if err != nil {
        return "", fmt.Errorf("could not read secret file: %w", err)
    }
    return strings.TrimSpace(string(content)), nil
}

```

In the provided Go code, the application attempts to read `database_password` first from a file path, `/run/secrets/database_password`, then it falls back to an environment variable if it is not present. The rest of the environment variables, are read using `os.Getenv()` as you would normally do in Go. This combined approach allows you to load secrets securely when using docker secrets, and seamlessly fallback to other sources of environment variables if needed.

As for recommended reading, I would highly suggest starting with "Docker in Action" by Jeff Nickoloff. This book provides a deep dive into docker concepts including secrets and environment variable handling. Additionally, the official Docker documentation remains an invaluable resource for staying updated with the latest best practices. Specifically, the section on docker secrets and the `docker run` command documentation are essential reads. Finally, "Cloud Native Patterns" by Cornelia Davis offers high-level architectural patterns and considerations when working with containers, which can inform your strategies about deployment and environment configuration. These texts should give you a comprehensive understanding to make more informed decisions regarding setting up your container environment.

The key takeaway is that there isn't one single best method. The ideal approach depends largely on the project's needs, security requirements, and team workflows. But mastering these methods and having a clear understanding of their respective trade-offs will save you countless hours and potential deployment issues down the road, which, trust me, is invaluable.
