---
title: "How can I access a JSON file from a shell script within a Docker container?"
date: "2024-12-23"
id: "how-can-i-access-a-json-file-from-a-shell-script-within-a-docker-container"
---

Alright, let's tackle this one. I've bumped into this exact scenario multiple times during various deployments – the need to pull data from a json configuration file located inside a docker container, directly from within a shell script *running* inside that same container. It’s more common than you might initially think, especially with microservice architectures where configuration is often externalized.

The core challenge revolves around how your shell script, executing inside the container's environment, can reliably access and parse a file sitting in the container’s filesystem. The solution, essentially, boils down to combining a couple of standard shell commands. It isn’t complicated, but it's important to approach it methodically.

First and foremost, we need to *locate* the json file. This is straightforward if you know the file's path inside the container. Let’s assume, for argument’s sake, that our json file is `config.json` and it resides at `/app/config/`. We are not going to consider the case where the file is in an unknown location, as this should never happen in a well-designed system and would require the use of `find` or `locate`, which is not recommended in production environments.

Next, we need a way to *read* the json file’s contents into a variable that our shell script can then use. This is where `cat` comes into play: it outputs the entire file's contents to standard output. Now, we need a way to parse that output in a structured manner. We are not going to consider text-based parsing (like `grep` or similar) since the JSON format has a specific structure and we need something that respects that. The tool of choice here is `jq`, which is a lightweight and flexible command-line JSON processor.

**The Key: `cat` and `jq`**

Combining `cat` and `jq` allows us to efficiently read the file and then selectively extract data.

Here is a basic example demonstrating extracting a single scalar value:

```bash
#!/bin/bash

# First, set the file path, for clarity and maintainability
CONFIG_FILE="/app/config/config.json"

# Extract a specific property: 'api_endpoint' from the json file
API_ENDPOINT=$(cat "$CONFIG_FILE" | jq -r '.api_endpoint')

# Use the variable
echo "API Endpoint: $API_ENDPOINT"

# Example JSON in the file:
# {
#  "api_endpoint": "https://api.example.com/v1",
#  "port": 8080,
#  "logging": {
#    "level": "info",
#    "destination": "/var/log/app.log"
#  }
# }

```

In this example, the `cat "$CONFIG_FILE"` command outputs the json file content, the `|` pipes that output to `jq`'s input. The `jq -r '.api_endpoint'` part specifies that we want the value associated with the key `api_endpoint`, and the `-r` option tells `jq` to output the raw (unquoted) value. The result of all this is then stored in the variable `API_ENDPOINT`. This allows you to subsequently work with the value within your script.

Let’s consider a slightly more complex scenario. Suppose that the json structure contains nested objects, and we are interested in grabbing a property inside one of those objects:

```bash
#!/bin/bash

# Same path as before
CONFIG_FILE="/app/config/config.json"

# Extract a nested property: 'logging.level' from the json file
LOG_LEVEL=$(cat "$CONFIG_FILE" | jq -r '.logging.level')

# Use the variable
echo "Log Level: $LOG_LEVEL"


# Example JSON in the file:
# {
#  "api_endpoint": "https://api.example.com/v1",
#  "port": 8080,
#  "logging": {
#    "level": "info",
#    "destination": "/var/log/app.log"
#  }
# }

```

Here, we access the nested property `logging.level` by using a dot operator inside the jq expression `'.logging.level'`. This is a very common use case when dealing with structured configuration.

Finally, what if we need an array? We can select a whole array or part of an array and process the data accordingly. Imagine our config contains an array of allowed hosts:

```bash
#!/bin/bash

CONFIG_FILE="/app/config/config.json"

# Extract an array property: 'allowed_hosts' from the json file
ALLOWED_HOSTS=$(cat "$CONFIG_FILE" | jq -r '.allowed_hosts[]')

# Use the variable and print each host in a separate line
echo "Allowed Hosts:"
while IFS= read -r host; do
  echo "- $host"
done <<< "$ALLOWED_HOSTS"

# Example JSON in the file:
# {
#   "api_endpoint": "https://api.example.com/v1",
#   "port": 8080,
#  "allowed_hosts": [
#      "host1.example.com",
#      "host2.example.net",
#     "127.0.0.1"
#   ]
# }

```

In this case, `jq -r '.allowed_hosts[]'` will output each host in the `allowed_hosts` array on a new line. We then iterate through this list using a `while` loop and print each host. This demonstrates how to handle collections or arrays of data extracted from json using `jq`.

**Important Considerations:**

*   **Error Handling:** In production scenarios, you should incorporate error handling. You can check the return code of `jq` (`$?`) to see if it successfully parsed the json, and if not, take an appropriate action. `jq` will return a non-zero exit code when an error occurs, like the key not being found in the document.
*   **`jq` Availability:** Ensure that `jq` is installed in your docker container. This is usually done via the dockerfile using package manager (e.g. `apk add jq` for alpine images or `apt-get install jq` for debian-based images).
*   **Complex JSON:** While the provided examples are simple, `jq` can handle significantly more complex json structures with filters, complex paths, and other manipulations. I'd advise reading the `jq` manual directly; it is extremely well-documented.
*   **Security:** Be mindful of security. If you're obtaining the file or specific values from an external source, be absolutely certain to sanitise the data before using it. Also, avoid storing sensitive information like passwords directly in your JSON config files. Using environment variables, secret managers, or dedicated configuration stores is considered best practice.

**Recommended Reading:**

*   **The `jq` Manual:** The official documentation of `jq` is essential. I suggest starting by downloading it directly: `man jq` in your terminal once you have it installed or consulting online versions.
*   **"The Linux Command Line" by William Shotts:** This book provides a comprehensive guide to shell scripting and command-line tools, including `cat`, pipes and redirects. This knowledge will provide a strong base understanding when using tools such as `jq`.
*   **"Docker Deep Dive" by Nigel Poulton:** If you are not an expert with Docker, this book helps to understand how to set up containers and related concepts. This is crucial since we are talking about a docker container environment.

In summary, accessing a json file from a shell script within a Docker container is straightforward if you utilize the right tools effectively. `cat` gets the content, and `jq` is your primary tool for the selection and parsing. Always consider error handling, security, and ensure `jq` is available in your container. With a little practice, this becomes a routine task. From my own experiences, mastering these simple but powerful tools has made my deployments more robust and manageable.
