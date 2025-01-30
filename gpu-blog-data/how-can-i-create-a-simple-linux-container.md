---
title: "How can I create a simple Linux container image without a permanently running application in Kubernetes?"
date: "2025-01-30"
id: "how-can-i-create-a-simple-linux-container"
---
Linux container images are primarily designed to encapsulate an application and its dependencies. However, their utility extends beyond perpetually running services. I've found, from experience building various CI/CD pipelines, that creating container images that perform a single, non-persistent task is incredibly valuable. This approach enables efficient and isolated execution of short-lived processes within a Kubernetes environment, without the overhead of a constantly running daemon.

The key to achieving this lies in understanding how the container’s entrypoint interacts with its intended process. The entrypoint, specified in the Dockerfile, defines the executable that runs when a container is started. When this entrypoint’s process completes, the container itself exits. By providing a script or command that performs the desired single task, and letting it finish, we can leverage Kubernetes to orchestrate these one-off jobs without requiring a long-running application within the image.

Essentially, the container image becomes a packaged execution environment for a discrete task. Think of it as a highly portable and repeatable script executor. This pattern is particularly useful for batch processing, data transformations, migrations, and other similar activities where a persistent service isn't needed. Crucially, Kubernetes handles the scheduling and resource allocation while the container manages the execution context.

The fundamental construct is a simple Dockerfile that specifies the necessary base image, copies any required scripts or binaries, and sets an entrypoint that executes this logic. Let me illustrate with a few examples.

**Example 1: Simple Data Extraction**

This example demonstrates a container image designed to extract data from a file and print the result to standard output. The container is not intended to stay alive beyond this operation.

*Dockerfile*

```dockerfile
FROM alpine:latest

WORKDIR /app

COPY extract_data.sh .

ENTRYPOINT ["/app/extract_data.sh"]
```

*extract\_data.sh*

```bash
#!/bin/sh

if [ -f data.txt ]; then
  grep "important" data.txt
else
  echo "data.txt not found." >&2
  exit 1
fi
```

*data.txt*

```
This is some data.
This line is not important.
This one is important.
Another line of data.
```

**Commentary:**

*   `FROM alpine:latest`: Uses a minimal Linux base image for a small footprint.
*   `WORKDIR /app`: Sets the working directory inside the container.
*   `COPY extract_data.sh .`: Copies the script into the working directory.
*   `ENTRYPOINT ["/app/extract_data.sh"]`: Defines the script as the executable to run when the container starts.
*   The shell script `extract_data.sh` searches for a pattern within `data.txt`.
*   Error handling prevents the script from running indefinitely if `data.txt` is absent.
*   The `>&2` redirects error messages to standard error.
*   The exit code allows Kubernetes to recognize the job's completion status.

To use this image within Kubernetes, a Job resource is employed. Once the container runs, executes the script, and exits, the Kubernetes Job marks itself as complete, effectively disposing of the pod.

**Example 2: Data Transformation with Python**

Here, the container image transforms data utilizing Python. The script itself is very straightforward.

*Dockerfile*

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY transform_data.py .
COPY input.json .

RUN pip install jsonpath-ng

ENTRYPOINT ["python", "/app/transform_data.py"]
```

*transform\_data.py*

```python
import json
from jsonpath_ng import jsonpath, parse
import sys

try:
    with open('input.json', 'r') as f:
        data = json.load(f)

    jsonpath_expression = parse('$.items[*].value')
    values = [match.value for match in jsonpath_expression.find(data)]

    for value in values:
        print(value)
except FileNotFoundError:
    print("input.json not found.", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError:
    print("Error decoding JSON file.", file=sys.stderr)
    sys.exit(1)
```

*input.json*

```json
{
  "items": [
    {"id": 1, "value": "abc"},
    {"id": 2, "value": "def"},
    {"id": 3, "value": "ghi"}
  ]
}
```

**Commentary:**

*   `FROM python:3.9-slim-buster`: Uses a minimal Python base image.
*   `RUN pip install jsonpath-ng`: Installs the necessary Python library.
*   The Python script `transform_data.py` reads JSON data, extracts specific values using `jsonpath-ng`, and prints each value.
*   Error handling includes checks for file existence and JSON decoding errors.
*   This example illustrates how a non-interactive process can be embedded within a container image to perform specific tasks within a Kubernetes job.

The `pip` installation is performed within the build stage of the image, ensuring a complete runtime environment.

**Example 3: Simple Database Migration**

This image executes database migrations using the `flyway` tool, a command line tool for schema changes.

*Dockerfile*

```dockerfile
FROM flyway/flyway:latest

WORKDIR /flyway/sql

COPY migrations ./migrations

ENTRYPOINT ["flyway", "migrate"]
```

*./migrations/V1__initial_schema.sql*

```sql
CREATE TABLE IF NOT EXISTS my_table (
    id INT PRIMARY KEY,
    value VARCHAR(255)
);
```

*./migrations/V2__insert_test_data.sql*

```sql
INSERT INTO my_table (id, value) VALUES (1, 'Test value');
```

**Commentary:**

*   `FROM flyway/flyway:latest`: Uses an image with the pre-installed flyway tool.
*   `COPY migrations ./migrations`: Copies the migration scripts to the appropriate directory.
*   `ENTRYPOINT ["flyway", "migrate"]`: Executes the `flyway migrate` command to apply the pending database migrations.
*   This demonstrates a slightly more complex task - applying schema changes.
*   `flyway` reads the configuration (not included in this example) to locate the database credentials and connect to the server to perform the migration.

In this case, the container image performs database operations, again emphasizing its non-interactive and task-oriented nature. The user provides SQL migrations and lets flyway execute them within the isolated environment.

For container image development, there are a number of resources available including documentation on Dockerfile syntax, and guides to the specific tooling used, for instance Flyway. For Kubernetes specifics, consult the official Kubernetes documentation and examples for interacting with Jobs and CronJobs. The best way to learn these skills, however, is via direct hands-on experience, practicing with various combinations of images and scripts. These resources will provide a solid foundation for understanding how to structure single-task container images in a Kubernetes context.
