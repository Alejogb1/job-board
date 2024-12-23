---
title: "How can I validate YAML files on GitHub?"
date: "2024-12-23"
id: "how-can-i-validate-yaml-files-on-github"
---

Alright, let's talk about YAML validation on GitHub. I’ve spent a fair amount of time navigating the intricacies of configuration management, and invalid YAML slipping through is, let's be honest, a headache we all try to avoid. The crux of the matter isn't just about preventing deployments from failing; it's about ensuring consistency, predictability, and maintainability. Having dealt with that firsthand – a particularly memorable incident involving a missing colon and a multi-hour debugging session – i can confidently say that proactive validation is key.

The challenge isn't whether validation *can* be done, but rather how to incorporate it seamlessly into your GitHub workflow. We're essentially aiming for a system that automatically flags invalid YAML files, ideally before they even merge into the main branch. This isn't just about finding syntax errors, although that's crucial. We need to consider logical errors as well, where the YAML might be syntactically valid but semantically incorrect for your application.

Now, let's get into how to make this happen. The most effective approach involves leveraging GitHub Actions. This allows us to set up continuous integration (CI) pipelines that execute checks whenever changes are pushed to a repository or a pull request is created. We'll need two main components: a YAML validator and a GitHub Actions workflow to run it. There are several validation tools out there, but i often fall back on a combination of `yamllint` for general syntax checks and custom schema validation, particularly when dealing with complex configurations.

Here’s how I'd break down the process:

1.  **Choose Your Validator(s):** `yamllint` is a powerful linter that goes beyond simple syntax checks; it enforces style guidelines and best practices. There are others, but for the purpose of illustration, we'll focus on this, as it is relatively straightforward to use and readily available. Additionally, when application logic requires it, custom schema validation should be a consideration, which, while more involved, is crucial for complex configurations.

2.  **Implement a GitHub Actions Workflow:** This is where we define the steps to execute our validation. We'll set up a workflow that runs on push and pull requests. It will install `yamllint`, find the yaml files, and validate them. We can also add a step to handle the custom schema validation, which often relies on a defined schema document that describes the expected content, structure and datatype within the YAML files.

3.  **Action on Validation Failures:** When errors are detected, the action should fail the workflow, preventing the changes from being merged. Ideally, the workflow outputs specific error messages and line numbers in the GitHub UI for ease of correction.

Let’s look at some code snippets that illustrate this process, specifically focusing on utilizing `yamllint` and including an example of a simple custom schema validation.

**Snippet 1: Basic GitHub Actions Workflow with `yamllint`**

```yaml
name: YAML Validation
on:
  push:
    branches: [main]
  pull_request:
jobs:
  yaml-validation:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
            python-version: '3.x'

      - name: Install yamllint
        run: pip install yamllint

      - name: Validate YAML files
        run: |
          find . -name "*.yaml" -o -name "*.yml" | while read file; do
            yamllint "$file"
            if [ $? -ne 0 ]; then
                echo "YAML validation failed in $file"
                exit 1
            fi
          done

```

This workflow first checks out the code, then sets up Python (as `yamllint` is python-based). Next, it installs `yamllint` and searches for all files ending in .yaml or .yml. It then runs `yamllint` on each file. If `yamllint` returns a non-zero exit code, meaning an error was found, the workflow will exit, effectively marking the build as failed.

**Snippet 2: Adding Custom Schema Validation (Simplified Example)**

For our custom schema, let's pretend we have a very specific configuration format expected in a file called `config.yaml`. We expect it to have at least a version, name, and a list of servers, each with an address and port.

First, let’s add a python script to validate against this schema

```python
import yaml
import sys

def validate_config(file_path):
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
         print(f"Error: File not found {file_path}")
         sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format: {e}")
        sys.exit(1)

    if not isinstance(config, dict):
        print("Error: YAML should be a dictionary.")
        sys.exit(1)

    if 'version' not in config or not isinstance(config['version'], str):
        print("Error: 'version' should be a string field.")
        sys.exit(1)

    if 'name' not in config or not isinstance(config['name'], str):
        print("Error: 'name' should be a string field")
        sys.exit(1)
    if 'servers' not in config or not isinstance(config['servers'], list):
        print("Error: 'servers' should be a list.")
        sys.exit(1)

    for server in config['servers']:
        if not isinstance(server, dict):
            print("Error: Each server should be a dictionary")
            sys.exit(1)
        if 'address' not in server or not isinstance(server['address'], str):
            print("Error: Server 'address' should be a string.")
            sys.exit(1)
        if 'port' not in server or not isinstance(server['port'], int):
            print("Error: Server 'port' should be an integer.")
            sys.exit(1)

    print("Config is valid.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validator.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    validate_config(config_file)

```

Now let's adjust the action workflow to utilize this custom validator.

**Snippet 3: Updated Workflow with Custom Schema Check**

```yaml
name: YAML Validation
on:
  push:
    branches: [main]
  pull_request:
jobs:
  yaml-validation:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'

      - name: Install yamllint
        run: pip install yamllint

      - name: Validate YAML files with yamllint
        run: |
          find . -name "*.yaml" -o -name "*.yml" | while read file; do
            yamllint "$file"
            if [ $? -ne 0 ]; then
              echo "YAML validation failed with yamllint in $file"
              exit 1
            fi
          done

      - name: Validate config.yaml with custom validation
        run: python validator.py config.yaml
```

In this updated workflow, we’ve added a new step to run the custom `validator.py` script specifically against `config.yaml`. If the script finds any errors, it will exit with a non-zero code, causing the action to fail.

These snippets are just the starting point, of course. In a real project, the schema validation might get more complex, involving libraries for more rigorous schema enforcement.

For resources on delving deeper, I'd recommend looking at:
- **"Effective DevOps" by Jennifer Davis and Katherine Daniels**: This provides a great overview of automation and infrastructure as code best practices, including the role of validation.
- **The official yamllint documentation**: For understanding all the possibilities offered by `yamllint`, including extensive configurations.
- **"Designing Data-Intensive Applications" by Martin Kleppmann:** Although broader in scope, it covers crucial concepts related to data formats and schema, vital for understanding why strict validation is important.
- **The JSON Schema website**: This is not yaml specific, but provides a standard format to define data schemas which can be used with validators in python or other languages to enforce structure and data types.
- Specific documentation on libraries that allow schema validations in python such as: `cerberus` or `jsonschema`.

In closing, validating YAML files on GitHub isn't merely an optional task; it's a fundamental aspect of building reliable and maintainable systems. By incorporating these steps into your workflow, you reduce the risk of misconfigurations, prevent costly errors, and ensure the overall stability of your projects. The example approach we covered provides a starting point, but the key takeaway is to proactively validate every time, regardless of the workflow details.
