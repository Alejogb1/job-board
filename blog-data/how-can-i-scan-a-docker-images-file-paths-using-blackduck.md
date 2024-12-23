---
title: "How can I scan a Docker image's file paths using BlackDuck?"
date: "2024-12-23"
id: "how-can-i-scan-a-docker-images-file-paths-using-blackduck"
---

Alright, let’s tackle this. I’ve spent considerable time working with Black Duck and container security, so I can shed some light on how to effectively scan Docker image file paths. It's not as straightforward as, say, scanning a source code repository, but it's entirely achievable and crucial for maintaining a secure application environment.

The fundamental idea when working with container images in Black Duck is that you're not directly looking at the filesystem as you would on a running system. Instead, you’re analyzing the image layers and their content. Black Duck, when performing its scan, breaks down the image into these layers and identifies the components present within. The 'file path' information you’re after, therefore, comes as a byproduct of this analysis, not as a primary input.

My experience with this began when I was tasked with auditing a complex microservices deployment. We were dealing with multiple layers deep in some images, and it was imperative to understand not just the components we’d explicitly included, but also any unexpected files lurking in base images or introduced by third-party libraries. Finding these via manual inspection is, frankly, a nightmare.

Now, let’s talk specifics. Black Duck uses a combination of signatures, hash matching, and component analysis techniques to identify the software components present in your image. During this process, it logs not only the identified components and their versions but also the locations (file paths) where these components are found within the image's file system. This information is accessible in the Black Duck user interface after a scan completes.

The first practical step is to ensure you’re using the appropriate Black Duck integration method. Ideally, you’d use a CI/CD pipeline integration that allows you to scan your Docker images automatically. The `synopsys-detect` command-line tool is your friend here. It’s the workhorse for Black Duck scans and supports Docker image scanning through image archives or directly from a Docker registry. Let me show you an example of scanning a Docker image from a local archive:

```bash
detect.sh \
  --blackduck.url=<your_blackduck_url> \
  --blackduck.api.token=<your_api_token> \
  --detect.project.name="MyProject" \
  --detect.project.version.name="1.0.0" \
  --detect.docker.image=<path_to_your_docker_image.tar.gz> \
  --detect.docker.image.tar=true \
  --detect.tools.included=DOCKER
```

In this example, the key element is `--detect.docker.image=<path_to_your_docker_image.tar.gz>` and `--detect.docker.image.tar=true`. This instructs `synopsys-detect` to treat the specified file as a Docker image archive and to scan it. Once the scan is complete, you’ll find the file path data within the scan results. Specifically, if you drill down into the identified components you should see their associated locations. This provides the contextual insight you’re after.

However, relying solely on the web interface for deep inspection can be cumbersome when dealing with many files or when you need programmatic access. This brings us to Black Duck's API. The API allows you to retrieve scan data, including file paths, programmatically. This is particularly useful for building your own custom reporting or automated workflows. Let's consider retrieving file paths programmatically using python.

```python
import requests
import json

def get_scan_results(blackduck_url, api_token, project_name, version_name):
    headers = {'Authorization': f'Bearer {api_token}', 'Accept': 'application/vnd.blackducksoftware.scan-v1+json'}
    project_url = f'{blackduck_url}/api/projects'
    params = {'q': f'name:"{project_name}"'}
    response = requests.get(project_url, headers=headers, params=params)
    response.raise_for_status()
    projects = response.json()['items']

    if not projects:
        raise Exception(f"Project '{project_name}' not found.")
    project_id = projects[0]['id']

    version_url = f'{blackduck_url}/api/projects/{project_id}/versions'
    params = {'q': f'versionName:"{version_name}"'}
    response = requests.get(version_url, headers=headers, params=params)
    response.raise_for_status()
    versions = response.json()['items']
    if not versions:
        raise Exception(f"Version '{version_name}' not found for project '{project_name}'.")
    version_id = versions[0]['id']

    components_url = f'{blackduck_url}/api/projects/{project_id}/versions/{version_id}/components'
    response = requests.get(components_url, headers=headers)
    response.raise_for_status()

    for component in response.json()['items']:
        print(f"Component: {component['name']}")
        if "origins" in component:
            for origin in component["origins"]:
              if "path" in origin:
                print(f"  Path: {origin['path']}")

if __name__ == '__main__':
  blackduck_url = "<your_blackduck_url>"
  api_token = "<your_api_token>"
  project_name = "MyProject"
  version_name = "1.0.0"

  get_scan_results(blackduck_url, api_token, project_name, version_name)
```

This snippet first retrieves the project and version IDs, then iterates through the components identified, printing out their name and associated file path when present. This requires some familiarity with Black Duck's API structure but allows very granular extraction of scan data. This is a simplified example; you'd likely need to add error handling and pagination, especially with large scans.

Finally, if you’re dealing with more complex scenarios, such as scanning images residing in a private container registry or needing more detailed filtering, you can refine your `synopsys-detect` command and utilize the API to query the scan results even further.

Consider this variation, which adds an image pull directly from a private registry and focuses on scanning specific components:

```bash
detect.sh \
  --blackduck.url=<your_blackduck_url> \
  --blackduck.api.token=<your_api_token> \
  --detect.project.name="MyProject" \
  --detect.project.version.name="1.0.0" \
  --detect.docker.image=<your_registry>/<your_image>:<your_tag> \
  --detect.docker.image.pull.credentials.username=<username> \
  --detect.docker.image.pull.credentials.password=<password> \
  --detect.tools.included=DOCKER
  --detect.bom.aggregate.remediation.mode=false \
  --detect.dependency.type.enabled=COMPONENT,FILE,PACKAGE
```

Here, we're using `--detect.docker.image` to specify the location in the registry directly and providing authentication details to access it. The additional options such as `--detect.bom.aggregate.remediation.mode=false` and `--detect.dependency.type.enabled=COMPONENT,FILE,PACKAGE` help to refine the components and their details that we are interested in.

For additional information, I’d recommend consulting the following:

*   **Synopsys Detect documentation:** Specifically, the section covering Docker image scanning parameters is essential. The documentation is typically found in their help center or support portal.
*   **The Black Duck API documentation:** It’s indispensable for programmatically accessing and manipulating scan data. Also available via Synopsys support channels.
*   **"Continuous Security in Practice: Automating DevOps Security"** by Jennifer Davis and Matthew Duft. While not Black Duck specific, it covers the principles and practices of integrating security into pipelines, which is very relevant.

In short, while Black Duck doesn't directly expose filesystem browsing, its component analysis process indirectly provides this via the 'path' information associated with detected components. Leveraging the right tools like `synopsys-detect` and the API makes this information easily accessible and applicable to real-world security analysis and automated workflows. The key is to understand the underlying process and how to access the information effectively after scans have completed. I hope this clarifies the process.
