---
title: "How do I identify the author of a pushed container image in GitLab?"
date: "2025-01-30"
id: "how-do-i-identify-the-author-of-a"
---
Container image authorship, within the context of GitLab's Container Registry, is not directly embedded into the image itself; instead, it is derived from the context of the pipeline or interaction that pushes the image to the registry. I’ve encountered situations where teams presumed this metadata existed within the image, leading to confusion during incident analysis. Understanding the mechanisms GitLab provides for identifying authors is crucial for robust traceability and security.

GitLab’s Container Registry doesn’t store the identity of the *person* pushing the image, but rather the *pipeline* or *user* executing the push operation. The distinction is important. A pipeline push is attributed to the *CI user* associated with that pipeline. Manual pushes, performed via `docker push` (or similar) using a GitLab Personal Access Token (PAT) or Deploy Token are attributed to that specific PAT or Deploy Token. Therefore, tracing authorship necessitates looking at either the pipeline history or the token usage.

Here’s a more detailed breakdown:

**Pipeline-based Pushes:**

When a pipeline pushes an image, GitLab records the pipeline ID, the commit SHA, the branch name, and the user who initiated the pipeline. This information is viewable within the GitLab UI for that specific pipeline run. To access this, navigate to the pipeline that built and pushed the image. Look for the “Jobs” tab; the specific job responsible for the push operation will typically use either `docker push`, `podman push`, or equivalent commands. The job details, including the author of the triggering commit, are crucial for establishing a link between the image and a specific developer.

**Token-based Pushes:**

When an image is pushed using a PAT or a deploy token, GitLab records the token’s associated user or deploy token information. To track these instances, the user has to audit their PATs or Deploy Tokens. Monitoring token usage is crucial; tokens with excessive permissions or shared among multiple developers can make identification exceedingly difficult. Furthermore, there is no direct link within the GitLab UI that connects token-based push events to image tags or repositories. This limitation reinforces the need for diligent security practices surrounding token management.

Now, consider these code examples in the context of a GitLab CI/CD pipeline:

**Example 1: Standard Docker Push**

```yaml
build-image:
  image: docker:stable
  stage: build
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

*   **Commentary:** This snippet illustrates a common Docker push. The crucial variables, `$CI_REGISTRY_USER` and `$CI_REGISTRY_PASSWORD`, are GitLab-provided environment variables representing the CI user for the project. The pushed image, tagged with the commit SHA, becomes linked to this specific pipeline run. The author, in this case, is indirectly the person who triggered this pipeline run through a push to the repo. Looking up the pipeline details will unveil the initiating user. The image tag `CI_COMMIT_SHA` provides additional traceability linking to a specific commit author.

**Example 2: Pushing with a Custom Tag**

```yaml
build-image:
  image: docker:stable
  stage: build
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:latest .
    - docker tag $CI_REGISTRY_IMAGE:latest $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG
    - docker push $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG
```

*   **Commentary:** In this instance, we push both with the `latest` tag and a branch-specific tag. This practice allows for simpler identification of the commit by the branch name, using the `$CI_COMMIT_REF_SLUG` variable, which corresponds to the branch name. The `latest` tag is often problematic from a traceability perspective because it can be overwritten by different pipelines, making its authorship ambiguous. Tracking the individual pipeline is crucial to pinpoint the user. Despite the dual push, the authorship is still tied to the CI user for that specific pipeline execution.

**Example 3: Using a Specific Deploy Token**

```yaml
build-image:
  image: docker:stable
  stage: build
  services:
    - docker:dind
  script:
    - echo "$DEPLOY_TOKEN_PASSWORD" | docker login -u $DEPLOY_TOKEN_USER --password-stdin $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

*   **Commentary:** Here, a `DEPLOY_TOKEN_USER` and `DEPLOY_TOKEN_PASSWORD` are used for authentication instead of the CI user. The `DEPLOY_TOKEN_*` variables are assumed to be declared as masked CI/CD variables within the project.  The authorship here, contrary to the prior examples, is *not* linked directly to the pipeline's author but to the *deploy token* itself. To identify the author in this case, one would have to examine the configured deploy tokens and identify which individual is associated with the token. This highlights the importance of rotating tokens regularly and enforcing strict usage policies.

To summarize, authorship isn’t embedded in the container image itself, but derived from the pushing context. This context can be either:

1.  **The CI Pipeline User:**  Identifiable through GitLab's pipeline interface, where commit and pipeline initiation details are available.
2.  **A Personal Access Token (PAT) User:** Where the user tied to the PAT can be determined by inspecting the token.
3.  **A Deploy Token:** Where the user or service account linked to the Deploy Token is the author.

There are some additional considerations:

*   **Image Metadata:** While the authorship isn't directly embedded, incorporating metadata *into* the image using labels can facilitate traceability within the image itself. For instance, one could include the Git commit SHA, build ID, or build date as a label during the build process. However, this isn't a native feature; you would need to manage and implement this.
*   **Container Registry Auditing:** GitLab maintains limited auditing for the container registry. These logs might assist in further investigation, depending on the configuration of logging. However, they won't necessarily resolve the "who" of the push directly.
*   **Security Implications:** Excessive token permissions can blur the lines of authorship, making it harder to identify the author. Regularly audit token permissions and prioritize pipeline-driven pushes where traceability is more readily available.

For resources, GitLab’s official documentation on their Container Registry offers an in-depth explanation of its capabilities. Consult relevant sections concerning CI/CD variables, pipelines, and deploy tokens. Additionally, resources surrounding best practices for container image security and token management are highly recommended. I also advise exploring literature on Git workflow practices to ensure that authorship is carefully tracked before code reaches the pipeline. The official Docker documentation provides information on `docker push` and its authentication mechanisms, which also aligns with GitLab’s usage. Finally, various cybersecurity publications focus on token security and the importance of identifying image authors.
