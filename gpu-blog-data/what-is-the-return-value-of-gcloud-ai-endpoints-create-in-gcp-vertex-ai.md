---
title: "What is the return value of `gcloud ai endpoints create` in GCP Vertex AI?"
date: "2025-01-26"
id: "what-is-the-return-value-of-gcloud-ai-endpoints-create-in-gcp-vertex-ai"
---

The `gcloud ai endpoints create` command in Google Cloud Platform's (GCP) Vertex AI, when successfully executed, returns a JSON representation of the newly created endpoint resource. This response is not simply a success/failure flag, but a detailed record of the endpoint's configuration and current state within Vertex AI. I've worked extensively with Vertex AI model deployment and management, and understanding the precise content of this return value is critical for automating subsequent steps in a machine learning pipeline.

The primary reason for the structured JSON output is to enable programmatic interaction with Vertex AI via scripting and other automation tools. After creating an endpoint, you typically need its identifier or other properties to attach deployed models. Without a defined return value format, you’d face substantial parsing challenges. The JSON structure includes essential fields such as the endpoint's name, display name, creation timestamp, update timestamp, the network it is connected to, and resource labels. These details allow for accurate tracking and management of your deployed machine learning infrastructure.

To understand what you’d find, consider a typical invocation and the subsequent return:

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="my-test-endpoint"
```

The successful execution of this command doesn’t return a simple acknowledgment. Instead, it yields a large, structured JSON object similar to the one I will describe below.

```json
{
  "name": "projects/1234567890/locations/us-central1/endpoints/1234567890123456789",
  "displayName": "my-test-endpoint",
  "createTime": "2024-01-26T14:30:00.123456Z",
  "updateTime": "2024-01-26T14:30:00.123456Z",
  "deployedModels": [],
  "network": "projects/1234567890/global/networks/default",
  "etag": "ABmJ-...",
  "labels": {},
  "encryptionSpec": {
    "kmsKeyName": ""
  },
  "trafficSplit": {},
  "disableMonitoring": false,
  "enablePrivateServiceConnect": false,
  "modelDeploymentMonitoringJob": ""
}
```

**Explanation of Key Fields:**

*   `name`: This is the fully qualified resource name of the created endpoint, essential for uniquely identifying it within GCP. It includes the project ID, location, and a specific endpoint identifier. This is the most crucial value to capture if you need to reference this endpoint in other gcloud commands or API calls.

*   `displayName`: This field reflects the user-defined name specified with the `--display-name` flag. It's purely for readability and management purposes.

*   `createTime` and `updateTime`: These are timestamps representing when the endpoint was created and last updated, respectively. They follow the RFC3339 UTC format.

*   `deployedModels`: An initially empty array. As models are deployed to this endpoint, they will appear in this array with details including the deployed model ID.

*   `network`: Specifies the VPC network associated with the endpoint, relevant if your endpoint requires private connectivity.

*   `etag`: Used for optimistic concurrency control. It’s crucial for ensuring atomicity of update operations on the endpoint resource.

*  `labels`: This dictionary allows you to add user defined labels to the endpoint for better resource organization.

*   `encryptionSpec`: Details the encryption configuration. An empty string in `kmsKeyName` means Google-managed encryption is used.

*   `trafficSplit`: Controls the percentage of traffic routed to specific deployed models associated with this endpoint. Initially empty as no models are deployed.

*  `disableMonitoring`: A boolean indicating if the monitoring for model deployment on this endpoint is disabled.

*  `enablePrivateServiceConnect`: A boolean indicating if the private service connect option is enabled for this endpoint.

*   `modelDeploymentMonitoringJob`: Refers to the ID of the monitoring job related to this endpoint. It's empty initially and will be populated when a model deployment is monitored.

The specific fields and their values can vary depending on the options you include in the `gcloud ai endpoints create` command, such as specifying a network or applying labels.

To illustrate, let's consider some practical examples and their corresponding, simplified, return values:

**Example 1: Adding a Network**

Assume you want the endpoint created in the 'us-central1' region to use the private network specified.

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="private-endpoint" \
    --network="projects/1234567890/global/networks/my-custom-network"
```

The resulting JSON would be similar, but with the `network` field populated differently.

```json
{
  "name": "projects/1234567890/locations/us-central1/endpoints/9876543210987654321",
  "displayName": "private-endpoint",
  "createTime": "2024-01-26T15:00:00.123456Z",
  "updateTime": "2024-01-26T15:00:00.123456Z",
  "deployedModels": [],
  "network": "projects/1234567890/global/networks/my-custom-network",
  "etag": "BcdE-...",
  "labels": {},
  "encryptionSpec": {
    "kmsKeyName": ""
  },
  "trafficSplit": {},
  "disableMonitoring": false,
  "enablePrivateServiceConnect": false,
  "modelDeploymentMonitoringJob": ""
}
```

Note the `network` field now points to the custom network defined in the command. This is key for those deploying in controlled network environments.

**Example 2: Using Labels**

This example demonstrates adding labels for better organization.

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="labeled-endpoint" \
    --labels="env=dev,team=ml"
```

The JSON response will have the `labels` section filled accordingly:

```json
{
  "name": "projects/1234567890/locations/us-central1/endpoints/0123456789012345678",
  "displayName": "labeled-endpoint",
  "createTime": "2024-01-26T15:30:00.123456Z",
  "updateTime": "2024-01-26T15:30:00.123456Z",
  "deployedModels": [],
  "network": "projects/1234567890/global/networks/default",
  "etag": "Cdef-...",
  "labels": {
     "env": "dev",
     "team": "ml"
  },
  "encryptionSpec": {
    "kmsKeyName": ""
  },
  "trafficSplit": {},
  "disableMonitoring": false,
  "enablePrivateServiceConnect": false,
  "modelDeploymentMonitoringJob": ""
}
```

The `labels` field contains the specified key-value pairs. This is helpful for filtering and grouping your deployed endpoints within the GCP console or using other gcloud commands.

**Example 3: Specifying Encryption**

Here I'll create an endpoint that uses a custom KMS key for encryption

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="encrypted-endpoint" \
     --encryption-kms-key "projects/1234567890/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key"
```

The JSON response will have the `encryptionSpec` section filled accordingly:

```json
{
  "name": "projects/1234567890/locations/us-central1/endpoints/7654321098765432109",
  "displayName": "encrypted-endpoint",
  "createTime": "2024-01-26T16:00:00.123456Z",
  "updateTime": "2024-01-26T16:00:00.123456Z",
  "deployedModels": [],
  "network": "projects/1234567890/global/networks/default",
  "etag": "EfGh-...",
  "labels": {},
    "encryptionSpec": {
      "kmsKeyName": "projects/1234567890/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key"
  },
  "trafficSplit": {},
  "disableMonitoring": false,
  "enablePrivateServiceConnect": false,
  "modelDeploymentMonitoringJob": ""
}
```

As you can see, the `encryptionSpec.kmsKeyName` field is populated with the specified custom key. This ensures that the endpoint is encrypted using a customer managed encryption key.

**Resource Recommendations:**

To further delve into this topic and associated details, I suggest examining the official Google Cloud documentation for Vertex AI, particularly the pages on endpoint creation and management. Also, reviewing the Vertex AI API reference is crucial for understanding the JSON structure and possible values. Additionally, working through practical labs provided by Google and third-party platforms can solidify these concepts with hands-on experience. Finally, engaging with the wider Vertex AI community through forums can offer alternative viewpoints and resolutions to specific issues you might encounter when handling the structured data returned from this command. Understanding the intricacies of this returned JSON is key to efficiently managing Vertex AI endpoints.
