---
title: "Why am I Unable to parametrize an ML pipeline endpoint name in Azure Data Factory?"
date: "2024-12-23"
id: "why-am-i-unable-to-parametrize-an-ml-pipeline-endpoint-name-in-azure-data-factory"
---

Let's address this. It's a situation I've encountered more than a few times, specifically when trying to achieve dynamic deployments and parameter-driven orchestration in azure data factory (adf). The frustration is understandable – you want a pipeline endpoint name to be configurable, not hardcoded, particularly when dealing with multiple environments (dev, test, prod) or different machine learning models. The core issue, as I've seen it play out, is that ADF's integration with azure machine learning (aml) often relies on explicit, pre-defined linked services and activities which, by default, do not directly support variable substitution within the ml endpoint name property.

Essentially, when you create an "aml batch execution" or "aml pipeline execution" activity within adf, the endpoint name is usually defined at the linked service level or the activity itself, and those definitions often expect a literal string, not a reference to a parameter. This differs from parameters that you can pass to the AML pipeline itself, which are readily accessible for configuration. Trying to directly insert pipeline parameters within the activity's endpoint field typically leads to an error during validation or pipeline execution.

I recall a particularly challenging scenario with a client where we were deploying multiple identical machine learning models, each serving a different geographical region. Each region had its own aml workspace, its own endpoint, and obviously, we wanted to avoid manual configuration for each region's pipelines. we needed a systematic approach that didn't involve generating separate adf pipelines or hardcoding names. Here's a breakdown of the methods I've used successfully, including working examples:

**The Solution: Leveraging the Power of Web Activity and Dynamic Content**

The primary method to parametrize the endpoint name is to circumvent the direct binding of the ml endpoint in the adf activities and instead employ an intermediate step utilizing the `web activity`. This approach gives us greater flexibility and control over how we interact with the aml endpoints. We essentially use a web activity to fetch the correct endpoint details based on our parameters and then feed that into the subsequent pipeline execution activity.

**Example 1: Simple Endpoint Lookup Using Parameters**

Imagine we have an adf pipeline parameter called `environment`. Depending on whether the value is 'dev' or 'prod', we need a corresponding endpoint. First, define two linked services to AML workspaces: one for `dev` and another for `prod`. Each workspace may have its different endpoint deployed under the same name. Within adf, you can set up a pipeline with a `web activity` before your aml pipeline activity. The `web activity` is configured to make a `get` request to the azure rest api for aml, specifically aimed at getting the details of the deployed endpoint based on the value of the parameter.

```json
// web activity settings:
{
  "name": "GetAMLDeploymentDetails",
  "type": "WebActivity",
    "dependsOn": [],
  "policy": {
      "timeout": "0.00:05:00",
      "retry": 0,
      "retryIntervalInSeconds": 30,
      "secureOutput": false,
      "secureInput": false
  },
  "userProperties": [],
  "typeProperties": {
      "url": {
          "value": "@concat('https://management.azure.com/subscriptions/',pipeline().globalParameters.subscriptionId,'/resourceGroups/', pipeline().globalParameters.resourceGroupName,'/providers/Microsoft.MachineLearningServices/workspaces/',linkedService().workspaceName,'/onlineEndpoints/',if(equals(pipeline().parameters.environment, 'dev'), 'dev-endpoint', 'prod-endpoint'),'?api-version=2022-10-01')",
        "type": "Expression"
      },
      "method": "GET",
      "headers": {
          "Authorization": {
            "value": "@concat('Bearer ', pipeline().globalParameters.accessToken)",
            "type": "Expression"
          },
        "Content-Type":"application/json"
      },
      "body": null,
      "disableCertValidation": false
  }
}
```

In this `web activity`, we are dynamically generating the url using various pipeline parameters and the linked service workspace name. Importantly, the if expression conditionally selects which endpoint to fetch the details for based on the value of `pipeline().parameters.environment`.  The pipeline's `globalParameters` object contains details such as your `subscriptionId`, `resourceGroupName`, and a suitable access token. I would strongly advise researching *azure adf's global parameter setup* and how to obtain a suitable *managed identity access token* which provides proper security.

The output of this `web activity` will contain the endpoint details in the response body as json. Then you can use dynamic content to grab the relevant information and feed it into the subsequent AML execution activity.

**Example 2: Utilizing ADF's `lookup activity` with a configuration table**

Another approach, beneficial when you have many parameters or a more complex mapping, involves using an azure sql database table to store the mappings between environment parameters and the corresponding endpoint names and other relevant settings. The adf pipeline will include a `lookup activity` that queries this mapping table before calling the web activity or the aml execution activity. This decouples configuration from pipeline code and enhances maintainability.

First, you'd create a table such as `EndpointConfiguration` with columns like `environmentName`, `endpointName`, `endpointUri`, `workspaceName` and any other parameters needed for the AML execution.

```sql
-- simplified schema for `EndpointConfiguration`
create table EndpointConfiguration (
    environmentName varchar(50) not null primary key,
    endpointName varchar(255) not null,
    endpointUri varchar(max) not null,
    workspaceName varchar(255) not null
);

-- Example entries:
insert into EndpointConfiguration (environmentName, endpointName, endpointUri, workspaceName) values
('dev','dev-endpoint', 'https://dev-example.azureml.net/score', 'dev-workspace'),
('prod', 'prod-endpoint', 'https://prod-example.azureml.net/score', 'prod-workspace');
```

Next, in adf, after a parameter named 'environment' has been created, add a `lookup activity` to execute the query.

```json
//lookup activity settings
{
    "name": "LookupEndpointConfiguration",
    "type": "Lookup",
    "dependsOn": [],
    "policy": {
        "timeout": "0.00:05:00",
        "retry": 0,
        "retryIntervalInSeconds": 30,
        "secureOutput": false,
        "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
        "source": {
            "type": "AzureSqlSource",
             "sqlReaderQuery": {
                "value": "select endpointName, endpointUri, workspaceName from EndpointConfiguration where environmentName = '@{pipeline().parameters.environment}'",
                "type": "Expression"
            },
            "partitionOption": "None"
            },
        "dataset": {
            "referenceName": "linked_azure_sql_server",
            "type": "DatasetReference"
         },
        "firstRowOnly": true
    }
}
```

Now, based on the results, you'll have the required values to use the `web activity` as outlined in the first example, or use the output directly in your aml execution activity.

**Example 3: Combining with an adf `set variable` Activity**

After retrieving the endpoint details using one of the above methods, I often leverage an adf `set variable` activity to create a well-defined variable that is used as input in subsequent activities. This makes the pipeline flow easier to read and reduces the complexity in each activity itself. This also can provide the advantage of type safety in case the activity is sensitive to the data type.

For instance, you could have the output of the `lookup activity` (as in example 2) and want to set variables: `endpoint_name`, `endpoint_uri` and `aml_workspace_name` to hold the values to use in the downstream activities. For each, you would configure a `set variable` activity like below:

```json
// set variable activity settings: endpoint_name
{
  "name": "SetEndpointNameVariable",
  "type": "SetVariable",
  "dependsOn": [
    {
      "activity": "LookupEndpointConfiguration",
      "dependencyConditions": [
        "Succeeded"
      ]
    }
  ],
  "userProperties": [],
  "typeProperties": {
    "variableName": "endpoint_name",
    "value": {
       "value": "@activity('LookupEndpointConfiguration').output.firstRow.endpointName",
        "type": "Expression"
    }
  }
}
```

and subsequently

```json
// set variable activity settings: endpoint_uri
{
  "name": "SetEndpointUriVariable",
  "type": "SetVariable",
  "dependsOn": [
    {
      "activity": "SetEndpointNameVariable",
      "dependencyConditions": [
        "Succeeded"
      ]
    }
  ],
  "userProperties": [],
  "typeProperties": {
    "variableName": "endpoint_uri",
    "value": {
       "value": "@activity('LookupEndpointConfiguration').output.firstRow.endpointUri",
        "type": "Expression"
    }
  }
}
```

These variables can then be used directly in aml activities, such as the `aml batch execution` activity by referencing `variables('endpoint_name')`, `variables('endpoint_uri')`, and so on. This clearly shows how the retrieved values can be used to drive the actual execution of your machine learning pipeline, parametrized for different environments or deployment contexts.

**Key Resources for Further Learning**

For more in-depth knowledge and a better grasp of the concepts mentioned, I highly recommend these resources:

*   **Microsoft Azure Documentation:** Specifically, the section on Azure Data Factory's web activity, lookup activity, and the integration with azure machine learning services. The official documentation is invaluable and the first port of call.
*   **"Programming Azure Data Factory: A Comprehensive Guide"** (If a book exists) this would provide a comprehensive overview. While specific titles may vary, similar books dedicated to azure data services are beneficial.
*   **Azure Rest API Documentation:** Familiarizing yourself with the aml rest api will greatly enhance your ability to build dynamic solutions such as the web activity example shown.

In closing, parameterizing an adf pipeline endpoint name requires an intermediate step, leveraging tools like the `web activity` or `lookup activity`, along with adf's expression language. This method allows for greater flexibility and better management of complex deployment scenarios. The examples provided offer a starting point, but adapting them to your specific needs is key. Remember, the best solutions are those that are both robust and easy to maintain.
