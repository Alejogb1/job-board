---
title: "How to add an array to Cloud Firestore using the REST API?"
date: "2024-12-23"
id: "how-to-add-an-array-to-cloud-firestore-using-the-rest-api"
---

Alright, let’s tackle this. Adding an array to a Firestore document via the REST API is something I've definitely navigated a few times, particularly during those early days when we were migrating from a different system and needed to seed a ton of data programmatically. I recall a specific project involving a collaborative content platform where user permissions were stored as arrays within each document. Manually creating or editing those through the console would've been a nightmare, so automating it via the REST API became crucial.

The core idea revolves around crafting a properly formatted json payload and then using the `PATCH` or `POST` method, depending on whether you're updating an existing document or creating a new one. The `PATCH` operation, as you'd expect, updates existing fields and can add new ones as needed, while `POST` is used for initial document creation. The key here is understanding how Firestore interprets array data structures within its json representation, and that’s where some specific syntax comes into play.

When working with arrays in Firestore's REST API, they're treated as standard json arrays, so something like `["item1", "item2", "item3"]` is perfectly valid, provided it aligns with the target data type schema for the document field. This means if your field is defined as an array of strings, you have to ensure that your array contents are all strings. If the field is defined as an array of numbers, the contents need to be numeric. You can also have arrays of complex objects, but those need to adhere to json format just as strictly, and that’s a whole other layer we can get into if you have questions.

Let’s walk through some examples. Suppose I'm working with a user document, and it has a `roles` field, defined as an array of strings.

**Scenario 1: Creating a new document with an initial array.**

Here's how the code might look, assuming you have your authentication set up and ready to go:

```python
import requests
import json

# Replace with your actual values
project_id = "your-project-id"
database_id = "(default)"
collection_name = "users"
document_id = "new_user_id"  # use the document id you want
api_key = "your-api-key" # Use api-key auth for now. Ideally Oauth2.
endpoint = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/{database_id}/documents/{collection_name}/{document_id}?key={api_key}"

payload = {
  "fields": {
    "name": { "stringValue": "John Doe" },
    "roles": { "arrayValue": { "values": [
        { "stringValue": "user" },
        { "stringValue": "editor" }
      ] }
    }
  }
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(endpoint, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
  print("Document created successfully.")
  print(response.json())
else:
  print(f"Error creating document: {response.status_code} - {response.text}")

```

Here, we use a `POST` request to create a new document. The important part is the structure within the `fields` section. Notice how the `roles` field is assigned an `"arrayValue"`, and within that, a `values` array that contains our string values, each wrapped in `{"stringValue": ...}`. This explicit structure is crucial; you can't just send a raw array.

**Scenario 2: Updating an existing document by appending elements to an array.**

Now let's say the user already exists and we want to add a new role.

```python
import requests
import json

# Replace with your actual values
project_id = "your-project-id"
database_id = "(default)"
collection_name = "users"
document_id = "existing_user_id" # use the document id you want
api_key = "your-api-key" # Use api-key auth for now. Ideally Oauth2.
endpoint = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/{database_id}/documents/{collection_name}/{document_id}?key={api_key}"

payload = {
  "fields": {
    "roles": {
      "arrayValue": {
        "values": [
          {
            "stringValue": "admin"
          }
        ]
      }
    }
  },
  "updateMask":{
      "fieldPaths":[
          "roles"
      ]
  }
}
#update options
params = {
    "updateMask.fieldPaths": "roles",
    "currentDocument.exists": True
}

headers = {
    "Content-Type": "application/json"
}

response = requests.patch(endpoint, params=params, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
  print("Document updated successfully.")
  print(response.json())
else:
  print(f"Error updating document: {response.status_code} - {response.text}")
```

In this second example, we use a `PATCH` request to modify the document. We are using `updateMask` and `params` to ensure we only update the `roles` field and that it will fail if no such document already exists. We also are adding an `admin` role. If you wanted to merge this, you would have to read the existing document and append this to it.

**Scenario 3: Replacing the entire array.**

Finally, if we want to overwrite the entire `roles` array, it's similar to the above but without a merge.

```python
import requests
import json

# Replace with your actual values
project_id = "your-project-id"
database_id = "(default)"
collection_name = "users"
document_id = "existing_user_id" # use the document id you want
api_key = "your-api-key" # Use api-key auth for now. Ideally Oauth2.
endpoint = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/{database_id}/documents/{collection_name}/{document_id}?key={api_key}"

payload = {
  "fields": {
    "roles": {
      "arrayValue": {
        "values": [
          {
            "stringValue": "superadmin"
          },
            {
            "stringValue": "developer"
          }
        ]
      }
    }
  },
   "updateMask":{
      "fieldPaths":[
          "roles"
      ]
  }
}

params = {
    "updateMask.fieldPaths": "roles"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.patch(endpoint, params=params, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
  print("Document updated successfully.")
  print(response.json())
else:
  print(f"Error updating document: {response.status_code} - {response.text}")
```

Here, we’re simply sending a new array value which replaces the existing one. The same fieldpaths need to be used to define which parts of the document will change.

Some important things to consider in this kind of work:

1.  **Data Type Consistency**: Ensure that the data types in your arrays match the document's schema, or you will face errors. Firestore is strictly typed, and any inconsistencies will result in the operation failing.
2.  **Transaction Considerations**: In production scenarios where atomic updates are needed, particularly when modifying related documents or making a series of changes, you would need to leverage Firestore's transaction mechanisms. This isn't directly via the REST api, but using the clients SDKs.
3.  **Security Rules**: It's vital to define proper security rules in Firestore. This isn't related to this exact request, but the security rules will impact what you can do in requests.
4.  **Rate Limiting**: Be mindful of rate limits, especially when making large batch operations. Implement exponential backoff and retry strategies to handle throttled requests.
5.  **Error Handling**: Always handle non-200 status codes gracefully. Log the errors and consider retry strategies.

If you want to dig deeper into the technical aspects of the Firestore REST API, I recommend consulting Google's official Firestore documentation. Specifically, look into the "REST API Reference" and the “Data model” sections. You can also benefit from "Designing Data for Firestore" document, if you have not gone through this process yet, and it may save you time in the future.

Working with the Firestore REST API can seem a bit involved at first, but once you grasp the structure and the data types, it's a highly effective way to programmatically manage your data at scale. It's one of those skills you'll find invaluable as your projects grow and your data management requirements become more complex.
