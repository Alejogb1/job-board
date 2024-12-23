---
title: "How to send data to a Mailchimp audience?"
date: "2024-12-23"
id: "how-to-send-data-to-a-mailchimp-audience"
---

 Sending data to a Mailchimp audience might seem straightforward at first, but in my experience, the devil’s often in the details. I remember back when I was managing a migration project for an e-commerce platform; ensuring customer data synced correctly with Mailchimp was a recurring headache. It's not simply a matter of 'pushing' data; it’s about doing it reliably, securely, and adhering to Mailchimp's API limitations. So, let's break down the approach, considering both simple and more complex scenarios, and I'll throw in some code examples to clarify things.

First, the fundamental concept: Mailchimp primarily interacts with external systems through its REST API. This means we’re going to be dealing with HTTP requests, specifically POST requests for adding new subscribers or PATCH requests for updating existing ones. The critical part here is understanding how Mailchimp identifies its members – they use a unique ‘email’ as the primary key. So, every operation involves the email address.

Now, the simplest method involves adding a new subscriber to a specific audience list. I’ve seen this used frequently for signup forms on websites. Here’s a python snippet demonstrating a basic approach using the `requests` library:

```python
import requests
import json

api_key = 'YOUR_MAILCHIMP_API_KEY'
server_prefix = 'YOUR_SERVER_PREFIX' # found in your mailchimp api key
list_id = 'YOUR_AUDIENCE_ID'

def add_subscriber(email, first_name, last_name):
    url = f'https://{server_prefix}.api.mailchimp.com/3.0/lists/{list_id}/members'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'email_address': email,
        'status': 'subscribed',
        'merge_fields': {
            'FNAME': first_name,
            'LNAME': last_name
        }
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
      print(f"Successfully added {email} to Mailchimp audience.")
    else:
        print(f"Error adding subscriber: {response.status_code} - {response.text}")

if __name__ == '__main__':
  add_subscriber('test@example.com', 'John', 'Doe')
```

In the example above, replace `YOUR_MAILCHIMP_API_KEY`, `YOUR_SERVER_PREFIX`, and `YOUR_AUDIENCE_ID` with your actual Mailchimp API key, server prefix (e.g., us20), and audience ID. This code sends a POST request to the `/lists/{list_id}/members` endpoint, including the subscriber's email address, status (set to ‘subscribed’), and merge fields for the first and last names. The API key is used for authorization using the bearer token header, which is common to this kind of rest api request. We have to format our data as JSON in the body of the request. This should handle the simple scenario when you just want to add a single new record to your mailchimp audience.

Let’s move on to something more challenging. Often, we need to update an existing subscriber, say, if they’ve changed their name or if we’ve acquired additional information. Instead of creating a new subscriber, we need to patch the record that corresponds to their email address. Mailchimp's API uses a unique identifier for each record, generated using a hash of the email address, but I've found that creating it yourself isn't typically necessary when you're using the API. Here's how that looks with a slight change in the code:

```python
import requests
import json
import hashlib

api_key = 'YOUR_MAILCHIMP_API_KEY'
server_prefix = 'YOUR_SERVER_PREFIX'
list_id = 'YOUR_AUDIENCE_ID'


def update_subscriber(email, new_first_name=None, new_last_name=None, new_status=None):
    email_hash = hashlib.md5(email.lower().encode()).hexdigest() #mailchimp requires md5 hash of the lowercase email
    url = f'https://{server_prefix}.api.mailchimp.com/3.0/lists/{list_id}/members/{email_hash}'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
       'merge_fields': {},
    }

    if new_first_name:
      data['merge_fields']['FNAME'] = new_first_name
    if new_last_name:
        data['merge_fields']['LNAME'] = new_last_name
    if new_status:
      data['status'] = new_status

    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
      print(f"Successfully updated {email} in Mailchimp audience.")
    else:
      print(f"Error updating subscriber: {response.status_code} - {response.text}")

if __name__ == '__main__':
    update_subscriber('test@example.com', new_first_name='Jane', new_last_name='Doe', new_status='pending')
```

This code snippet updates an existing subscriber, using a PATCH request. Notice that the endpoint now includes the md5 hash of the email address in the URL. We only need to provide the fields that we're actually changing; the other fields on the member will remain unaffected, so we can just include new_first_name, new_last_name, and new_status as optional parameters in the function.

Finally, let’s consider scenarios where you might want to synchronize a large amount of data. While the above methods work fine for small-scale updates, for large datasets it’s often more efficient to leverage Mailchimp’s batch operations. This allows you to send multiple requests in a single API call, greatly reducing the number of network requests and speeding up the process.

```python
import requests
import json
import hashlib

api_key = 'YOUR_MAILCHIMP_API_KEY'
server_prefix = 'YOUR_SERVER_PREFIX'
list_id = 'YOUR_AUDIENCE_ID'

def batch_update_subscribers(subscribers):
    operations = []
    for subscriber in subscribers:
        email_hash = hashlib.md5(subscriber['email'].lower().encode()).hexdigest()
        operation = {
            'method': 'PATCH',
             'path': f'/lists/{list_id}/members/{email_hash}',
             'body': json.dumps({
                 'merge_fields': {
                    'FNAME': subscriber.get('first_name'),
                    'LNAME': subscriber.get('last_name')
                 },
                 'status': subscriber.get('status','subscribed')
             })
        }
        operations.append(operation)
    url = f'https://{server_prefix}.api.mailchimp.com/3.0/batches'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'operations':operations
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        batch_id = response.json()['id']
        print(f"Batch job initiated with id: {batch_id}. Check batch status for results.")
    else:
        print(f"Error submitting batch update: {response.status_code} - {response.text}")
if __name__ == '__main__':
    subscribers = [
        {'email':'test1@example.com', 'first_name':'New First 1', 'last_name':'New Last 1', 'status':'subscribed'},
        {'email':'test2@example.com', 'first_name':'New First 2', 'last_name':'New Last 2', 'status':'unsubscribed'},
    ]
    batch_update_subscribers(subscribers)
```

This code creates a list of operations (PATCH requests), then submits them to the `/batches` endpoint. Batch operations are asynchronous, so you’ll receive a batch id in the response. You'd need to use that id to poll the batches endpoint later to get results.

For a deeper dive, I would suggest looking at the Mailchimp API documentation, especially the sections on audience members and batches. Further, the book “RESTful Web Services” by Leonard Richardson and Sam Ruby provides a good overview of the principles behind REST APIs, which can be quite valuable when working with Mailchimp’s API or any other rest based service. Additionally, I recommend the book "Designing Data-Intensive Applications" by Martin Kleppmann to gain a broader perspective on systems that manage large amounts of data, which might help inform decisions when planning larger synchronizations. Finally, become extremely comfortable with the http status codes, as it is the best indicator of if a request succeeded or not.

In conclusion, sending data to a Mailchimp audience is fundamentally about understanding Mailchimp’s API and how to craft your HTTP requests to match its expectations, including authorization with an api key. Whether it’s simple subscriber adds, updates, or more complex batch processes, a good grasp of these principles is vital for building robust integrations and making the process much smoother in the long run.
