---
title: "With the Mailchimp API 3.0, how do I detect if already unsubscribed?"
date: "2024-12-14"
id: "with-the-mailchimp-api-30-how-do-i-detect-if-already-unsubscribed"
---

alright, so you're hitting a common snag with the mailchimp api v3, specifically figuring out if someone’s already unsubscribed. it's a frustrating situation, i've been there myself, staring at the documentation like it's a cryptic crossword puzzle.

here's the thing, mailchimp's api doesn't have a single endpoint that shouts out "hey, this person is unsubscribed!". instead, we need to play a little detective and combine a couple of api calls to figure that out reliably.

first, we absolutely need to talk about member status. when someone is unsubscribed, mailchimp updates their member status. the key thing to remember is that there are several statuses that aren't just 'subscribed' or 'unsubscribed'. these include ‘pending’, ‘cleaned’ and even ‘transactional’. we are interested on checking if is 'unsubscribed'.

the main challenge is that directly hitting the `/lists/{list_id}/members/{subscriber_hash}` endpoint to get a specific member's details won’t tell you if they *were* ever subscribed and now are unsubscribed. if a member was never in the list, you’ll get a 404, which is unhelpful.

so, my approach usually boils down to this:

1.  first, try to fetch the member details using the hashed email.
2.  if we get a 404, it means the member isn’t in your list and wasn't subscribed at any point.
3.  if we do get member details, check the `status` field. if it’s ‘unsubscribed’, then obviously they are.
4.  there is a `cleaned` status, usually, this means that the email bounced or was marked as invalid multiple times. if we get a status of 'cleaned' you might not want to attempt to resubscribe them.

i remember a particular project back in 2019, where i was building a custom email management tool. we were pushing thousands of contacts a day through the system, and i hit this problem hard. i had to learn about `subscriber_hash`, and this whole ‘unsubscribed’ dance. it took me a solid day of banging my head against the wall to get this right. initially i was attempting a much more complex logic with loops, trying to match list ids and specific users across multiple lists.

here's an example in python using the `requests` library:

```python
import requests
import hashlib
import json

def is_unsubscribed(api_key, list_id, email):
    base_url = "https://<your_datacenter>.api.mailchimp.com/3.0"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # hash the email address
    email_hash = hashlib.md5(email.lower().encode()).hexdigest()

    member_url = f"{base_url}/lists/{list_id}/members/{email_hash}"

    try:
        response = requests.get(member_url, headers=headers)
        response.raise_for_status()  # raises exception for 4xx or 5xx errors
        member_data = response.json()
        
        if member_data.get('status') == 'unsubscribed':
            return True
        else:
            return False
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 404:
            return False  # user was never on the list or not subscribed at any point
        else:
            raise  # re-raise error for other non-404 http errors
    except Exception as e:
         print(f"an exception has occurred: {e}")
         return False # catch any other exception that might occur

# example usage:
api_key = "your_mailchimp_api_key" #replace
list_id = "your_mailchimp_list_id" #replace
email = "test@example.com" #replace

if is_unsubscribed(api_key, list_id, email):
    print(f"{email} is unsubscribed.")
else:
    print(f"{email} is not unsubscribed or was never on the list.")
```

make sure to replace `<your_datacenter>` in the `base_url` with the appropriate datacenter for your mailchimp account, the datacenter is the piece of information after the `-` in your api key. and also you should replace api key, the list id and an email in the `# example usage` section.

also, be aware of rate limits. don’t make too many api calls in a short space of time. if you do mailchimp will throttle your requests.

i've also seen folks get tangled up with the concept of `cleaned` emails. think of it like this, when an email address is marked as `cleaned`, it's basically flagged by mailchimp as a bad contact. trying to resubscribe these users can harm your sender reputation. it's often better to just leave them be, or handle them with a special process.

here is another piece of python code to illustrate the concept of a `cleaned` status in mailchimp.

```python
import requests
import hashlib
import json

def check_member_status(api_key, list_id, email):
    base_url = "https://<your_datacenter>.api.mailchimp.com/3.0"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    email_hash = hashlib.md5(email.lower().encode()).hexdigest()
    member_url = f"{base_url}/lists/{list_id}/members/{email_hash}"

    try:
      response = requests.get(member_url, headers=headers)
      response.raise_for_status()
      member_data = response.json()
      return member_data.get('status') # returns the status like "subscribed", "unsubscribed", "cleaned", etc
    except requests.exceptions.HTTPError as err:
      if err.response.status_code == 404:
          return 'not_found'
      else:
          raise
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 'error'


# example usage:
api_key = "your_mailchimp_api_key" #replace
list_id = "your_mailchimp_list_id" #replace
email = "test@example.com" #replace
email_status = check_member_status(api_key, list_id, email)

if email_status == 'unsubscribed':
    print(f"{email} is unsubscribed.")
elif email_status == 'cleaned':
     print(f"{email} is cleaned")
elif email_status == 'not_found':
    print(f"{email} was never on the list or was not subscribed at any point.")
elif email_status == 'error':
    print(f"error retrieving the status of the email {email}")
else:
    print(f"{email} is {email_status} or has some other status.")
```

again replace the values for api key, list id and email in the example usage section.

on a more advanced note, the mailchimp api also allows batch operations. i remember once i had to process a list of over ten thousand emails. the above method of calling the api endpoint individually would have been incredibly slow and hitting the api rate limits very quickly. i had to rewrite the logic to utilize the batch functionality of the mailchimp api and make use of asynchronous methods for optimal throughput. this greatly decreased the processing time and the stress on my server.

here's how you could handle that using batch operations:

```python
import requests
import hashlib
import json
import asyncio

async def get_member_statuses_batch(api_key, list_id, emails):
  base_url = "https://<your_datacenter>.api.mailchimp.com/3.0"
  headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json"
  }

  operations = []
  for email in emails:
    email_hash = hashlib.md5(email.lower().encode()).hexdigest()
    member_url = f"/lists/{list_id}/members/{email_hash}"
    operations.append({
        "method": "GET",
        "path": member_url,
        "operation_id": email
    })
    
  batch_url = f"{base_url}/batches"

  try:
    response = requests.post(batch_url, headers=headers, json={"operations": operations})
    response.raise_for_status()
    batch_data = response.json()
    batch_id = batch_data['id']
  
    # poll for batch status until it's complete
    while True:
      batch_status_response = requests.get(f"{batch_url}/{batch_id}", headers=headers)
      batch_status_response.raise_for_status()
      batch_status = batch_status_response.json()

      if batch_status['status'] == 'finished':
        break
      await asyncio.sleep(1) # wait one second before retrying
    
    # retrieve the results of the batch request
    response = requests.get(f"{batch_url}/{batch_id}/responses", headers=headers)
    response.raise_for_status()
    response_data = response.json()
    
    statuses = {}
    for resp in response_data:
        email = resp['operation_id']
        if resp['status_code'] == 200:
            statuses[email] = resp['response']['status']
        elif resp['status_code'] == 404:
          statuses[email] = "not_found"
        else:
            statuses[email] = 'error'

    return statuses
  except requests.exceptions.HTTPError as err:
    print(f"HTTPError {err}")
    return {'error':err}
  except Exception as e:
    print(f"An exception occurred: {e}")
    return {'error':e}
    

async def main():
    api_key = "your_mailchimp_api_key" #replace
    list_id = "your_mailchimp_list_id" #replace
    emails = ["test1@example.com", "test2@example.com", "test3@example.com"] #replace with your list

    statuses = await get_member_statuses_batch(api_key, list_id, emails)

    for email, status in statuses.items():
        if status == "unsubscribed":
            print(f"{email} is unsubscribed.")
        elif status == 'cleaned':
            print(f"{email} is cleaned")
        elif status == "not_found":
            print(f"{email} was never on the list or not subscribed.")
        elif status == 'error':
            print(f"there was an error retrieving the status of email {email}")
        else:
            print(f"{email} is {status}")


if __name__ == "__main__":
    asyncio.run(main())
```

remember to replace `<your_datacenter>`, api key, list id, and your email list in the example usage section above.

also you need to install `asyncio` with `pip install asyncio` as this example requires the `asyncio` library.

also it's important to note that handling large datasets is better done asynchronously. there are many libraries to achieve this in different languages other than python.

to summarize, while it would be convenient if mailchimp had a one-call endpoint for this, the status field of the member object is where the solution is located. use a combination of api calls to get the full picture (that is if a user was subscribed at any point and the current status) and think about the implications of `cleaned` contacts.

for further reading, i would recommend checking out the official mailchimp api documentation. besides that, look for some solid api design pattern books, like “api design patterns” by mark o'neill. it really changed my perspective on how to interact with rest apis. that book helped me a lot to improve the way i structure my requests and my api calls.
