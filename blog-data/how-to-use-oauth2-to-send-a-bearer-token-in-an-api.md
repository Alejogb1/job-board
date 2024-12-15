---
title: "How to use Oauth2 to send a bearer token in an api?"
date: "2024-12-15"
id: "how-to-use-oauth2-to-send-a-bearer-token-in-an-api"
---

alright, so you're asking about using oauth2 to send a bearer token in an api. it's a pretty common scenario, and i've definitely banged my head against this wall a few times over the years. let me walk you through how i typically approach it, based on my own experiences.

the core idea is that oauth2 is all about authorization, not authentication. we're not trying to verify *who* the user is directly, but *what* they're allowed to do. the bearer token becomes a key that proves they've been authorized.

my first encounter with this was back in my early days working on a microservices project. we had several services that needed to communicate, and rather than passing around credentials we decided to use oauth2 with a central authorization server. it was a bit of a mess at first, as we tried to roll our own solution before discovering the magic of existing libraries, but lesson learned the hard way.

the general process for sending a bearer token goes something like this. first, your client application (whether it's a mobile app, a web frontend, or another service) needs to obtain an access token. this usually involves going through an oauth2 flow like the authorization code grant. that flow is out of this topic, but lets suppose that we already have an access token.

once you have that access token, it needs to be included in the *authorization* header of your http request to your api, using the *bearer* scheme. the header should be like this `authorization: bearer <your_access_token>`.

now, let's look at some code examples. i'm going to show examples in python with the `requests` library, and assuming you already have the access token. python is mostly what i use for api related tasks. if you're using a different language or library, the concepts should be pretty similar, the syntax is the thing that will be different.

**example 1: simple get request with bearer token**

```python
import requests

access_token = "your_actual_access_token_here" # <- replace with your token

headers = {
    "authorization": f"bearer {access_token}"
}

url = "https://api.example.com/resource" # <- replace with your endpoint

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"error: {response.status_code}")
    print(response.text)
```

in this snippet, we define the `access_token` variable that is where your token will go. then we craft the `headers` dictionary, including the *authorization* header with the bearer scheme. the f-string makes it easy to insert the token. finally, we make the get request and handle the response.

it might be worth noting that i once spent hours debugging a similar setup just to realize that i had a single character typo in the `bearer` keyword. a lesson in copy pasting carefully and testing the most basic things first (and not assuming things are right on the first go).

**example 2: post request with json payload and bearer token**

```python
import requests
import json

access_token = "your_actual_access_token_here" # <- replace with your token

headers = {
    "authorization": f"bearer {access_token}",
    "content-type": "application/json"
}

url = "https://api.example.com/resource"  # <- replace with your endpoint

payload = {
    "key1": "value1",
    "key2": "value2"
}


response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 201:
    data = response.json()
    print(data)
else:
    print(f"error: {response.status_code}")
    print(response.text)
```

here, we're doing a post request, and we're sending a json payload. we include the `content-type: application/json` header in this case because we are sending a json payload. otherwise the backend wouldn't know how to decode the payload correctly and would return a 400 or a 415 error code. this is important because the server must know how to parse the incoming data.

**example 3: handling expired tokens**

```python
import requests
import json
import time

def make_api_request(url, access_token, payload=None):
    headers = {
        "authorization": f"bearer {access_token}",
        "content-type": "application/json"
    }
    try:
        if payload:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
        else:
            response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 401: # 401 Unauthorized, which is commonly used to denote expired tokens
            print("token expired")
            # put your refresh token mechanism here (not implemented in the example, but it should go here)
            # usually this is a function to obtain a new access token using a refresh token, like get_new_access_token()
            # new_token = get_new_access_token()
            # if new_token:
              #  return make_api_request(url, new_token, payload) # recursively call with new token
            print("no refresh token mechanism here")
        else:
            raise err
        print(f"http error {err}")
    except Exception as err:
        print(f"general error {err}")



# assume access_token is valid now
access_token = "your_actual_access_token_here" # <- replace with your token
url = "https://api.example.com/resource" # <- replace with your endpoint

payload = {
    "key1": "value1",
    "key2": "value2"
}
response = make_api_request(url, access_token, payload=payload)

if response:
    print(response)
```

this example shows a more robust approach by handling expired tokens. we wrap the request logic in a `make_api_request` function, and check for a `401 unauthorized` status code. if we get it, it *should* mean that our token has expired. in a real application, you'd use the refresh token to get a new access token and retry the request. in this example, the refresh token part is not implemented. please remember to put it when you use this example in a production application, or your users may have a bad experience.

now, about resources: i'd recommend checking out the official oauth2 specification (rfc 6749) if you want a deep understanding of the protocol. it can be a bit dry, but it's a great resource to know how things work at the lower levels. also, a very good book i recommend is "oauth 2 in action", it really helped me clarify many doubts i had initially.

a good practice, in my opinion is to use a library to help with this process, like `python-oauthlib`. usually reinventing the wheel leads to a lot of unnecessary complications, and it's better to rely on battle tested libraries for the low level mechanisms. and remember to always validate the token on the api side and not just on the client side.

that said, if your api is public, be sure to implement rate limiting, and other security mechanisms to protect your service from abuse and denial of service attempts. security is not an afterthought, rather, is something that should be thought from the design of the system, to the deployment. and well, that's basically the general approach to sending bearer tokens in an api using oauth2. let me know if there's anything else, or any more code samples you need. and remember to always test your security configurations, because if it ain't tested, it ain't working. (i once had a bug in my code where i was setting the bearer token correctly, but my security rule was misconfigured so, basically the api was unprotected, it was a hard lesson).
