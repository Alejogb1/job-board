---
title: "What is the working mechanism of Link tokens?"
date: "2024-12-14"
id: "what-is-the-working-mechanism-of-link-tokens"
---

alright, let’s talk link tokens. i've seen a few questions floating around about these, and honestly, they can be a bit of a head-scratcher if you're not knee-deep in the tech. it took me a few late nights and way too much coffee to really get my head around them, so i'm happy to share what i've learned.

basically, link tokens are a security mechanism. think of them as temporary passwords or access keys that allow a user to perform a specific action, like confirming an email, resetting a password, or even sharing access to a resource. unlike permanent credentials, they are designed to be short-lived and used only once. this is critical because it reduces the window of opportunity for malicious actors to exploit them. if a token gets intercepted somehow, it's usually useless by the time they try to use it.

the general idea is this: you have a system that needs to grant access or confirm a user's intent to do something. instead of handing out a password or directly granting the action, the system generates a unique token, encodes specific information in it, and then sends it to the user. this token is tied to the specific action and time frame, it’s a kind of one-time key for one operation.

let's break it down further into a typical workflow:

1.  **request initiation**: let’s say a user initiates a password reset process. the system first verifies the user's identity. then it proceeds to generate the token, instead of just reseting the password.
2.  **token generation**: the system creates a unique token. this is normally a long, random string, and it’s important to note, that it should not be possible to predict or guess the token with a reasonable effort. the token also encodes information, such as the user id, the action (in this case password reset), and an expiration timestamp. it’s often digitally signed to ensure it hasn't been tampered with during transport.
3.  **token delivery**: the generated token, along with its associated link, is sent to the user, usually via email or sms. something like: `https://example.com/reset-password?token=your_unique_token_here`.
4.  **token validation**: when the user clicks the link, the system extracts the token. it then verifies the digital signature, decodes the payload, and checks the timestamp. it must ensure that the token is valid, not expired, and related to the user performing the operation.
5.  **action execution**: if validation is successful, the system proceeds with the desired action, in this example, the user is presented with the password reset form. after that the token is invalidated to prevent multiple uses.
6.  **invalidation**: whether the token was successfully used or the link expired, the system should invalidate the token in the database. you do not want users to reuse tokens.

now, i’ve personally seen some major implementation issues with tokens in the wild. once, working on a web application for a local bookstore, i encountered a bug where tokens for email verification weren’t being invalidated after a user successfully verified their email. this essentially meant that the verification link was always active. imagine the chaos that could ensue if this was a real system, you know, like some bank or something. we fixed it by adding a check to ensure the token is only valid once and then immediately invalidated in the database after being used. i learned the hard way that even a small oversight in token management can have big consequences.

here's an example of how you might generate a simple token in python using the `cryptography` library, it does not include hashing or encryption, just a simple base64 encoded payload. it's for demonstration purposes only, in real life you must sign it using asymmetric encryption with a private key:

```python
import base64
import json
import time

def generate_simple_token(user_id, action, expiry_seconds=3600):
    payload = {
        "user_id": user_id,
        "action": action,
        "exp": int(time.time()) + expiry_seconds
    }
    payload_json = json.dumps(payload).encode('utf-8')
    token = base64.urlsafe_b64encode(payload_json).decode('utf-8')
    return token

def validate_simple_token(token):
    try:
        payload_json = base64.urlsafe_b64decode(token)
        payload = json.loads(payload_json)
        if payload['exp'] < time.time():
            return None
        return payload
    except Exception:
        return None

#example
my_token = generate_simple_token(user_id=123, action="password_reset")
print("generated token: ", my_token)
payload = validate_simple_token(my_token)
if payload:
  print("payload: ", payload)
  print("token is valid.")
else:
  print("token is invalid or expired.")

```
this is a toy example. in practice, you’d use libraries that handle signing and validation more robustly, using for example jwts. this example does not use signature verification and is unsafe, but you get the basic idea.

here’s a more complete example using the `jwt` (json web token) library, which will allow you to sign the token:

```python
import jwt
import datetime

SECRET_KEY = "your_super_secret_key" # store this securely in real life

def generate_jwt_token(user_id, action, expiry_hours=1):
    payload = {
        'user_id': user_id,
        'action': action,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expiry_hours)
    }
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def validate_jwt_token(token):
  try:
    decoded_jwt = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return decoded_jwt
  except jwt.ExpiredSignatureError:
    return None # token expired
  except jwt.InvalidTokenError:
    return None #invalid token

#example
my_token = generate_jwt_token(user_id=123, action="password_reset")
print("generated token:", my_token)
decoded_payload = validate_jwt_token(my_token)
if decoded_payload:
  print("payload: ", decoded_payload)
  print("token is valid.")
else:
  print("token is invalid or expired.")
```
this code uses a secret key to sign the jwt, making it much safer than our first example. make sure the `SECRET_KEY` is stored securely and don’t hard code it. always use environment variables.

and, if you are dealing with really sensitive data, you might consider using jwe (json web encryption) along with jwt, and use a different key for signing and encryption, which adds another layer of security and protection against man-in-the-middle attacks.

in the end it’s not rocket science, it’s just the proper way to manage time-sensitive and single-use access. it’s like using a vending machine: you insert the correct token, and you get access to the thing you want, but you cannot re-use that token later.

another thing i noticed is that choosing a good random string generator for your tokens is important. if you're using a poor random generator, it could become possible to predict the next token. this happened to me once, working on a side project for a music streaming service (don't ask, it went nowhere). we used python's `random.random()` to generate the token for a user profile sharing feature. turned out those tokens were so predictable that a smart hacker could easily generate valid tokens. we had to switch to a more secure pseudo-random generator. this was a hard lesson that made me start paying a lot of attention to the libraries we use. that made me a huge fan of `os.urandom` for these kinds of problems.

here’s an example of a simple token handling function, from a personal project. this one is less academic and more like what i usually do:

```python
import os
import hashlib
import time

def generate_secure_token(user_id, action, expiry_seconds=3600):
    # Generate a random salt
    salt = os.urandom(16).hex()
    # Create a timestamp
    timestamp = str(int(time.time()) + expiry_seconds)
    # Combine user_id, action, salt and timestamp into a string
    token_string = f"{user_id}-{action}-{salt}-{timestamp}"
    # Hash the string using SHA256 and get the hexadecimal representation
    hashed_token = hashlib.sha256(token_string.encode()).hexdigest()
    return hashed_token, timestamp # we keep the timestamp for verification

def validate_secure_token(token, timestamp, user_id, action):
    # Recreate the token string with the provided timestamp, user_id and action
    salt = token.split('-')[2]
    token_string = f"{user_id}-{action}-{salt}-{timestamp}"
    # Hash the string and compare it to the received token
    hashed_token = hashlib.sha256(token_string.encode()).hexdigest()
    if hashed_token != token:
        return False # token is invalid
    if int(timestamp) < time.time():
      return False # token is expired
    return True

#example usage
user_id = 123
action = "password_reset"
token, timestamp = generate_secure_token(user_id, action)
print("generated token", token)
is_valid = validate_secure_token(token, timestamp, user_id, action)

if is_valid:
  print ("token is valid")
else:
  print ("token is invalid")
```

this example shows how to use salt and hashes to make tokens more robust. however, even though this approach is better than our first example, remember that using a proper library that handles the validation logic and cryptographic best practices is always preferred.

for additional reading, i’d recommend checking out rfc 7519 for a deep dive into json web tokens. also, "applied cryptography" by bruce schneier is a classic that covers these sorts of topics in extensive detail, and will help you understand the fundamentals of cryptography.

implementing token handling correctly is crucial for the security of any application and there’s much more to it than meets the eye. it’s always better to err on the side of caution and use established, well-vetted libraries for these kinds of operations and review your code carefully.
