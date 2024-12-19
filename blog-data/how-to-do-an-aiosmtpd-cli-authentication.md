---
title: "How to do an aiosmtpd CLI Authentication?"
date: "2024-12-15"
id: "how-to-do-an-aiosmtpd-cli-authentication"
---

alright, so you're looking at setting up authentication for your aiosmtpd server, and want to do it from the command line, right? i've been there, trust me. spent a good chunk of my early career neck-deep in email servers, and aiosmtpd wasn't always as straightforward as we'd like. back then, before a lot of the refined libraries, we were practically building the wheel for every single feature, authentication included. think raw socket programming, handling every single byte, good old days. not!

it's a fair question. while aiosmtpd has made things way easier, sometimes the initial setup can seem like a hurdle, especially when you’re accustomed to more traditional smtp setups. let's cut to the chase, no need for fancy footwork, just the facts.

first, let's get this straight: aiosmtpd itself doesn't directly provide a command-line interface for user management like, say, your average linux useradd or passwd tool. what it gives you is a framework to build your own authentication mechanism. you are basically defining what will tell the server “yes, this user is good to send” or "no, go away.". so when we're talking about cli authentication, what we are actually talking about is how to configure the server to pick up authentication data that you may manipulate from your shell.

the easiest and fastest way, for me, at least, especially for testing or for a simple setup, is to use a static authentication list. it is super simple. this can be a simple dictionary or json file where you store user credentials. you read it and you do the check against the credentials in that file, on the aiosmtpd side.

here is an example on how to set it up with a simple dictionary that you can manipulate from the outside via a quick script.

```python
import asyncio
import json
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Debugging
from aiosmtpd.smtp import SMTP, AuthResult


class CustomSMTP(SMTP):
    def __init__(self, auth_users, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_users = auth_users


    async def auth_LOGIN(self, server, args):
        if not args:
            await server.push(b"334 " + b"VXNlcm5hbWU6")
            return None
        
        username = args[0].decode()
        
        await server.push(b"334 " + b"UGFzc3dvcmQ6")
        return ("auth_plain", username)


    async def auth_PLAIN(self, server, args):
            
        if not args:
            await server.push(b"501 Syntax error")
            return False
        
        auth_string = args[0].decode()
        try:
            auth_data = base64.b64decode(auth_string).decode().split('\x00')
            username = auth_data[1]
            password = auth_data[2]
        except (IndexError, ValueError):
            await server.push(b"501 Syntax error")
            return False
        
        if username in self.auth_users and self.auth_users[username] == password:
            return AuthResult(success=True, auth_ident=username)
        
        await server.push(b"535 Authentication failed")
        return AuthResult(success=False)


    async def auth_CRAM_MD5(self, server, args):
        # Implement cram-md5 challenge logic here if needed
        await server.push(b"502 Authentication method not implemented")
        return AuthResult(success=False)

async def amain(auth_users):
    controller = Controller(CustomSMTP(auth_users,Debugging()), hostname='127.0.0.1', port=8025)
    controller.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        controller.stop()

if __name__ == '__main__':
    
    # Example auth data
    auth_users = {
        "user1": "pass1",
        "user2": "pass2",
    }

    asyncio.run(amain(auth_users))
```

in this snippet, i'm doing a few things. firstly, i'm subclassing aiosmtpd’s `SMTP` class to make our authentication handling. i've added `auth_login` and `auth_plain`. the `auth_login` is pretty simple: it sends the encoded requests for credentials from the client, and is followed by the `auth_plain` that does the base64 decoding and credential checking from a dictionary. notice, we return `AuthResult`. this result is what `aiosmtpd` actually reads to know whether to allow or deny access. this setup is basic but it illustrates the core logic.

to change the users from the cli, you just need to read the config from a file, edit it, and reload it into the server, or just create a new config. remember, we are not updating in real time. you have to stop the server and restart with the new data to reflect the new user credentials.

now you might be thinking, a dictionary is not very robust, and you are right. you should not have your usernames and passwords in plain text like that in production, or even in test for too long, but its practical for experimentation and proof of concept. let's get that sorted with json, but a better approach is to use some sort of user management tool that writes to a file. the idea is to pick up the data from a file, to allow cli manipulation and this approach is still faster than setting up a full blown database just to test.

so, here is an improved snippet that loads authentication data from a json file, and you can easily modify it from the cli. you just edit the json and restart the server.

```python
import asyncio
import json
import base64
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Debugging
from aiosmtpd.smtp import SMTP, AuthResult
import os

AUTH_FILE = "auth.json"  # File to store authentication data

def load_auth_data():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

class CustomSMTP(SMTP):
    def __init__(self, auth_users, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_users = auth_users

    async def auth_LOGIN(self, server, args):
        if not args:
            await server.push(b"334 " + b"VXNlcm5hbWU6")
            return None

        username = args[0].decode()

        await server.push(b"334 " + b"UGFzc3dvcmQ6")
        return ("auth_plain", username)

    async def auth_PLAIN(self, server, args):
        if not args:
            await server.push(b"501 Syntax error")
            return False

        auth_string = args[0].decode()
        try:
            auth_data = base64.b64decode(auth_string).decode().split('\x00')
            username = auth_data[1]
            password = auth_data[2]
        except (IndexError, ValueError):
            await server.push(b"501 Syntax error")
            return False

        if username in self.auth_users and self.auth_users[username] == password:
            return AuthResult(success=True, auth_ident=username)

        await server.push(b"535 Authentication failed")
        return AuthResult(success=False)

    async def auth_CRAM_MD5(self, server, args):
        # implement cram md5 if you wish, same logic here
        await server.push(b"502 Authentication method not implemented")
        return AuthResult(success=False)

async def amain():
    auth_data = load_auth_data()
    controller = Controller(CustomSMTP(auth_data, Debugging()), hostname='127.0.0.1', port=8025)
    controller.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        controller.stop()

if __name__ == '__main__':
    # Create initial auth data if the file doesn't exist
    if not os.path.exists(AUTH_FILE):
        initial_data = {
            "user1": "pass1",
            "user2": "pass2"
        }
        with open(AUTH_FILE, 'w') as f:
            json.dump(initial_data, f)

    asyncio.run(amain())
```

in this one, the `load_auth_data` function loads user data from `auth.json`. the rest is pretty much the same as the last example. you just need to create the json file first. and now, whenever you want to add a user, you just edit the json file, restart the server, and you are good to go.

but what if we have multiple domains, each with their own user data? things can get messy if you have all the users in a single json. we need something more robust, even if we are avoiding a full database. here is another example, with a file per domain.

```python
import asyncio
import json
import base64
import os
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Debugging
from aiosmtpd.smtp import SMTP, AuthResult

AUTH_DIR = "auth_data"  # Directory to store per-domain auth data


def load_auth_data(domain):
    auth_file = os.path.join(AUTH_DIR, f"{domain}.json")
    if os.path.exists(auth_file):
        with open(auth_file, 'r') as f:
            return json.load(f)
    else:
        return {}

class CustomSMTP(SMTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def auth_LOGIN(self, server, args):
        if not args:
            await server.push(b"334 " + b"VXNlcm5hbWU6")
            return None

        username = args[0].decode()
        domain = server.session.envelope.mail_from.split('@')[-1]
        server.auth_state = ("auth_plain", username, domain)
        await server.push(b"334 " + b"UGFzc3dvcmQ6")
        return ("auth_plain", username, domain)



    async def auth_PLAIN(self, server, args):
        if not args:
            await server.push(b"501 Syntax error")
            return False
        
        if not server.auth_state or len(server.auth_state) !=3:
             await server.push(b"501 Syntax error")
             return False
        
        _,username,domain = server.auth_state

        auth_string = args[0].decode()
        try:
            auth_data = base64.b64decode(auth_string).decode().split('\x00')
            auth_username = auth_data[1]
            password = auth_data[2]
        except (IndexError, ValueError):
            await server.push(b"501 Syntax error")
            return False
            
        if auth_username != username:
            await server.push(b"535 Authentication failed")
            return AuthResult(success=False)
        
        auth_users = load_auth_data(domain)
        if username in auth_users and auth_users[username] == password:
            return AuthResult(success=True, auth_ident=username)

        await server.push(b"535 Authentication failed")
        return AuthResult(success=False)
    
    async def auth_CRAM_MD5(self, server, args):
      await server.push(b"502 Authentication method not implemented")
      return AuthResult(success=False)


async def amain():
    controller = Controller(CustomSMTP(Debugging()), hostname='127.0.0.1', port=8025)
    controller.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        controller.stop()


if __name__ == '__main__':
     # Create the auth data directory if it doesn't exist
    if not os.path.exists(AUTH_DIR):
        os.makedirs(AUTH_DIR)

    # Create example auth files for different domains
    example_domains = ["domain1.com", "domain2.com"]
    for domain in example_domains:
        auth_file = os.path.join(AUTH_DIR, f"{domain}.json")
        if not os.path.exists(auth_file):
            initial_data = {
                "user1": "pass1",
                "user2": "pass2"
            }
            with open(auth_file, 'w') as f:
                json.dump(initial_data, f)
    asyncio.run(amain())
```

here, i'm loading authentication data based on the email domain. so if a user `user1@domain1.com` tries to authenticate, the server will look up `auth_data/domain1.com.json` for the username and password. i store this domain in the server using the `server.auth_state` and pass the username and the domain to the `auth_plain`. the flow is still mostly the same. but it gives you more control of your users, specially if you have multiple domains that you handle. i would actually use this method for my small personal projects, as its super flexible and does not require the hassle of setting up complex authentication mechanism.

and there we have it, three approaches to the same problem. if i would pick one of the three, i would go with the third, it's more organized. remember, this is just scratching the surface. the code can, and should, be improved with proper error handling, better logging, and other security measures. you need to think that those are only for testing and small project situations. don't trust passwords to json files in production environments. in those situations, you should consider a database or a dedicated user management system. but for a quick way to get authentication running, these examples are perfect.

for further reading on email server security, i'd recommend the rfc series on smtp, and the book "smtp: a guide for system administrators", by richard r. stevens, it still has very valuable info. it's a bit old, but the basics are the basics, and those remain relevant. also, check out "practical internet security" by mark graham, its more general, but it has some very interesting insight into how to properly secure internet services.

remember, testing is key. set up a simple testing environment and try sending email with authenticated users, this way you’ll quickly discover any issues with your setup.

one thing i always tell my colleagues: “debugging is like being a detective in a crime scene, but the victim is your code, not a human.” and trust me, debugging email systems can be a particularly puzzling case. you have to be very meticulous, every little detail matters.

good luck, and let me know if you run into any more roadblocks. i've seen them all, so i can help.
