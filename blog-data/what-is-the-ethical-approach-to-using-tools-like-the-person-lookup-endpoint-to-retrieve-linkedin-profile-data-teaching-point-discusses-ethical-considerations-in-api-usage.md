---
title: "What is the ethical approach to using tools like the Person Lookup endpoint to retrieve LinkedIn profile data? (Teaching point: Discusses ethical considerations in API usage.)"
date: "2024-12-12"
id: "what-is-the-ethical-approach-to-using-tools-like-the-person-lookup-endpoint-to-retrieve-linkedin-profile-data-teaching-point-discusses-ethical-considerations-in-api-usage"
---

Yo lets talk about snagging LinkedIn data with those person lookup endpoints man its a can of worms for real

First thing is like are we even allowed to be doing this at all LinkedIn's terms of service that's the first hurdle always gotta peek at those usually tucked away in some corner of the internet they're basically the legal rulebook telling you what you can and cannot do and messing with that is like knowingly stepping on a landmine no bueno especially when we are dealing with personal data remember that at the core its about respecting people's privacy and if those terms say 'no unauthorized scraping' or whatever well that's a red flag a big one

And just because something is technically doable doesnt mean its ethically sound we gotta go beyond the 'can we' and start asking 'should we' ethical hacking is all about this the idea that power needs responsibility its like having the keys to a super fast car cool right but you cant just go driving on the sidewalk right its about using tools wisely with intention and a sense of what's right

think about the implications man if you're slurping up someone's entire profile without them knowing thats a violation of their autonomy and its kinda creepy for real imagine someone doing that to you would you like it I highly doubt it its not a good feeling it feels invasive

then theres the whole data security thing how are you storing this info are you just dumping it into a spreadsheet or do you have proper encryption and access controls people trust their data on LinkedIn so we gotta be like a trustworthy guardian not some data pirate its a big responsibility the kind that keeps folks up at night thinking about leaks and breaches

lets go beyond the privacy aspect there's also this thing called bias these lookup tools dont operate in a vacuum they're only as good as the data they're trained on so if the underlying data is skewed well you're just gonna perpetuate that bias if you're trying to build some kind of AI-powered recruiting tool based on this data you might end up unintentionally discriminating against certain groups because of the data which is not cool at all it perpetuates inequality instead of combating it

and its about transparency too people deserve to know how their data is being used if you're pulling their profiles for some kind of purpose its kinda messed up to do it behind their backs being upfront about it shows respect and builds trust even if they're not stoked about your use case you're at least handling it with a semblance of honesty

sometimes the purpose can justify some of these ethical concerns but the bar has to be high a very high bar like are you using the data to improve accessibility for people with disabilities or to counter discriminatory hiring practices if its for the greater good and theres proper safeguarding in place well that changes the calculus but the default should always be to err on the side of caution and privacy

now lets dig into some of this code side a bit ok so lets say we are doing something like a basic HTTP get request to an api endpoint now this is not exactly the linkedin api but its just a hypothetical endpoint the core idea is the same we would have our headers something like this

```python
import requests

def get_profile_data(profile_id, api_key):
    url = f"https://example.com/api/v1/person/{profile_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "MyCoolApp/1.0 (you@example.com)", # be transparent about who you are
        "Accept": "application/json" # specify the data format
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # raise HTTP errors
        return response.json() # parse the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
```

in this example see the `User-Agent` header its super important to identify yourself to the api its good netiquette and shows you arent some bot trying to scrape blindly its also a reminder that you should have some kind of contact point like an email in there also `raise_for_status` is a nice one for doing error checking its not just about whether the request succeeded or not but also if we had any kind of HTTP error so a 404 or 500 we'd be able to track that in the exception

ok another idea think about rate limiting and pagination when we're pulling data from these apis you dont wanna overload the server its not a polite thing to do we need to introduce some kind of delay between our requests and handle cases where data is split over multiple pages lets look at some code

```python
import time

def fetch_profiles_paginated(api_key, base_url, per_page=20, delay=1):
    profiles = []
    page = 1
    while True:
        url = f"{base_url}?page={page}&per_page={per_page}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if not data:
                break # exit when there are no more profiles
            profiles.extend(data)
            page += 1
            time.sleep(delay) # polite delay to avoid rate limits
        except requests.exceptions.RequestException as e:
            print(f"Error while fetching page {page}: {e}")
            break # exit on failure
    return profiles

```

so this one introduces pagination and a time delay using `time.sleep` see how its trying to be polite with a default of `1` second between requests its like knocking on a door you dont want to bang on it repeatedly we also have an exit clause when we hit an empty page so we stop sending requests plus error handling just in case things go south and of course that all this hinges on our code being authorized through the header

and then its worth discussing how you might store the data safely lets assume for a sec that the data returned is a list of dictionaries each representing a user so its json data to persist into a more robust format consider something like this

```python
import json
import os

def save_profiles_to_json(profiles, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(profiles, f, indent=4)
    except Exception as e:
        print(f"Error saving profiles to file: {e}")

```

now here we're writing the list of dictionaries into a json file and we are specifying some formatting via the `indent=4` which helps improve readability of the json file but this is just local storage not ideal for sensitive data this file could be compromised and we dont want to keep credentials in the code ideally so we should store it elsewhere via an environment variable instead of directly in the code something more akin to a key vault or an encrypted storage solution

as for resources its not about links but more about concepts so check out the book "Ethics and Data Science" by DJ Patil, Hilary Mason and Mike Loukides it's a really thoughtful dive into the ethical implications of data science and the kinds of things that can happen when you start handling data that other people trust to keep safe

another good read is the Belmont Report which is kinda the cornerstone for ethical principles in human research its a bit more academic but its got some universal principles in there that apply to pretty much anything where human subjects are involved and its very useful to see how these things came to be

for the technical side read up on the OWASP API Security Project it has really practical guides and checklists on how to secure your APIs which is directly related to the issue of data security if you're pulling data from an api you also need to know what to do when you become an api provider and you need to ensure you are providing data safely

and then there's like ongoing conversation around privacy you really want to keep tabs on regulations like GDPR and CCPA so you are always up to date its an evolving space and you gotta stay nimble and adapt as new challenges arrive

so yeah ethically snagging linked in data its a tangled mess we need to think about legal rights data security potential biases and just basic respect for others privacy its not enough to just see if something is technically possible we gotta stop and consider if its actually the right thing to do because if we dont we are going to create more problems than we are fixing

thats my brain dump on this topic peace out for now
