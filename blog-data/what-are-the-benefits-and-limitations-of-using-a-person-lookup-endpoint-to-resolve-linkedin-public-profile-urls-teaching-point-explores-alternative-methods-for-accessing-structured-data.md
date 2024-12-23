---
title: "What are the benefits and limitations of using a Person Lookup endpoint to resolve LinkedIn public profile URLs? (Teaching point: Explores alternative methods for accessing structured data.)"
date: "2024-12-12"
id: "what-are-the-benefits-and-limitations-of-using-a-person-lookup-endpoint-to-resolve-linkedin-public-profile-urls-teaching-point-explores-alternative-methods-for-accessing-structured-data"
---

 so like we're talking about grabbing linkedin profile urls and i'm guessing we're not talking about just scraping the site cause that's a mess we're talking about the thing linkedin offers a 'person lookup' endpoint and yeah that sounds convenient right

the big win with the person lookup is definitely speed and structure it's like getting handed a neatly formatted file instead of rummaging through a messy closet it's supposed to be optimized for what you want which is to find data based on some identifier like an email or a name plus the format is consistent so you don't have to write a bunch of error handling stuff to parse different page layouts or weird html things that change every other week you know what i mean

that endpoint means they're doing the heavy lifting of looking up the person on their massive database and then they give you back specific bits of information you've asked for like their public profile url or their name or job title or whatever they allow you to access and it cuts down on load on your system cause you don't need to do a full web page scrape every time

but that's also where the limitations come in its a walled garden controlled access only they decide what information you can get at they decide how often you can make these requests they decide when they wanna change the format or API they might throttle you if they think you're requesting too much and you're just at their mercy so they can just cut you off if you do something they don't like or if they change the rules even the smallest change can break your whole pipeline of data fetching and that can be a big headache to fix

and then there's the whole public vs private thing cause even though its a 'person lookup' the results are going to be limited to what the person has made publicly available which means no hidden stuff or sensitive information they're not handing that out and you might have to request additional permission for different information and it adds another layer of complexity for how you handle data if you get access to more fields and might need more security measures in place

and then the cost some of these linkedin endpoints are not exactly free you might have to pay for the right to get data or pay for additional queries and that can add up fast so you have to balance convenience and accuracy against budget and that can be a major consideration

and when we compare it to other ways like web scraping yeah scraping is more flexible in theory you can grab any data that is visible on a public profile that's the positive right but it's also super fragile like i was saying their page structure might change they may try to block scraping attempts it can be very slow and error prone and it's not easy to maintain but its also free if you put in the time to write the scraper and if you're very careful and mindful to avoid making too much requests and the ethics are complicated we're going into a grey area so it makes things uncertain sometimes

and then there are other methods like using data brokers but they have their own issues again cost is a factor and it's a whole other mess of ethical considerations like where they got that data and is it accurate and is it even legal for them to sell it you have to be aware of data privacy legislation

so you have to think what are your real needs how much data do you need to get how accurate does it need to be and how quickly do you need the data and how much can you afford and if there is even a public access of information available because people could make their profile private and even with the API it will not be available

think about these things before you choose a method

so maybe you're looking at the linkedin endpoint but then you need to parse the data it returns to find the actual URL here's a quick example using python and the json library let's say the person lookup gave you this json back

```python
import json

data = '''
{
  "person": {
    "profile_url": "https://www.linkedin.com/in/some-user/",
    "name": "Some User",
    "headline": "Tech Enthusiast"
  }
}
'''
parsed_data = json.loads(data)
profile_url = parsed_data["person"]["profile_url"]

print(profile_url) # Output: https://www.linkedin.com/in/some-user/
```

that's a simple example right but what if we dont have the key of `"person"` or `"profile_url"` and we are not sure and what we want to be more generic

```python
import json

def extract_profile_url(json_string):
    try:
        data = json.loads(json_string)
        for key, value in data.items():
            if isinstance(value, dict):
                if "profile_url" in value:
                    return value["profile_url"]
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, dict) and "profile_url" in v:
                            return v["profile_url"]
        return None
    except json.JSONDecodeError:
        return None

data1 = '''
{
  "person": {
    "profile_url": "https://www.linkedin.com/in/some-user/",
    "name": "Some User",
    "headline": "Tech Enthusiast"
  }
}
'''

data2 = '''
{
  "results":
  {
        "user1": {
           "profile_url": "https://www.linkedin.com/in/other-user/",
           "name": "Other User"
            }
    }
}
'''
data3 = '''
{
  "data":
  {
        "items": [
         {
             "profile": {
              "profile_url": "https://www.linkedin.com/in/another-user/",
             "name": "Another User"
                }
            }
         ]
    }
}
'''


url1 = extract_profile_url(data1)
url2 = extract_profile_url(data2)
url3 = extract_profile_url(data3)
print(url1) # Output: https://www.linkedin.com/in/some-user/
print(url2) # Output: https://www.linkedin.com/in/other-user/
print(url3) # Output: https://www.linkedin.com/in/another-user/
```
see how we are going deeper into the json to try to find the profile url and return `None` if we can't find it that's more robust

and if you're scraping you might use something like beautifulsoup and requests in python to fetch the page and then use css selectors to grab the url here is a basic and simple example:
```python
import requests
from bs4 import BeautifulSoup

def scrape_linkedin_profile_url(linkedin_url):
    try:
        response = requests.get(linkedin_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        profile_link = soup.find('link', {'rel': 'canonical'})

        if profile_link:
            return profile_link['href']
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

url_to_scrape = "https://www.linkedin.com/in/some-user/" #this is a dummy
scraped_url = scrape_linkedin_profile_url(url_to_scrape)
if scraped_url:
    print(f"Scraped URL: {scraped_url}") # Output: Scraped URL: https://www.linkedin.com/in/some-user/

```

but it's brittle if linkedin changes its structure that won't work and you might need to change that code frequently to keep up with website updates

for going deep into how APIs work in general you should check out "RESTful Web APIs" by Leonard Richardson and Sam Ruby it gives a super good foundation for building your own APIs but also using them effectively it goes deep into things that are important like idempotence request methods and so on

then if you are really interested in the legal and ethical considerations regarding data access i'd suggest "Privacy is Power" by Carissa Veliz it provides a good starting point to the ethics of data collection and access and all the implications that exist these days

so yeah it's a balancing act there is no silver bullet for accessing public profile urls you need to look at the tradeoffs of each method that's really it
