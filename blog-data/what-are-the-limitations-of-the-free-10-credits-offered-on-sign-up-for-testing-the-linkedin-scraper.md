---
title: "What are the limitations of the free 10 credits offered on sign-up for testing the LinkedIn scraper?"
date: "2024-12-10"
id: "what-are-the-limitations-of-the-free-10-credits-offered-on-sign-up-for-testing-the-linkedin-scraper"
---

 so you're hitting me with the LinkedIn scraper free credits thing right  ten credits sounds kinda stingy right off the bat  like seriously ten  what am I gonna do with ten credits I'm gonna scrape like three profiles tops before I'm out of juice  that's the main limitation right there  it's a total tease  they dangle this free trial in front of you  make you think you can get a feel for the thing  then bam you're scraping pennies  you're scraping air  you're scraping the bottom of the barrel basically

Another huge limitation is the data you *can* get with those ten measly credits  I'm guessing you're not getting the whole enchilada  probably only basic stuff  name title maybe company  forget about detailed work history  education  skills  recommendations  all the juicy bits you actually want  that's locked behind a paywall for sure   they're not dumb  they know people will want more than a cursory glance at some profiles

Think about it  LinkedIn's a goldmine of data  recruiters use it  marketers  sales people everyone  and they don't give away the good stuff for free  no way no how  this free trial is purely for marketing purposes  to get you hooked  to show you a glimpse of what's possible and then make you pay up to unlock the full potential  It's like a sample pack of gourmet coffee  you get a teeny tiny cup  it's delicious you're hooked  but you need to buy the whole bag  same deal here

Speaking of what's possible lets talk code  because that's what we're all about here  right  I'm not going to give you some super slick production ready code  just some simple examples to illustrate how things might work  remember this is just a starting point   You'll need to adapt it based on the API  which will change  APIs change all the time  it's the nature of the beast

First off  assuming you've got your API key and stuff sorted you'll probably be using something like Python  it's the go to for this kind of thing  and you'll need a library  something like `requests` to actually make the API calls  Here's a super basic example


```python
import requests

api_key = "YOUR_API_KEY_HERE"  #Replace with your actual key

headers = {
    "Authorization": f"Bearer {api_key}"
}

url = "https://api.example.com/linkedin/profiles/scrape" #Replace with the actual API endpoint

payload = {
    "url": "https://www.linkedin.com/in/elonmusk" #Target profile URL
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")

```

This is super basic  it just sends a request to the  API  get's a response and prints it out  But you'll probably need error handling  rate limiting  and a bunch of other stuff to make this actually robust  plus you'll need to parse the JSON response to extract the data you want  this is just the skeleton


Next up  lets talk about rate limits   you'll almost certainly hit them with only ten credits  but even with a paid account you need to be mindful of this  LinkedIn's API probably has usage limits to prevent abuse  and you'll get throttled if you make too many requests too quickly   This means you might need to add delays between requests using something like `time.sleep()`   Here's a very basic way to illustrate the concept:

```python
import requests
import time

# ... (previous code) ...

for profile_url in profile_urls: #a list of profile urls you want to scrape
    payload = {
        "url": profile_url
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.status_code}")
    time.sleep(2) #wait 2 seconds between requests to avoid rate limiting
```

This adds a two-second pause after each request  but you might need to adjust this based on LinkedIn's specific rate limits you have to check the API documentation  You might get a  429 response (Too Many Requests)  if you go too fast  and handling that properly is key   Remember this  rate limiting is your friend   avoid getting blocked

Finally you'll probably want to store this data somewhere once you scrape it  a database  a CSV file  something  You don't want to lose it  Here's a simple example of writing to a CSV


```python
import requests
import csv

# ...(previous code)...

with open('linkedin_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['name', 'title', 'company']  #Add other fields as needed
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for profile_url in profile_urls:
      # ... (API call as before) ...
      if response.status_code == 200:
          data = response.json()
          #Extract relevant data from the JSON response, be sure to handle potential key errors!
          try:
            writer.writerow({'name': data['name'], 'title': data['title'], 'company': data['company']})
          except KeyError as e:
            print(f"Missing key in JSON response: {e}")

```


This just takes the data and puts it in a CSV file  but it's a starting point  You will likely need more sophisticated error handling  especially dealing with missing keys in the JSON  data cleaning  and potentially handling different data structures depending on what the API returns  It will also depend on exactly which API you are using  

Remember to consult the  official LinkedIn API documentation  if one exists  or  research third party scraping libraries  and be mindful of LinkedIn's terms of service  because if you violate them  you might get your account suspended or face legal action  Don't be a jerk  respect the platform  be responsible


For further reading I would suggest looking into  "Web Scraping with Python"  a very popular book  It has good coverage of the basics  Also look for papers on web scraping best practices and ethical considerations  I can't point you to specific ones offhand but searching on those topics will turn up many relevant resources  Always remember to check the robots.txt file of the website you are scraping  to see if scraping is permitted  and always be respectful of the website's terms of service.
