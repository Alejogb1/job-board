---
title: "What role does data enrichment play in building robust LinkedIn scraping workflows without active sessions? (Teaching point: Explains the value of combining multiple data sources.)"
date: "2024-12-12"
id: "what-role-does-data-enrichment-play-in-building-robust-linkedin-scraping-workflows-without-active-sessions-teaching-point-explains-the-value-of-combining-multiple-data-sources"
---

Okay so building robust LinkedIn scraping without live sessions that's the tricky part right scraping LinkedIn directly when you're not actually logged in its a whole different ballgame than when you're just browsing normally. Data enrichment becomes super important for that. You can’t just grab the bare minimum info from the page and hope for the best. Think of it like this LinkedIn shows you a snippet of a profile right its not the complete picture. You want more than that you need depth you need context and that's where data enrichment comes in.

You're basically trying to reconstruct a more complete user profile from limited public data or data that you've scraped earlier or data that's stored somewhere. Imagine you got a user's name and job title from a public page that’s step one. It’s like having a rough sketch and data enrichment is like filling in the colors and details. So say you get `John Doe Software Engineer at Google` the basic data but thats it. It's barebones it’s missing a lot of potential data points like their skill set education background past job history and that's where enrichment kicks in.

You can enrich this data using a variety of sources. It's not just about scraping another website directly to get a full profile it's more complex than that. It might mean consulting other public APIs or even just using internal datasets you've previously collected. Think of it as combining different sources to create a complete picture. For instance you might use data from common knowledge data sets or open data repositories to verify location if a location is not found on the page.

Data enrichment is basically saying okay we got some initial data now lets look for more relevant stuff that can help validate what we have or add missing data points. So maybe I'm looking for someone's email address and its not readily available on LinkedIn's page I might use other services that are focused on finding emails and see if that gives me some leads that would be enrichment through a 3rd party service and often you have to pay for those services but if it's important you know you can do it that way

Why is this crucial for scraping without live sessions Well its because when youre not logged in LinkedIn is extra stingy with the data. You only get the public basic stuff. If you're scraping public pages you aren't going to get deep detailed profile details that LinkedIn reserves for logged in users. So enrichment is about building on that initial very limited data. Its like making sure you have a strong foundation and using various tools to add value. Its less about the individual source that you have available and more about the process of making sure you have better data.

Let's say you are tracking tech trends. If you scrape basic info like user job titles and their company you can enhance that with additional skill information from sources that tracks software developers skill sets. So you are adding skills based on the fact that this person has this job title and works at that company. And you’re not getting that directly from LinkedIn. You’re basically creating a richer profile of skills that person is likely to have. Its all about making the data more actionable and not so much about getting the data from a single place. Its all about filling in gaps and making your data more complete.

Here's some python code that illustrates the idea

```python
import requests
import json

def enrich_with_geo_api(location_name):
  """Enriches location data using a public geo API."""
  api_url = f"https://api.opencagedata.com/geocode/v1/json?q={location_name}&key=YOUR_API_KEY"
  response = requests.get(api_url)
  if response.status_code == 200:
      data = response.json()
      if data['results']:
        first_result = data['results'][0]
        latitude = first_result['geometry']['lat']
        longitude = first_result['geometry']['lng']
        return {"latitude": latitude, "longitude": longitude}
  return None

user_profile_raw = {"location": "London UK"}
location_data = enrich_with_geo_api(user_profile_raw["location"])

if location_data:
  user_profile_raw.update(location_data)
  print(json.dumps(user_profile_raw, indent=2))
else:
  print("Could not enrich location data.")
```

This is a very simple example that is taking a location string and enriching with lat long. This can help with making sure you know the exact location. It is a common use case for enriching basic data. This is an example that takes something like `london UK` and gives you geo data. Think about how many other types of enrichments you can perform.

Another example

```python
import json

def enrich_with_skills(job_title, company_name):
  """ Enriches job title and company with common tech skills using a simple rule set """

  if "Software Engineer" in job_title and "Google" in company_name:
    return ["python","java", "go", "distributed systems"]
  elif "Data Scientist" in job_title:
      return ["python","R", "machine learning", "data mining"]
  else:
    return []

user_data = {"name":"Jane Doe","job_title": "Software Engineer", "company": "Google"}

skills = enrich_with_skills(user_data['job_title'], user_data['company'])
if skills:
    user_data["skills"] = skills

print(json.dumps(user_data, indent=2))

```
This example is a basic example but imagine a much larger dataset of rules that can derive skills for various jobs titles and companies. This is a form of enrichment that takes two data points like job title and company and derives new data based on a predefine rule set which is often much better then simple keyword searches.

Let’s look at a third example for email

```python

import requests

def enrich_email(name, company_name):
  """Attempt to get a public email using Clearbit API (API Key Required)."""
  api_key = "YOUR_CLEARBIT_API_KEY"
  url = f"https://person.clearbit.com/v2/people/find?name.fullName={name}&employment.company.name={company_name}"

  headers = {'Authorization': f'Bearer {api_key}'}

  response = requests.get(url, headers=headers)
  if response.status_code == 200:
      data = response.json()
      if data and 'email' in data:
        return data['email']
  return None

user = {"name":"John Doe","company_name": "Google"}

email = enrich_email(user["name"], user["company_name"])
if email:
   user["email"] = email
   print(user)
else:
    print("Could not find email for this person")

```

This takes an input of the name and company and tries to find a publicly available email. This is a 3rd party service that provides enrichment. This is a very common use case. It is very important to use 3rd party services for enrichment as that data is often very difficult to get and will greatly improve your overall scraping process especially when there is no active session.

You want to use enrichment to improve the quality and coverage of the data you collect so it allows you to build a more complete picture. Without enrichment your data set will be very limited and your results will be lacking. Think of enrichment as adding missing pieces to a puzzle. The puzzle is the complete user profile the initial scrape only gives you 1 or 2 pieces the rest you have to gather and put together using various methods like I’ve shown here. The more data you gather the higher the quality of the outcome and the better your scraping process is.

For further reading a great place to understand more about data integration and data quality concepts are books and research papers on data warehousing and data mining. There's a whole area of research dedicated to this. The book "Data Mining: Concepts and Techniques" by Jiawei Han is a classic in the field while "Building a Scalable Data Warehouse with Data Vault 2.0" by Dan Linstedt is good if you are thinking about a large dataset project and how to manage the data. Academic papers focusing on record linkage and entity resolution are also a good source to learn more about these topics. You can often find papers on google scholar.
