---
title: "How to add SharePoint list data or a Google sheet to Active Campaign lists?"
date: "2024-12-15"
id: "how-to-add-sharepoint-list-data-or-a-google-sheet-to-active-campaign-lists"
---

alright, so you’re looking at getting data from sharepoint lists or google sheets into active campaign lists, that’s a common challenge. i've been there, many times actually. feels like a rite of passage for any marketing automation setup, doesn’t it? integrating data sources like these into active campaign can get a bit fiddly, but it's not insurmountable. let's break it down.

first things first, direct integration via active campaign’s ui is usually not going to cut it for sharepoint or google sheets. there isn't a straightforward “import directly from sharepoint” or “pull from google sheet” button. what we're looking at is more of a programmatic approach, where we need to extract the data, transform it to fit active campaign's structure, and then push it in. this will involve writing some code or using middleware. i’ve tried a few tools over the years, from dedicated integration platforms to custom scripts, and i'll tell you my experiences as we go along.

**sharepoint lists:**

sharepoint, in my experience, always throws curveballs. it’s not just a simple database. getting the data out usually involves using the sharepoint api, specifically the rest api. this api allows you to query and retrieve list items in json format. once you get the hang of it, it becomes a familiar pattern but initially, it can feel overwhelming, i remember spending days with the sharepoint documentation. here’s the gist of it:

```python
import requests
import json

def get_sharepoint_list_data(site_url, list_name, client_id, client_secret, tenant_id):
    """fetches sharepoint list data using rest api."""

    auth_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
    resource = "https://{site_url}".format(site_url=site_url)
    
    auth_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'resource': resource
    }
    
    auth_response = requests.post(auth_url, data=auth_data)
    auth_response.raise_for_status() # check if request was successful
    auth_token = auth_response.json()['access_token']

    list_url = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"

    headers = {
        'Authorization': f'Bearer {auth_token}',
        'Accept': 'application/json;odata=verbose'
    }

    list_response = requests.get(list_url, headers=headers)
    list_response.raise_for_status() #check if request was successful
    
    return list_response.json()['d']['results']

# example usage
site_url = "https://yourcompany.sharepoint.com/sites/yoursite"
list_name = "your_list_name"
client_id = "your_client_id"
client_secret = "your_client_secret"
tenant_id = "your_tenant_id"

data = get_sharepoint_list_data(site_url, list_name, client_id, client_secret, tenant_id)
print(json.dumps(data, indent=4))

```
*important: never embed your secrets directly in the code, i’m showing a simplified version for this context. use environment variables or a secure vault*.

this python snippet uses the requests library to make api calls to sharepoint. you’ll need to register an application in azure active directory to get the client id and client secret. the response comes back as a json, which you can then parse. the actual structure of that json depends on how your list columns are structured, so you’ll have to adjust your code accordingly. now, once you've got the data, you need to massage it to fit active campaign. active campaign expects data in specific formats, usually a json structure like: `[{'email':'email@email.com','first_name':'john', 'last_name':'doe'}]`. so a little bit of transformation is required, and then pushing to active campaign’s api endpoints. if you’re feeling brave you could even try to send it in chunks to not hit the active campaign limits, that's what i had to do on my last project, it involved some headache but it worked.

**google sheets:**

google sheets are a bit easier to handle. they have an excellent api that’s less convoluted than sharepoint’s. if the data structure is simple then it’s even easier, basically a breeze, compared to the sharepoint option. the approach here involves using the google sheets api which you can access via google’s client libraries. python has an easy-to-use library for this, `google-api-python-client`. i remember when i first used this i thought: "wow, that's surprisingly easy". here’s how you’d typically go about it:

```python
from googleapiclient.discovery import build
from google.oauth2 import service_account
import json

def get_google_sheet_data(spreadsheet_id, range_name, credentials_file):
    """fetches google sheet data using the google sheets api."""

    creds = service_account.Credentials.from_service_account_file(credentials_file)

    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    
    keys = values[0]
    
    sheet_data = []
    for row in values[1:]:
      row_data = dict(zip(keys,row))
      sheet_data.append(row_data)
      
    return sheet_data

# example usage
spreadsheet_id = "your_spreadsheet_id"
range_name = "Sheet1!A1:Z" # or any specific range, like A1:B10
credentials_file = "path/to/your/credentials.json" # a google service account json file

data = get_google_sheet_data(spreadsheet_id, range_name, credentials_file)
print(json.dumps(data, indent=4))
```

the crucial part here is setting up the google service account and getting the `credentials.json` file, which is not very hard to do, thankfully. this gives your script access to the google sheets api. then, like with the sharepoint data, you will need to transform the data retrieved to meet the requirements of active campaign before sending it to the api. again, similar data requirements are expected in the active campaign side, but one advantage of the sheets approach is that you can easily manipulate the data directly in the sheet before importing it, if you only need simple changes. i learned this the hard way after spending an entire afternoon transforming data in code that could have been easily done in google sheets.

**pushing to active campaign:**

now, once you have your data prepared, you need to send it to active campaign. active campaign's api, luckily, is quite straightforward. you’ll primarily be using the contacts endpoint to create or update contacts, and then, if needed, you can add those contacts to specific lists. active campaign has a php library, but it's also very easy to interact with the api via http requests if you want, here's a simple example:

```python
import requests
import json

def add_or_update_contacts_active_campaign(api_url, api_key, contacts):
    """add or update contacts in active campaign."""

    headers = {
        'Api-Token': api_key,
        'Content-Type': 'application/json'
    }

    for contact in contacts:
       contact_data = {'contact': contact}
       response = requests.post(f"{api_url}/api/3/contacts", headers=headers, json=contact_data)
       response.raise_for_status() #check if request was successful

    return response

# example usage
api_url = "https://your_account.api-us1.com" # replace with your active campaign api url
api_key = "your_active_campaign_api_key"   # replace with your active campaign api key
contacts = [
     {'email': 'email1@example.com', 'firstName': 'john', 'lastName': 'doe'},
     {'email': 'email2@example.com', 'firstName': 'jane', 'lastName': 'smith'}
     # add more contacts as needed
]
result = add_or_update_contacts_active_campaign(api_url, api_key, contacts)
print(result.json()) # this will give you the details of the result of the request

```

you'll need to replace the placeholders with your active campaign api url and api key. remember to never commit your api key directly to any code repository. this is the most common mistake, and i've seen quite a few cases on stackoverflow of people exposing their keys. you can also use environment variables for that or a secret vault to store your api key in your production systems.

**important considerations:**

before i let you go on with your coding, there's a few considerations i've had over the years that i'd like to share:

*   **rate limits:** active campaign, sharepoint, and google sheets, all have api rate limits. be mindful of these to avoid getting your requests throttled or blocked. implement proper error handling and retry mechanisms if necessary.
*   **error handling:** be sure to include error checking in all aspects of your code. catching bad authentication, wrong schema, or invalid data before it reaches your api, will save time debugging.
*   **data types and fields**: make sure the fields you pull from sharepoint/google sheets match the active campaign fields data types. this will save you headaches when importing the data.
*   **security:** never expose your api keys or credentials in your code, use environment variables or secure vaults.
*   **data volume**: if you are processing a lot of data, consider batching your operations when you update in active campaign. sending thousands of requests at once might not be the most optimal route.
*   **data integrity**: data hygiene is essential. make sure to have a proper process to remove/update contacts that unsubscribed in active campaign. having bad data is worse than having no data.

**resources:**

as for additional learning resources: i’d recommend taking a look at these, as they helped me understand better some nuances:

*   for a general understanding of apis i recommend: “restful web services” by leonard richardson and sam ruby, it’s not focused on any specific technology but it covers a lot of the fundamentals.
*   the official microsoft graph api documentation for sharepoint: the documentation will be your constant companion on your sharepoint journey. they have specific samples for sharepoint lists in most languages.
*   the google sheets api documentation: is also well structured and easy to understand. their quickstarts are especially useful to get up and running.
*   active campaign's api documentation: is another great resource that you can use to check the specifics about the format needed for the requests.

that is basically it. remember this is all code; i’m not a fan of the term “code magic”, it’s just code, and debugging, and learning to make the best choices for your situation. hope that helps.
