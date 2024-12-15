---
title: "How to add SharePoint list data or Google sheet to Active Campaign lists?"
date: "2024-12-15"
id: "how-to-add-sharepoint-list-data-or-google-sheet-to-active-campaign-lists"
---

alright, so you're looking to get data from sharepoint lists or google sheets into active campaign lists, huh? i've been down this road a few times, and it's not always a walk in the park. it usually boils down to a few key steps: fetching the data, transforming it, and then pushing it into active campaign. let's break this down.

first off, fetching the data. for sharepoint, you’re usually dealing with the sharepoint rest api. you'll need to authenticate, of course, which can involve some dance with azure active directory, and then construct the correct api calls. i remember my first time using the sharepoint api, i spent a solid afternoon just getting the authorization header correct. it was a headache. here's a snippet of what that might look like with python using the requests library for the api calls:

```python
import requests
import json

def fetch_sharepoint_list_data(site_url, list_id, client_id, client_secret, tenant_id):
    # construct the authentication url
    auth_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "resource": "https://<your_company>.sharepoint.com" # replace it
    }
    auth_response = requests.post(auth_url, data=auth_data)
    auth_response.raise_for_status() # raise error if the request failed
    auth_token = auth_response.json()["access_token"]

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json;odata=verbose"
    }
    api_url = f"{site_url}/_api/web/lists(guid'{list_id}')/items"
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()

    return response.json()["d"]["results"]

# example usage (replace the placeholders with your actual values)
site_url = "https://<your_company>.sharepoint.com/sites/<your_site>"
list_id = "<your_list_guid>" # example: "7a29c567-9538-4186-9647-514325637633"
client_id = "<your_client_id>" # from azure app registration
client_secret = "<your_client_secret>" # from azure app registration
tenant_id = "<your_tenant_id>" # from azure app registration
sharepoint_data = fetch_sharepoint_list_data(site_url, list_id, client_id, client_secret, tenant_id)
print(sharepoint_data) # output
```

remember, you'll need to set up an app registration in azure active directory and grant it the necessary permissions for the sharepoint site. that part always seems simple on paper, but it’s got its own intricacies.

now, for google sheets, it's a bit less involved in terms of raw api calls. you'll use the google sheets api, and you usually authenticate with a service account and a json key file. i recall setting this up once for a weekly report export – got it done, but i forgot to set the timezone properly, and all the dates were off by a few hours. that was fun to debug at 2am. here’s some python code, this time using the `google-api-python-client` library:

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

def fetch_google_sheet_data(spreadsheet_id, range_name, key_file_path):
    creds = service_account.Credentials.from_service_account_file(
        key_file_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    return values

# example usage (replace placeholders with your actual values)
spreadsheet_id = "<your_spreadsheet_id>" # "1vQ8a9X8g_yJ1jB5t3-y7U_x-1Z2i3K5c_uJvF8q"
range_name = "Sheet1!A1:Z"
key_file_path = "path/to/your/google_service_account_key.json" # local path
google_sheet_data = fetch_google_sheet_data(spreadsheet_id, range_name, key_file_path)
print(google_sheet_data)
```

after you've fetched your data, you'll need to transform it to match the format active campaign expects. typically, active campaign needs fields like email, first name, last name, maybe some custom fields. depending on how the data is structured in sharepoint or google sheets, you might have to map and rename columns. this part involves some data massaging, perhaps using python dictionaries, or using something like pandas if you're handling a large volume. i once used a ton of nested `if` statements for this step and it was a terrible idea; ended up refactoring it with a lookup dictionary, much cleaner.

now, let's talk about pushing the data into active campaign. active campaign provides an api, and you'll typically use the contact creation or update endpoint. there are a few gotchas here. if a contact already exists, you usually update it instead of creating a duplicate. also, keep an eye on rate limits; they're real, and you don’t want to get your requests throttled. here's a simplified example using python's requests library, again, showing how you can create or update contacts:

```python
import requests
import json

def add_or_update_active_campaign_contact(api_url, api_key, contact_data):
    headers = {
        "Api-Token": api_key,
        "Content-Type": "application/json"
    }
    # check if the contact already exists, for example, using the email as an identifier
    email = contact_data.get("email")
    get_contacts_url = f"{api_url}/api/3/contacts?email={email}"
    get_response = requests.get(get_contacts_url, headers=headers)
    get_response.raise_for_status()
    existing_contacts = get_response.json().get("contacts", [])

    if existing_contacts:
        # update the existing contact with provided fields if contact exists
        contact_id = existing_contacts[0].get("id")
        update_url = f"{api_url}/api/3/contacts/{contact_id}"
        payload = {"contact": contact_data}
        response = requests.put(update_url, headers=headers, data=json.dumps(payload))
    else:
        # add a new contact if no contact exists in the api
        payload = {"contact": contact_data}
        response = requests.post(f"{api_url}/api/3/contacts", headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()

# example usage
api_url = "https://<your_account>.api-us1.com" # replace with your active campaign url
api_key = "<your_active_campaign_api_key>" # replace with your active campaign key
example_contact_data = {
    "email": "test@example.com",
    "firstName": "john",
    "lastName": "doe",
    "fieldValues": [{"field": "1", "value": "test value"}], # if you have custom fields
    "listids": [1]
}
active_campaign_response = add_or_update_active_campaign_contact(api_url, api_key, example_contact_data)
print(active_campaign_response)
```

some things you might want to consider additionally. if you are moving from sharepoint or sheets to active campaign i assume you want some data consistency on a routine basis. you will probably need to automate this with something like cron jobs or scheduled tasks. also, error handling is very important. log errors and consider retries, especially with api calls. you might also need to manage data volumes effectively. if you are moving tens of thousands of contacts, doing this in batches helps. there is always that one contact that screws everything, in my experience. i swear i saw a contact with a weird unicode character email once and it broke the entire batch job.

for further reading, i'd suggest looking into "restful web apis" by leonard richardson for general rest api best practices. for details on handling data manipulation with python, "python for data analysis" by wes mckinney is a solid choice. and of course, thoroughly reviewing the active campaign api documentation is crucial, as it can and does change from time to time. also, keep the sharepoint and google api docs handy; you’ll be needing them.

that's the gist of it, more or less. it’s a process involving careful fetching, transformation, and pushing data. i hope this helps in your data integration journey and avoids some of the "fun" i’ve had over the years.
