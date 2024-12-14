---
title: "What is the Mailchimp webhook payload for an abuse or spam unsubscribe?"
date: "2024-12-14"
id: "what-is-the-mailchimp-webhook-payload-for-an-abuse-or-spam-unsubscribe"
---

alright, so you're diving into mailchimp webhook payloads, specifically for abuse or spam unsubscribes. i've been there, trust me. i remember back in '13, when i was knee-deep in a startup's email marketing, we had a similar issue. we needed to react *fast* to those abuse reports to keep our sending reputation clean. it's not just about knowing *if* someone unsubscribed; it's about the *why*, specifically when mailchimp flags it as abuse or spam. let's break down what that payload looks like.

mailchimp's webhooks are json, and when someone marks an email as spam or reports abuse, the payload you receive is going to follow a similar structure. it'll include quite a bit of data, but the key here is pinpointing exactly where that unsubscribe originated: was it voluntary, or triggered by abuse actions?

from what i’ve seen, the specific field you’re looking for is often within the `data` object, and usually comes in a `reason` or `type` field. the `action` field on the main payload will say `unsubscribe` but you’ll need to look further to understand the details. it's not just a simple flag, it’s often part of a wider data structure.

here’s an example of what the payload might look like. remember mailchimp's api can have version differences, but the overall idea remains consistent:

```json
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
    "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "abuse",
    "campaign_id": "campaign_id_1234",
    "ip_opt": "192.168.1.1",
    "ip_signup": "192.168.1.1"
    
  }
}
```

in this example, the `reason` field within the `data` object is set to `"abuse"`. this is your flag. this clearly indicates that the unsubscribe wasn't a standard request; the user went out of their way to report the email. also, note that i am using json format with consistent keys. mailchimp doesn’t always have all these keys depending on the event.

now, let's suppose you want to extract this data using python and a bit of error handling.

```python
import json

def handle_mailchimp_webhook(payload_string):
    try:
        payload = json.loads(payload_string)
        if payload.get('type') == 'unsubscribe' and payload.get('data') and payload['data'].get('reason') == 'abuse':
            email = payload['data'].get('email')
            list_id = payload['data'].get('list_id')
            print(f"abuse reported for email: {email}, on list: {list_id}")
            #your processing logic goes here to handle abuse
        else:
           print('standard unsubscribe or non-related event')

    except json.JSONDecodeError:
      print("error: invalid json")

#example of use
test_payload = """
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
    "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "abuse",
     "campaign_id": "campaign_id_1234",
     "ip_opt": "192.168.1.1",
     "ip_signup": "192.168.1.1"
    
  }
}
"""
handle_mailchimp_webhook(test_payload)

test_payload2 = """
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
    "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "other",
     "campaign_id": "campaign_id_1234",
     "ip_opt": "192.168.1.1",
     "ip_signup": "192.168.1.1"
    
  }
}
"""
handle_mailchimp_webhook(test_payload2)
```
this python snippet will help you extract the email and list id, and prints to console if an abuse report is detected, it handles json decoding errors, also i added a second example with other reason, so you can see how the function works in standard unsubscribe too, not abuse. in your actual usage, you'd probably be storing that data in a database or queue for some follow up processing.

sometimes mailchimp might provide extra data. for example, if the user marks it as spam, there will be a flag that distinguishes it from abuse report, or there might be additional metadata related to the campaign that triggered this action. this is not always there and is a good practice to log all fields if possible.
here's a possible scenario that mailchimp might send when the user marks it as spam:

```json
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
     "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "spam",
     "campaign_id": "campaign_id_1234",
      "ip_opt": "192.168.1.1",
      "ip_signup": "192.168.1.1"
  }
}

```
in this example the "reason" field is set to "spam", this is useful to filter and understand the type of unsubscribe that was triggered.

dealing with webhooks can sometimes feel like a detective job, tracking down those little nuances in the data. once i spent a week trying to debug why my unsubscribe processor wasn't catching abuse reports only to find out that a specific filter in the mailchimp configuration was not configured correctly. i've seen developers spending a whole weekend trying to find if an email was sent or not with a wrong configured mailchimp account.

a good approach is to use a more generic handler, which is the one i would recommend. so you can log any type of unsubscribe event, this will make your life easier to debug.
```python
import json

def handle_generic_mailchimp_webhook(payload_string):
    try:
        payload = json.loads(payload_string)
        event_type = payload.get('type')
        if event_type == 'unsubscribe':
          reason = payload.get('data', {}).get('reason')
          email = payload.get('data',{}).get('email')
          list_id = payload.get('data',{}).get('list_id')
          print(f"unsubscribe detected for email: {email}, on list: {list_id} with reason:{reason}")
          #do some processing logic here
        else:
          print("other event detected:", event_type)
    except json.JSONDecodeError:
        print("error: invalid json")

#examples of use
test_payload = """
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
     "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "abuse",
     "campaign_id": "campaign_id_1234",
     "ip_opt": "192.168.1.1",
     "ip_signup": "192.168.1.1"
    
  }
}
"""
handle_generic_mailchimp_webhook(test_payload)

test_payload2 = """
{
  "type": "unsubscribe",
  "fired_at": "2024-03-08 12:34:56",
  "data": {
    "list_id": "your_list_id_here",
    "email": "user@example.com",
    "email_type": "html",
    "web_id": 123456,
     "merges": {
      "FNAME": "John",
      "LNAME": "Doe"
    },
    "reason": "other",
     "campaign_id": "campaign_id_1234",
     "ip_opt": "192.168.1.1",
     "ip_signup": "192.168.1.1"
    
  }
}
"""
handle_generic_mailchimp_webhook(test_payload2)

test_payload3 = """
{
  "type": "campaign",
  "fired_at": "2024-03-08 12:34:56",
    "data": {
        "campaign_id":"test_campaign_id_123"
     }
}
"""
handle_generic_mailchimp_webhook(test_payload3)
```

in this third example, a general handler is presented, you can log the event type, and log the specific unsubscribe reason to identify abuse or spam, and you can add additional logic in the `if` statement. in the example the handler detects campaign events too. this is a more general approach that will save you time.

for further learning, i'd recommend exploring books like "building microservices" by sam newman, which covers webhooks in general. also, there are a lot of academic papers discussing email deliverability that will cover topics of spam filtering, bounce rates, unsubscribe management etc. search on google scholar using terms like "email reputation","email bounce handling", "spam filters", "email deliverability best practices", "webhook implementation". don’t only rely on official documentation, read some formal papers, they give a more solid knowledge base and more background. they might seem theoretical but, in practical terms they are very useful to understand concepts. also, for json i recommend the official json documentation to learn nuances. and a good practice is to always check mailchimp official webhook documentation directly because versions change fast. that should give a good starting point for a more robust application.
remember to validate, validate and validate.
