---
title: "allure attachment test automation reporting?"
date: "2024-12-13"
id: "allure-attachment-test-automation-reporting"
---

Okay so you're asking about allure attachments in test automation reporting specifically the tricky parts I've been around the block a few times with this so let's dive in real quick

First things first you gotta understand allure isn't just some magic black box that spits out pretty reports it's a framework that needs data and that data primarily comes from attachments you're adding during your tests Think of it like this if your test is a detective then the attachments are the evidence they collect about what went wrong or right during the investigation If you fail to gather evidence your reports aren't gonna be worth much right

Alright let me tell you a story about a time I was fighting with this stuff I was working on this massive e-commerce platform remember these days people actually had to write code rather than ask ai to do it and we had a crapton of end-to-end tests written in selenium god bless those poor things The issue was the allure reports they were beautiful but they lacked the juicy bits no one could ever know what exactly went wrong we had some screenshots sure but they were after the test failed not during The core issue was I wasn't being proactive with my attachments I was just catching the failure screenshot and that's all I had to work with for debugging a massive test suite that ran for 4 hours That drove me absolutely nuts

The big revelation came after reading this paper on observability in automated testing which is somewhere in the depths of IEEE's digital archives you know the usual suspects I really do recommend checking it out it will change the way you test I started thinking what if i attached EVERYTHING I mean every relevant piece of information during the test execution not just on failure Why not dump the html source the browser logs api responses the whole nine yards It might be overkill I know but when a test fails you really do appreciate having all the data right in front of you

So what I did was to sprinkle attachment functionality all throughout my testing code. Let’s see some code snippets to show the approach I took back then

Here's how i was dealing with simple text attachments I was using python back then

```python
import allure

def log_info(message):
    allure.attach(message, name="log.txt", attachment_type=allure.attachment_type.TEXT)

# Example use
log_info("This is a test log message detailing the current state of the application")
```

Simple right But the real gold lies in attaching more complex stuff. For instance here's an example of how I'd grab a screenshot from selenium and attach it

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
import allure

def capture_and_attach_screenshot(driver, step_name):
    allure.attach(driver.get_screenshot_as_png(), name=f"{step_name}.png", attachment_type=allure.attachment_type.PNG)


#Example use
driver = webdriver.Chrome() #or any browser you like
driver.get("https://www.example.com")
driver.find_element(By.TAG_NAME, 'body').click()
capture_and_attach_screenshot(driver, "Example_screenshot")
driver.quit()

```

As you can see we take a screenshot and convert it to a png and then attach that to the allure report This was just the beginning i wanted moooore

Then came the fun part dealing with network requests because that e-commerce platform I was talking about heavily relied on api calls and debugging these was always a painful experience I was making my own api client using python's requests and i was really going deep into the request response bodies

```python
import requests
import allure
import json

def attach_request_response(url, method, headers, data, response):
    request_details = {
        "url": url,
        "method": method,
        "headers": headers,
        "data": data if isinstance(data, str) else json.dumps(data, indent=2)
    }

    allure.attach(json.dumps(request_details, indent=2), name="request.json", attachment_type=allure.attachment_type.JSON)
    
    response_details = {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response.text if len(response.text) < 300 else response.text[0:299] + " ..."
    }
    allure.attach(json.dumps(response_details, indent=2), name="response.json", attachment_type=allure.attachment_type.JSON)

# Example of a call and attachment
url = "https://api.example.com/products"
headers = {'Content-type': 'application/json'}
data = {"product_id": 123}
response = requests.post(url, headers=headers, json=data)
attach_request_response(url, "POST", headers, data, response)
```

Notice something This code attaches both the request and response information to allure using json this was so good for pinpointing issues with api calls

The key point here is to attach context not just data. So I started attaching data about the user the environment and the test parameters everything which might be relevant to debug an issue

Now I hear you saying ‘Okay that's a lot of attachments’ yes it is i know it can slow your tests and yes you gotta be smart about it. I started implementing conditional attachments for example only attaching detailed api responses if the test fails and when it passes only attach a summary response or log

The second hurdle which i had to tackle was how to make the attachments human readable. No one wants to dig through raw json objects or random strings That's when I started creating custom attachment functions each one designed to generate the information i needed in a presentable format For instance I made custom json formatting functions i started to add headers to my files that contained info about test step time and also the file path for easy debugging. The other thing I did was using some markdown i was able to add tables and lists within my text attachments to present data in a well formatted way

The last thing I did was to be consistent with naming conventions It's all fine and dandy if you attach 100 attachments but if those attachments are named like `screen1png.png screen2png.png` and `apires1.json apires2.json` no one is gonna be able to make sense of anything I was very careful with the names in each file so i can correlate failures with steps.

And before I finish this wall of text remember this once you master this attaching game it's no longer just about finding the failures but also knowing exactly *why* they failed. This makes your testing much more effective and your debugging process much less painful

Also funny story once I added an attachment with a cat picture when debugging a strange issue the bug was so obscure that i thought i was going crazy at that point I was looking at the test report saying 'well at least the cat picture is still nice' I never did it again I promise.

I would say to you check out the book 'Software Testing Techniques' by Boris Beizer it really helped me a lot in understanding how testing can be enhanced with well designed data extraction. Also look at the allure documentation itself there are a lot of useful details in there which you can benefit from.

And that's my journey with Allure attachments I hope this helps you in your adventures of test automation good luck out there and happy testing.
