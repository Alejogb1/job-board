---
title: "How to Run a python script with json input from a rails application?"
date: "2024-12-14"
id: "how-to-run-a-python-script-with-json-input-from-a-rails-application"
---

alright, so you're looking to get a python script to play nice with json data sent over from a rails app. i’ve been there, done that, got the t-shirt, and the server logs that screamed at 3am. it’s a pretty common scenario, and there are a few ways to tackle it, each with its own quirks. let’s break it down.

first off, the core issue is how to get that json data out of your rails app and into the waiting hands of your python script. you could go down a few paths, but i’ve found that keeping things simple is usually the best approach to avoid unnecessary headaches down the line. when i first started, i had this crazy idea to use websockets, only to discover it was overkill for what i needed. i spent a good week just trying to get a simple json structure sent over, just to feel very embarrassed when i re-evaluated my approach. lets skip that.

here is what has worked for me, it involves leveraging http requests with rails, and then consuming that data on the python side.

**the rails side (sending the json)**

in your rails controller, you'll need an action that spits out json. this is pretty standard rails stuff:

```ruby
# app/controllers/my_controller.rb
class MyController < ApplicationController
  def data_endpoint
    data = {
      'some_key': 'some_value',
      'another_key': 123,
      'list_of_items': [
         { 'item_key': 'item_value_1'},
          {'item_key': 'item_value_2'}
        ]
    }
    render json: data
  end
end
```
```ruby
# config/routes.rb
Rails.application.routes.draw do
  get '/my_data', to: 'my#data_endpoint'
end
```

what this does, is create a new endpoint accessible via `your_rails_app_url/my_data`. rails automatically handles converting that ruby hash `data` to json using the `render json:` instruction. this makes the data ready to be retrieved by your python script. simple, isn't it? when I first started, the magic of `render json` was completely mysterious to me, now it is the bread and butter of my day. I remember spending hours just trying to manually serialize the data. rookie mistake!

**the python side (receiving and processing the json)**

now for the python part. you'll need to use the `requests` library to fetch the json from the rails endpoint and then the built-in `json` library to parse it. if you do not have those, a simple `pip install requests` should solve the dependency issues. Here is how it looks like:

```python
# my_script.py
import requests
import json

def fetch_and_process_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        print("received json:", json.dumps(data, indent=2))
        # example processing
        if data['some_key'] == 'some_value':
            print("Processing some_key")
        for item in data['list_of_items']:
            print(f"processing item {item}")

    except requests.exceptions.RequestException as e:
        print(f"an error occurred: {e}")


if __name__ == "__main__":
    rails_url = "http://your_rails_app_url/my_data"  # change this, obviously
    fetch_and_process_json(rails_url)

```

here, we are using the `requests.get(url)` to make the http call. we also have added `response.raise_for_status()` which automatically throws an error if the http request returns anything other than 200. we are processing the returned json with the `response.json()` method and then processing it as needed in the python code.

*important:* make sure you have the `your_rails_app_url` replaced with your actual url, and the rails application has to be running for the request to work.

**using arguments instead of hardcoded url**

let us make it a little bit more flexible and allow the user to pass the url as argument to the python script. this is especially useful when working with different environments:

```python
# my_script.py
import requests
import json
import argparse

def fetch_and_process_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        print("received json:", json.dumps(data, indent=2))
        # example processing
        if data['some_key'] == 'some_value':
            print("Processing some_key")
        for item in data['list_of_items']:
            print(f"processing item {item}")

    except requests.exceptions.RequestException as e:
        print(f"an error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and process json from a url.')
    parser.add_argument('url', type=str, help='url to fetch json from.')
    args = parser.parse_args()
    fetch_and_process_json(args.url)

```

now, you can run this script using something like:

```bash
python my_script.py http://your_rails_app_url/my_data
```

this makes it much easier to move your code and test it against different instances of your rails application.

**a few words of experience and common gotchas**

*   *error handling:* notice the `try...except` blocks. i cannot stress enough how important this is. network issues, server errors, bad json – all these can throw curveballs. my early scripts were notorious for crashing at the slightest hiccup. you need to handle these exceptions gracefully, or your script will not be reliable.
*   *authentication and authorization:* sometimes the rails endpoint might be protected, and you would need to pass in some sort of authentication credentials. i will not go into the specifics of how to do that, but there is a lot of literature about auth in http requests if you ever need it.
*   *input sanitization:* while i’m not going into the topic of sanitization, be very aware of the fact that you are using data provided by a service. make sure your python script validates and filters the data it receives. i’ve had a funny incident once where a colleague sent me json that looked like an html page. after a good laugh and some minor tweaks on the server side everything worked well.
*   *environment variables*: you should probably avoid hardcoding your rails url. using environment variables makes your application configurable and more secure. i can't give you the specifics, but this can be easily achieved with environment variables.
*   *logging:* while printing to the console is very useful during development, in production, you'll want to implement proper logging to a file or a service. this will help you track down issues more easily.
*   *data serialization:* the json module in python is not the only way to exchange data between programs, you can also use message brokers like rabbitmq or redis, this is more advanced and probably not the solution you are looking for, but it is good to know.

**resources i recommend**

if you want to delve deeper into http requests, i highly recommend diving into "http: the definitive guide" by david gourley and brian totty. also for the python specific part, the official python documentation is a treasure trove, and especially for the `requests` and `json` libraries. reading their documentation can make the difference between good code and amazing code. for practical examples i think you can check "python crash course" by eric matthes. finally if you intend on working more with python check the book "fluent python" by luciano ramalho, it is the bible of intermediate python programming.

remember the goal is to keep it simple but effective, and to avoid future problems down the line. start with this simple solution and only add complexity if you really need to. that's about it. good luck.
