---
title: 'Customizable aggregation of Twitter and Reddit data'
date: '2024-11-15'
id: 'customizable-aggregation-of-twitter-and-reddit-data'
---

Hey, so you wanna pull data from Twitter and Reddit, right? That's cool, I get it, lots of info there.  The trick is making it work the way YOU want.  So, let's break it down

First, gotta get the data. Twitter's API can be a bit of a pain, but there are libraries out there to help  look up "python twitter API"  you'll find stuff like Tweepy, it's pretty easy to use.  Reddit has an API too, pretty straightforward, just search for "python reddit API" and you'll find the PRAW library  Once you have those, you can start pulling what you need.

Now,  you said "customizable," right?  That's where things get fun.  Think of it like building your own dashboard. You'll need some sort of framework to organize everything  "python data visualization" is a good starting point.  Libraries like Plotly and Matplotlib are your friends.

Here's a simple example of how to visualize Twitter sentiment using Tweepy and Plotly

```python
import tweepy
import plotly.graph_objects as go

# Your Twitter API credentials here
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Search for tweets containing a specific keyword
keyword = "python"
tweets = api.search_tweets(q=keyword, count=100)

# Analyze sentiment (you'll need a sentiment analysis library like TextBlob)
sentiments = []
for tweet in tweets:
  # analyze sentiment using TextBlob
  sentiment = TextBlob(tweet.text).sentiment.polarity
  sentiments.append(sentiment)

# Create a bar chart of sentiment distribution
fig = go.Figure(data=[go.Bar(x=["Positive", "Negative"], y=[len([s for s in sentiments if s > 0]), len([s for s in sentiments if s < 0])])])
fig.show()

```

This is just the basics  you can get real fancy with this stuff.   Add more data sources, make interactive dashboards, create custom reports,  the possibilities are endless  Good luck!
