---
title: "How does LinkedIn's design of Sales Navigator URLs enforce exclusivity and user-specific access? (Teaching point: Discusses design strategies for premium tools.)"
date: "2024-12-12"
id: "how-does-linkedins-design-of-sales-navigator-urls-enforce-exclusivity-and-user-specific-access-teaching-point-discusses-design-strategies-for-premium-tools"
---

Second: What are the core algorithmic techniques behind the recommendations in YouTube's "Up Next" queue? (Teaching point: Explore recommendation system techniques.)
Third: Consider the architectural decisions that might underpin Slack's real-time messaging functionality across various platforms. (Teaching point: Examines real-time system design.)

lets start with sales nav urls. ok so the urls in linkedin sales navigator they are not like regular linkedin profiles right its not just linkedin.com/in/yourname. theres definitely some added complexity and the reason behind that is mostly about controlling access. consider it like this think of the base profile url like a public key that everyone can see but sales navigator urls are like a key to a specific room a more specialized location. each url usually has a bunch of extra parameters encoded in it like a series of encoded values these aren't just random characters they're carrying information like the user id who is accessing it the sales navigator subscription level and even sometimes specific filters they have applied. so when you share a sales nav url you're actually sharing more than just a profile you're sharing your personal view of that profile and the associated access rights granted by your subscription.

the encoding part this is where a lot of the magic happens its not always readily understandable by just looking at the url those seemingly random strings actually represent data. they are often using algorithms for data encoding and url manipulation this process ensures not only that data is included but also that its not easily tampered with. if you're interested in this kind of stuff you should check out some resources on url encoding and data serialization its like how you convert structured data into a string that can be safely transmitted over a network there are many tutorials online and even some books like "understanding url encoding" that can give you a good starting point. so yeah the urls are basically security tokens in disguise each one unique to that user's access level and view

now onto youtube up next. this is all about recommendation systems. it's not a single algorithm at play it's usually a layered approach. the main thing here is that youtube has to predict what you might want to watch next with all the choices in front of us it must have some data crunching going on. at the most fundamental level youtube likely relies on collaborative filtering this means that it looks at what users similar to you have watched and recommends those videos. but that alone isnt enough because not all users are the same youtube probably combines this with content based filtering where it analyses the video itself its metadata tags descriptions etc and then suggests videos that are similar in terms of topic and content.

then theres also deep learning coming in picture neural networks probably play a major role to create embeddings of videos and users an embedding is a representation of complex data as a vector of numbers where similar items have similar vector representations. this helps youtube identify relationships beyond simple tags or descriptions it can grasp underlying concepts. ranking is a whole other beast its not just about predicting whats relevant its also about ranking what to show first some of the considerations here include watch time clicks engagement like comments shares and also negative signals such as disliking videos or clicking "not interested" this creates a feedback loop. and theres also temporal dynamics for example if you just watched a 10 minute video about coding you might be interested in more similar videos immediately but maybe not an hour from now. all this is like a constantly updating model that adapts to your behavior.

here's some pseudo code to illustrate collaborative filtering

```python
def collaborative_filtering(user_id, user_video_matrix, similar_user_count=5):
  # find users similar to user_id based on videos watched
  similar_users = find_similar_users(user_id, user_video_matrix, similar_user_count)
  recommended_videos = {}

  for similar_user in similar_users:
    for video_id in user_video_matrix[similar_user]:
      if video_id not in user_video_matrix[user_id]:
        if video_id not in recommended_videos:
          recommended_videos[video_id] = 0
        recommended_videos[video_id] += 1
  # sort by count
  sorted_videos = sorted(recommended_videos.items(), key=lambda item: item[1], reverse=True)
  return [item[0] for item in sorted_videos]

#simplified function for finding similar users based on video watching history.
def find_similar_users(user_id, user_video_matrix, similar_user_count):
    #for simplicity we are checking if two user watched a particular video
    similarity_scores = {}
    for other_user in user_video_matrix:
        if other_user == user_id:
            continue
        score=0
        for video in user_video_matrix[user_id]:
            if video in user_video_matrix[other_user]:
                score=score+1
        similarity_scores[other_user]=score
    sorted_users = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    return [item[0] for item in sorted_users[:similar_user_count]]

```

if you're interested in the details i'd suggest diving into papers and books on recommendation systems like the ones from research conferences on information retrieval and data mining. books such as "recommender systems handbook" can give a deeper dive into the world of content and collaborative filtering algorithms.

lastly slack. real time messaging. the architecture is probably something like a distributed system with multiple layers. the core of slack is messaging and the core of messaging is real time delivery and ensuring that all clients mobile web desktop see messages pretty much at the same time. this definitely cant be a single server somewhere it's going to be a system of many servers spread across different geographic locations to reduce latency.

when a user sends a message the following things might be happening. first the message is sent to a load balancer which distributes the message to a web server which validates the user and creates an id for the message. then the message gets broadcast to all connected clients for the channel the message was sent to. for this a pub sub system is going to be a central part of the architecture think of it as a messaging bus that allows different parts of the slack architecture to communicate with each other. the message is first pushed to the pub/sub system and then from this pub sub the connected clients are getting pushed the message to display. to manage the different channels each message must have its channel id to make sure only relevant people are getting the message. theres likely also a persistent storage system for messages usually a database for example postgres or similar that lets slack retrieve previous messages.

there is also some client side logic like handling disconnections and reconnections. if your device is experiencing a poor connection it will have to automatically attempt to reconnect to the slack system and then ensure no messages are lost. offline handling could be possible using some local database on the client itself to save the recent messages until the client is reconnected. a lot of this depends on websocket connections which provide a persistent bi-directional connection between a client and a server. to ensure scaling the system can be designed in a microservice oriented approach where each different functionality like user management message delivery and channel management are all independent units.

here's a conceptual example of how you might handle the pub/sub aspect using python and a simple dict for the message queue

```python
class MessageBroker:
    def __init__(self):
        self.subscriptions = {} #channel:list of client
        self.message_queue = {} #channel:list of messages
    def subscribe(self, client_id, channel_id):
        if channel_id not in self.subscriptions:
            self.subscriptions[channel_id]=[]
        self.subscriptions[channel_id].append(client_id)
        if channel_id not in self.message_queue:
            self.message_queue[channel_id]=[]
    def unsubscribe(self,client_id, channel_id):
        if channel_id in self.subscriptions and client_id in self.subscriptions[channel_id]:
            self.subscriptions[channel_id].remove(client_id)
    def publish(self, channel_id, message):
         if channel_id not in self.message_queue:
             self.message_queue[channel_id]=[]
         self.message_queue[channel_id].append(message)
         if channel_id in self.subscriptions:
             for client in self.subscriptions[channel_id]:
                 print(f'sending message "{message}" to {client}')
message_broker = MessageBroker()
message_broker.subscribe("client_a", "general")
message_broker.subscribe("client_b", "general")
message_broker.subscribe("client_c", "random")
message_broker.publish("general","hello")
message_broker.publish("random", "random stuff")

```
this is a heavily simplified version. production systems can use dedicated message queue like rabbitmq or kafka

for further reading on this i'd recommend focusing on papers related to distributed systems real time data processing and message queueing. books on designing data intensive applications are also useful like "designing data intensive applications" by martin kleppmann it will help understand these distributed architectures in detail.
```python
#pseudo code example client side
def process_incoming_message(message, current_user, channel_id):
    if message.channel_id == channel_id:
        if message.sender_id == current_user:
            display_message_on_screen(message,"own")
        else:
            display_message_on_screen(message,"other")
def display_message_on_screen(message,type):
    print(f'displaying {message} and type {type}')
```
again very simplified example real world is more complex. these are just some highlights of how these systems might work
