---
title: "How to implement a Poll in a group chat?"
date: "2024-12-15"
id: "how-to-implement-a-poll-in-a-group-chat"
---

alright, so you're looking at building a poll feature within a group chat, right? i've been there, done that, got the t-shirt, and probably a few scars to show for it. i remember back in '08, i was working on a little side project—a sort of proto-discord for a gaming community i was part of. we really needed a way to decide on raid times, and well, that's when i learned about the joys and pains of real-time poll implementations. it sounds straightforward, but trust me, the devil is in the details.

the basic idea isn't that complex, you're talking about capturing user input (their votes) and displaying the results in a way that's both informative and updates live, preferably. we’re not building a general purpose survey engine here, just something simple for a chat group, so it should be streamlined for that specific use case, focusing on simplicity of interaction.

let’s break this down into the core components you're going to need to think about. first, you'll need a data structure to represent the poll itself. this would include the question being asked, the possible answers people can vote for, and obviously a mechanism to keep track of how many votes each option receives. you'll also probably want to know who voted for what, but if you’re aiming for anonymity you might need a more complex system, or perhaps you will just not care who voted what.

here's a basic python class i’d start with, just to visualize:

```python
class poll:
    def __init__(self, question, options):
        self.question = question
        self.options = options
        self.votes = {option: 0 for option in options}
        self.voters = {} # user_id -> option
    
    def vote(self, user_id, option):
      if user_id in self.voters:
        old_option = self.voters[user_id]
        self.votes[old_option] -= 1
      self.voters[user_id] = option
      self.votes[option] += 1
    
    def get_results(self):
      return self.votes
```

now that gives you a simple way to structure the poll. but the real challenge, as usual, lies with the real time aspect. if we were doing this over http, it is not very hard to implement using server side events. we would need server side events to push updates. but let's assume that we have a message-oriented middleware system such as redis.

first, every time someone votes, you need to update this representation and then, and this is crucial, you need to broadcast this change to all users currently participating in the chat. this means that you can't just have some simple polling mechanism as it will cause too many round trips. you need a server side publish-subscribe system. we are going to use redis for that. in that case, a python solution to process the vote can look like this. the example below will require a redis server up and running. you can install redis using apt install redis-server on linux.

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0) # assume redis is running on localhost

def process_vote_redis(poll_id, user_id, option):
    poll_key = f"poll:{poll_id}"
    poll_data = redis_client.get(poll_key)

    if poll_data:
      poll_data = json.loads(poll_data)
    else:
      return False # handle non existing polls

    voters = poll_data.get('voters',{})
    if user_id in voters:
      old_option = voters[user_id]
      poll_data['votes'][old_option] -= 1

    voters[user_id] = option
    poll_data['votes'][option] += 1
    poll_data['voters'] = voters

    redis_client.set(poll_key, json.dumps(poll_data))
    
    # Publish the update
    redis_client.publish(f"poll_updates:{poll_id}", json.dumps(poll_data))

    return True
```

in that case, on the client side, you would have a process constantly listening on the redis channel `poll_updates:{poll_id}` for the update. if you happen to have a javascript frontend, it can be as simple as the following using redis pub/sub client, but other languages have similar clients.

```javascript
import { createClient } from 'redis';

const redisClient = createClient();

redisClient.connect();
async function subscribeToPoll(poll_id, callback) {
  const subscriber = redisClient.duplicate();
  await subscriber.connect();

  await subscriber.subscribe(`poll_updates:${poll_id}`, (message) => {
      try {
          const update = JSON.parse(message);
          callback(update); // Call the provided callback
      } catch (e) {
        console.error("error while parsing the json ",e);
      }
  });
  return () => {
    subscriber.unsubscribe(`poll_updates:${poll_id}`);
    subscriber.disconnect();
    console.log("unsubscribed from poll", poll_id);
  }
}

// usage:
const unsub = subscribeToPoll(123,(update)=>{
  console.log("poll update received",update);
});

//to unsubscribe

unsub();
```

now, in terms of displaying the results to the user, you have several options. you could just represent it as text, something like "option A: 5 votes, option B: 2 votes". or, and i think this is a better way, you can use a visual component like a horizontal bar chart where the length of each bar indicates the percentage of votes it has, that is going to be more visual for sure. that depends on the platform you're building on, really. if we were on the web, i'd suggest canvas or maybe a library that builds such charts, but we're not talking about any framework specifically here.

one very important consideration in a chat environment is also how to handle the "i changed my vote" scenario. you have several ways to approach this. you could let them change their vote at will, with the latest vote overriding the previous one. that’s what my example code does. or, you might not allow people to change their minds once they voted, for simplicity. it depends really on the type of poll you are building, and the requirements of the chat environment.

another tricky bit is that you will probably need a way to handle concurrent votes. if two people vote for the same option at essentially the same time you can't let your backend code step on itself. in our code example we delegate the concurrency management to redis.

as for scaling this, the data structure will probably be fine for most group chats, but you might want to cache the results on the client side, so the user doesn’t have to do too much back and forth. you might need different redis channels, one per poll or one per chat, or even to keep the polls in memory when the user is in the chat window. it is all about trade-offs. if you end up having thousands of concurrent users you will need a more sophisticated caching mechanism. that is a topic on its own.

also, you’ll want to think about how to make the chat application resilient to network partitions and failures in this pub/sub system. having multiple redis instances, or using a message queue that is more fault tolerant could be required. this is more advanced.

now, for resources, i'd recommend looking into some of the classic papers on distributed systems like "time, clocks, and the ordering of events in a distributed system" by lamport. also books like "designing data-intensive applications" are always good to have on your bookshelf. these will give you a more deeper understanding of the theory behind these systems. also for redis pub/sub implementation the official redis documentation provides some very good examples, and a very simple and clear explanation of the available commands.

implementing a poll feature in a chat is, well, not rocket science… but i've seen enough projects where a simple feature like this becomes a source of pain, so proper planning and an understanding of the data flow and the concurrency aspects is essential. i guess this is as simple as i can get to explain it.

one last thing, they said that the first programmer was a woman, that is the reason why all errors are called "her". but don't tell anyone that. it is a secret.
