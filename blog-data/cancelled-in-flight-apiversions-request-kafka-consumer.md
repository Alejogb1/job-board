---
title: "cancelled in-flight api_versions request kafka consumer?"
date: "2024-12-13"
id: "cancelled-in-flight-apiversions-request-kafka-consumer"
---

Alright so you're hitting that lovely cancelled in-flight api_versions request with your Kafka consumer right Been there done that got the t-shirt probably several actually. Its always fun when things decide to go sideways mid-flight. Let me break down what this probably means and how you can actually troubleshoot it and get yourself un-stuck. 

From my past experience this usually means that your Kafka consumer is trying to talk to a Kafka broker but somewhere along the way the connection got severed or the broker is just not responding. It's kind of like yelling into a void and then getting angry cause nobody is answering you. Its frustrating I know. Now the "api_versions" bit this is crucial that's the initial handshake your consumer does when it first connects. It asks the broker what Kafka versions it supports and what features it can use. If that handshake gets canceled then your consumer's never going to be happy. Its probably gonna start throwing errors and stop doing useful work.

Usually what I see is that its not a consumer code problem its more often than not an infra or network type thing like network hiccups timeouts broker issues. Its easy to get tunnel vision when debugging and think you messed up the code but the problem is often in the plumbing. 

So lets get practical. First thing you want to do is check your Kafka consumer configs specifically the `connections.max.idle.ms` this is the time in milliseconds how long the connection can be idle before the client closes it. If that's too low then your consumer might be dropping connections unexpectedly. It's like you are sending a request and you are too impatient to wait a little bit and you hang up while its being processed. Something very much like that. 

```python
from kafka import KafkaConsumer
consumer = KafkaConsumer(
    'your_topic',
    bootstrap_servers=['your_kafka_broker:9092'],
    group_id='your_consumer_group',
    # other configs go here
    #...
    connections_max_idle_ms = 300000 # Set to 5 minutes
)
```

I normally set this to at least 5 minutes 300000 milliseconds it gives the connection more breathing room.  Now don't just blindly copy that and call it a day you need to test and see what works best for your setup. Also check your `request.timeout.ms` config this determines how long the consumer waits for a response from the broker before timing out.  

```python
from kafka import KafkaConsumer
consumer = KafkaConsumer(
    'your_topic',
    bootstrap_servers=['your_kafka_broker:9092'],
    group_id='your_consumer_group',
    # other configs go here
    #...
    request_timeout_ms = 10000 # 10 seconds is a good starting point
)

```

The default is usually pretty low sometimes 30 seconds or something you might need to increase it if you have a busy broker or network with some latency. Start with 10 seconds is a good start. 

Also important make sure your brokers are alive and kicking. I once spent a whole day debugging a consumer issue only to find out one of the brokers was just unresponsive. Its important to check if your cluster is happy before diving into code level debugging. Use tools like `kafka-topics.sh` or `kafka-consumer-groups.sh` to check the health of your cluster and consumer groups.

Here is the script I personally used that I made from one time when I had to deal with this same error. 

```python
from kafka import KafkaConsumer
from kafka.errors import KafkaError

def consume_messages(topic, brokers, group_id):
  consumer = KafkaConsumer(
    topic,
    bootstrap_servers=brokers,
    group_id=group_id,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    connections_max_idle_ms = 300000,
    request_timeout_ms= 10000
    # Set the values here
  )
  try:
    for message in consumer:
      print(f"Received message: {message.value}")
      
  except KafkaError as e:
    print(f"Error consuming messages: {e}")
  finally:
    consumer.close()


if __name__ == "__main__":
  topic = "your_topic" # Change the value
  brokers = ["your_kafka_broker:9092"] # Change the value
  group_id = "your_consumer_group" # Change the value
  consume_messages(topic, brokers, group_id)
```
This script that I use catches errors specific to Kafka so you dont need to be surprised all the time. I added the request timeouts and idle timeouts on the consumer configs. Use it and modify it as needed. 

Moving on to other things to check. Check if there are any firewalls or network issues that could be blocking the connection between your consumer and the Kafka brokers. This also includes load balancers or proxies if any. Sometimes a proxy gets overloaded and stops processing requests. Double-check if the DNS resolution is working properly. You might have the broker address written right in your config file but if the DNS server cant resolve it well you are out of luck. Its like having the right phone number but the phone company is down.

Also a subtle thing I learned the hard way is the Kafka client library version. Make sure that your client library is compatible with your Kafka broker version.  Older clients sometimes have issues when they try to connect with newer brokers and vice versa. The API changed a little in Kafka over the years and using the right version matters. It's generally a good practice to keep both the broker and client library up-to-date. When in doubt check the documentation of your client library. 

Another thing is check your broker logs. They usually contain important details about the issue if the issue is server side. Look for warnings or errors that correlate with the time you are having problems with your consumer. Usually you find a clue there. 

Now about the cancelled request itself the root cause could be a bunch of things. It could be the broker forcibly closing the connection because of some issue it detects. It could also be that the connection is idle and it has been closed by a firewall or some intermediate device or something like that. Its like the request sent but the other side does not receive it so they just abandon it.

Oh and before I forget always remember proper error handling. Make sure you are catching `KafkaError` exceptions so that your consumer doesn't just silently crash. It will save you hours of debugging. I made a mistake once and I saw that all the service was down but I did not know exactly where it was failing and I had a really bad time that day. 

One last thing. I once made a typo in the consumer group name and spent two hours wondering why my consumer wasn't receiving messages. Its the smallest typo on the configs that can break your whole day. Yes we all have been there. It's like trying to login to your account with a wrong password. You know its something simple but you need to check every detail and you are not sure what is it.

And to add something funny (I know you dont want jokes but this is a special case) a client version and broker version walk into a bar and the broker says sorry I can not support you here the API is not compatible... Its like they spoke different languages. I hope that helps with the pain this can generate sometimes

Now here are a few resources that I would recommend. Don't bother with random blog posts or youtube videos for this type of error. Stick with the official documentation and some hardcore books.

*   **"Kafka: The Definitive Guide" by Neha Narkhede et al:** This book is like the bible for Kafka it covers the internals and configurations in great details and it is a must have in any kafka user's bookshelf. It will have all the answers to the questions you have.
*   **The Official Apache Kafka Documentation:** The official docs is a primary source of all the features functionalities and configurations. You need to bookmark this website and visit it when in doubt. This the place to go to first when debugging.
*   **Confluent documentation:** Confluent is the company that makes Kafka and their docs are goldmines. They explain a lot about how Kafka is used in practice and they provide very good guidance. Look for sections about client configurations troubleshooting guides and best practices.

So to sum it up check your consumer configs check your broker health network problems error handling and client library versions. Start from the bottom up basic things and it will often lead to a solution. Happy coding and let me know if you have other questions.
