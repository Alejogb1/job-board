---
title: "benthos generate input from processor?"
date: "2024-12-13"
id: "benthos-generate-input-from-processor"
---

Alright so you’re asking about how Benthos takes data from processors right like how it gets the results of transformations or enrichments into its output stream yeah I've been there dude I've wrestled with that flow more times than I care to admit especially back in the day when I was setting up a really complex ETL pipeline for this old e-commerce company remember those good old days I’m talking about the late 2000s early 2010s before everything was serverless and fluffy in the cloud yeah those times We had this crazy setup with Kafka queues doing data ingestion and we needed Benthos to process that data in real time before we pushed it to the downstream data lake for the business analytics people always asking about sales figures like if I have a magic wand man

So here's the deal the way Benthos works is it's all about message processing stages think of it like an assembly line for your data each processor is a stage and these processors modify the message payloads based on their configuration The core part you're getting at is how the output of these processors gets injected back into the stream Benthos uses a concept of message parts and each processor in the pipeline acts on one or more of these message parts It can transform existing parts add new ones or even remove them But these changes are internal to the message itself I repeat they happen inside the message payload think of a message like a JSON object which can have different keys and the different keys can be modified independently

Okay let's break it down in a more practical and less theoretical manner think about a processor that transforms a JSON structure for instance suppose we have a Benthos config like this

```yaml
pipeline:
  processors:
    - jmespath:
        query: '{product_name: item.name, price: item.price}'
```

This processor is using JMESPath a query language for JSON to pull out the product name and price from an input structure that looks something like this

```json
{
    "item": {
        "name": "Awesome Widget",
        "price": 19.99,
        "category": "Electronics"
    }
}
```

The `jmespath` processor applies this query it modifies the message payload that was passed to it The important part is that the *output* of this transformation becomes the *new content* of the message part this processed message is what gets passed to the next processor or the output sink of your Benthos pipeline

Here is another example say you want to add a timestamp to each message you could use a `timestamp` processor like this

```yaml
pipeline:
  processors:
    - timestamp:
        add: true
        format: unix_nano
        target: timestamp
```
This will add a field called `timestamp` to each message with the current time in nanoseconds as an integer This is a bit simpler no query language needed just a parameter to add this new data so if you had an input like this
```json
{
    "item": {
        "name": "Another Product"
    }
}
```
Then the output of this process would be something like this depending on the exact timestamp
```json
{
    "item": {
        "name": "Another Product"
    },
    "timestamp": 1715944800000000000
}
```
Again notice how the processor modifies the internal payload of the message and outputs that to the next stage or sink

This modified payload gets sent to the next processor in your pipeline that's how processors compose the output of a previous stage is just the new input of a following stage Benthos itself is very pipeline oriented it just sends the output of a stage as input to the next stage in a sequence of processors and then to the output sink

Now let’s get a bit more advanced let's say you want to enrich messages with some data from a database for example I remember when I was working with the e-commerce guys we needed to add the customer address based on the customer ID that was available in the incoming purchase order the database lookup is where things can get a bit more complex but the mechanics of Benthos input output is the same

So for this you might use a `sql_select` processor which performs a database query and add the result to the message I’m going to skip over database connection details here we assume they are well configured somewhere else let's get straight to a SQL select statement

```yaml
pipeline:
  processors:
    - sql_select:
       query: 'SELECT address FROM customers WHERE customer_id = $customer_id'
       args_mapping: 'root.customer_id = $customer_id'
       target: customer_address
```

In this example the `sql_select` processor executes the SQL query against a database and use the `customer_id` found in the message as an argument and saves the address to the field `customer_address` the message might start like this

```json
{
    "order_id": 12345,
    "customer_id": 5678
}
```
and after the database query if the address in the database is `123 Main Street` then it becomes this

```json
{
    "order_id": 12345,
    "customer_id": 5678,
    "customer_address": "123 Main Street"
}
```
Again the processor modifies the internal content of the message by adding or modifying fields that’s what I mean by “output” becomes the “new input”

So no magic here Benthos processors don't have separate output channels The output of each processor is *the same message* after it has been modified by the processor that message flows through the pipeline The message parts are the mechanism by which Benthos preserves and passes on the data. The processors modify these message parts adding fields transforming them enriching them as it goes down the pipeline each processor takes the modified result of the processor that came before it That is the fundamental mechanism of how Benthos uses its processors for data transformation and processing.

One of my colleagues told me once "Benthos processors are like a well-oiled machine each part doing its job precisely" I told him "yeah but it can still crash if you give it garbage in" *badum tss*

You can find this detailed explanation in the Benthos documentation a resource you should be familiar with If you like reading books and understand system architecture I recommend the book "Designing Data-Intensive Applications" by Martin Kleppmann It's not specific to Benthos but it gives you a solid foundation on how these type of message processing systems work internally. Also a paper you might like is "The Log: What Every Software Engineer Should Know About Real-Time Data's Unifying Abstraction" by Jay Kreps which helps you understand how message queues and processing work together. I personally found it insightful when thinking about these systems. Hope this helps.
