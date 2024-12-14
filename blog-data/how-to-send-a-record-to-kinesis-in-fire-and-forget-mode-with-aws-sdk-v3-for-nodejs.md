---
title: "How to send a record to Kinesis in fire-and-forget mode with AWS SDK v3 for NodeJS?"
date: "2024-12-14"
id: "how-to-send-a-record-to-kinesis-in-fire-and-forget-mode-with-aws-sdk-v3-for-nodejs"
---

alright, so you're looking to fire data into kinesis using the aws sdk v3 for nodejs without waiting for a response, basically a fire-and-forget approach. i've been there, done that, got the t-shirt, and several support tickets to prove it. trust me, i've seen kinesis streams behave in more ways than i care to remember.

first off, let's break down what 'fire-and-forget' actually means in this context. it implies that after sending a record to kinesis, your application won't wait for a confirmation that the record was successfully processed. you basically send the record and move on. the benefit is that you can achieve higher throughput as your app isn't blocked waiting for i/o. the trade-off is the lack of immediate feedback, which means you need to have other mechanisms for monitoring and error handling if necessary.

my personal history with this goes back to a project where we were processing massive amounts of clickstream data. we had a bunch of servers generating these events, and having each of them wait for a response from kinesis became a bottleneck. we were getting hammered. moving to a fire-and-forget method increased the overall throughput substantially. at that time (around 2018) it was aws sdk v2, i had to do some crazy stuff with promises. but i am glad we have moved on.

now, let's get to the code. the aws sdk v3 for nodejs changed quite a bit how things work, so it's good to be up to date.

here is the basic structure, how to set up the kinesis client and the way to perform the action.

```javascript
const { KinesisClient, PutRecordCommand } = require("@aws-sdk/client-kinesis");

// configure the kinesis client
const kinesisClient = new KinesisClient({
  region: "your-aws-region", // replace with your actual region
  // you can also specify credentials here or let it be managed
  // by environment variables or aws configuration
});

// function to send a record in a fire-and-forget way
async function sendToKinesisFireAndForget(streamName, partitionKey, data) {
    const params = {
        Data: Buffer.from(data), // kinesis wants a buffer not plain text
        PartitionKey: partitionKey,
        StreamName: streamName,
    };

    const command = new PutRecordCommand(params);

    // send the command without awaiting the response
    kinesisClient.send(command).catch(error => {
        console.error("error sending to kinesis, it is a fire-and-forget so beware:", error);
        // here you can implement logging, or implement error mechanism
        // the important is you do not block
        // for example:  send this error to an sqs or other system.
    });
    // we are not awaiting the response, the client will send and resolve the promise
    // in the background, no blocking our application thread

    // you might think you can return something like 'sent' but actually this is
    // misleading because the command is not even resolved yet. if you need to
    // confirm then do not use fire-and-forget method
}

// example usage
const streamName = "your-kinesis-stream-name"; // replace with your kinesis stream name
const partitionKey = "my-partition-key-1"; // replace with a partition key, it is important for data sharding
const data = "my sample data"; // this is your actual data

sendToKinesisFireAndForget(streamName, partitionKey, data);

console.log("sent record in fire and forget mode");
// your application can continue working now.
```

notice i'm not using `await` here, and instead i'm catching the error directly on the promise. this is important for the fire-and-forget behavior. if you use `await` you are actually waiting for a response which is not the idea at all. if you are trying to debug why the message is not sent or you are unsure, i would advise you to switch to an async/await call and debug that first, then once you have a grasp of the issue go back to fire-and-forget. debugging is not as straight forward and you might end up wasting time trying to find out why your message is not being sent while not using the right debugging method.

now, let's talk about batching. while sending individual records works, it's usually not the most efficient way to utilize kinesis. sending multiple records in a single request is often better for performance (and also cheaper). here’s how you can do batching using `putrecordscommand`.

```javascript
const { KinesisClient, PutRecordsCommand } = require("@aws-sdk/client-kinesis");

// configure the kinesis client
const kinesisClient = new KinesisClient({
  region: "your-aws-region", // replace with your actual region
});

async function sendBatchToKinesisFireAndForget(streamName, records) {
    const params = {
        Records: records.map(record => ({
          Data: Buffer.from(record.data),
          PartitionKey: record.partitionKey
        })),
        StreamName: streamName,
    };

    const command = new PutRecordsCommand(params);

    // send the command without awaiting the response
    kinesisClient.send(command).catch(error => {
        console.error("error sending batch to kinesis, it is a fire-and-forget so beware:", error);
        // here you can implement logging, or implement error mechanism
        // the important is you do not block
    });

    // same as above you can not return any information, because the action is
    // happening in the background
}


// example usage
const streamName = "your-kinesis-stream-name"; // replace with your kinesis stream name
const records = [
  { partitionKey: "my-partition-key-1", data: "data 1" },
  { partitionKey: "my-partition-key-2", data: "data 2" },
  { partitionKey: "my-partition-key-1", data: "data 3" }
];

sendBatchToKinesisFireAndForget(streamName, records);

console.log("sent batch record in fire and forget mode");

```

now, remember that with batching, you need to keep in mind the kinesis limits regarding the size of the records and the batch size. if you exceed those limits, the send operation will fail, or you might receive a throttling exception. the error handling in the catch block will catch the error, but the fire-and-forget approach will not allow you to resend those messages. you need to implement proper mechanisms for this. think of implementing an exponential backoff retry strategy. or moving the messages to a dead letter queue.

one more point to consider, is that even though it's fire-and-forget, it's essential to have some kind of monitoring in place. kinesis has built-in cloudwatch metrics which are quite useful. you want to check things like `putrecordssuccess` or `putrecordsfailed` metrics. also setting up alarms is a good idea.  if you are not monitoring, you could be losing data without knowing it. believe me i have had situations like that, i was losing important data for several hours, and it went unnoticed because the monitoring was not configured.

and this is where my experience tells me to be extra careful. you do not want to lose the messages, so in a fire-and-forget scenario you need to implement proper error handling.

i had one instance where i forgot to implement proper error handling, i just logged the errors and forgot about it. i was not able to know what data was lost and when. it was a nightmare. it is crucial to implement those mechanisms properly.

also, something that often catches folks out is the difference between the `putrecord` and `putrecords` operations. use `putrecords` when you're sending multiple records together for batch processing, `putrecord` is for single records. they are subtly different and it took me some time to get used to that.

here's one more code example, this time with a small retry mechanism. it’s not a full-fledged solution, but it shows how to add some retry logic, to not loose your data, even when doing a fire-and-forget. just bear in mind that this may delay the fire-and-forget behavior, it's up to you to find the perfect balance between fire-and-forget and reliability.

```javascript
const { KinesisClient, PutRecordCommand } = require("@aws-sdk/client-kinesis");

// configure the kinesis client
const kinesisClient = new KinesisClient({
  region: "your-aws-region", // replace with your actual region
});

const maxRetries = 3; // maximum number of retries
const baseDelay = 100; // initial delay in milliseconds

async function sendToKinesisFireAndForgetWithRetry(streamName, partitionKey, data, retryCount = 0) {
  const params = {
      Data: Buffer.from(data),
      PartitionKey: partitionKey,
      StreamName: streamName,
  };

  const command = new PutRecordCommand(params);

  kinesisClient.send(command)
      .catch(async error => {
          console.error("error sending to kinesis, it is a fire-and-forget but we will retry:", error);

          if (retryCount < maxRetries) {
              const delay = baseDelay * Math.pow(2, retryCount); // exponential backoff
              console.log(`retrying in ${delay} ms, attempt: ${retryCount + 1}`);
              await new Promise(resolve => setTimeout(resolve, delay));
              return sendToKinesisFireAndForgetWithRetry(streamName, partitionKey, data, retryCount + 1);
          } else {
            console.error("max retries reached for kinesis send, logging error:");
            // implementation for dead letter queue
            // here you can implement logging, or implement error mechanism
            // the important is you do not block
          }
      });
}

// example usage
const streamName = "your-kinesis-stream-name";
const partitionKey = "my-partition-key-1";
const data = "my sample data";

sendToKinesisFireAndForgetWithRetry(streamName, partitionKey, data);
console.log("sent record in fire and forget mode with retry mechanism");

```

now about useful resources. forget about googling too much. go to the source. the official aws documentation for kinesis is very comprehensive, and the aws sdk v3 documentation is getting better each day. also, the "aws certified developer official study guide" book, is a good resource with examples and practical knowledge that can save you a lot of headaches. if you want more in depth theory, i suggest looking at the book "designing data intensive applications", it's a classic and goes in depth about the data streaming theory.

remember, the fire-and-forget approach is great for high-throughput use cases, but you need to balance that with the reliability of your system. think about what it means to not receive confirmation and how that can impact your application. it is not a silver bullet and it should be used wisely. finally, if you are sending so many messages that you start to get throttled, then it is time to re-think your whole kinesis architecture, and probably you need more shards. just to be clear, the number of shards is an important setting that has a direct impact on the throughput, if you have too few, then you are going to get throttled. so, this is another thing that you should take into consideration.

i hope that helps, let me know if you have more questions. and remember, sometimes the error is not on aws side, it is in front of the screen.
