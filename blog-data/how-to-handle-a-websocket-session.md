---
title: "How to handle a websocket session?"
date: "2024-12-14"
id: "how-to-handle-a-websocket-session"
---

alright, so you're looking at managing websocket sessions, huh? it's a pretty common situation when building real-time applications, and there are definitely some best practices i've picked up after doing this for a while. i've seen this go wrong more times than i'd like to remember, so i'm happy to share what i’ve learned.

first off, think of a websocket session as a continuous, two-way communication pipe between your server and a client. unlike http requests that are quick in and out, a websocket connection stays open for the duration of the session, allowing data to flow back and forth immediately. this means you need to treat it differently.

one of the big things to keep in mind is maintaining a stable connection. a client could disconnect at any time – internet drops, app crashes, anything really. you need to have code that can handle these situations gracefully. for me, i recall a time when i was working on a multiplayer game server, and we didn’t implement proper disconnect handling. it was chaotic. players would drop, the game state would get out of sync, and it was a debugging nightmare. we ended up adding heartbeats for the connection. we would ping the clients every 30 seconds or so and if we didn't get an answer after 3 pings, we would just drop their connection, and they would have to reconnect. it wasn't great but solved the problem.

so, what’s the techy stuff look like, practically?

let's start with the server-side code. usually, you have a kind of event loop that is always listening for new websocket connections. once one comes in, you need to do a few things. first, you need to keep track of it. i always use a simple data structure to keep track of these connections – a dictionary, a set, whatever works best for the language you are using. it’s essential to associate the open socket with some kind of user identifier or session id. here's how you might do it in python using websockets module:

```python
import asyncio
import websockets
import json

connected_clients = {}  # store active connections

async def handle_connection(websocket, path):
    user_id = str(uuid.uuid4()) #generate unique ids for each user
    connected_clients[user_id] = websocket
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                # handle message based on type or contents
                if data['type'] == 'chat':
                    await broadcast_message(data, user_id)
            except json.JSONDecodeError:
                print(f"invalid message received: {message}")

    except websockets.ConnectionClosed:
         print(f"connection closed for client: {user_id}")
    finally:
       del connected_clients[user_id]

async def broadcast_message(message, sender_id):
    for id, client in connected_clients.items():
        if id != sender_id and client.open:
            try:
                await client.send(json.dumps(message))
            except Exception as e:
               print(f"could not send to client: {id}, err:{e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

this simple server code establishes a connection, stores the socket against a unique id (using a random uuid), then it reads messages, parses the json and calls the broadcast function if the message is of type chat. the broadcast function sends the message to all connected clients (except the sender). also important to note, if the connection is closed, or the connection encounters an error the connection gets removed from the set of connections, cleaning up the resources.

on the client side it depends on the tech you are using. let’s imagine you’re in javascript on a web browser you could do something like this:

```javascript
const websocket = new WebSocket('ws://localhost:8765');

websocket.onopen = () => {
    console.log('connected to websocket server');
    // we can send initial messages if needed here
    websocket.send(JSON.stringify({type: "chat", msg: "hello server"}))
};

websocket.onmessage = (event) => {
    try{
    const message = JSON.parse(event.data)
    console.log('received message:', message);
    // display the messages in a chat interface
    } catch(e) {
      console.log("error parsing message: ", e)
    }
};

websocket.onerror = (error) => {
    console.error('websocket error:', error);
};

websocket.onclose = () => {
    console.log('disconnected from websocket server');
};

function sendMessage(message) {
  if(websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({type: "chat", msg: message}))
  } else {
     console.log("cannot send message: websocket closed")
  }
}
```

this establishes the connection, sends the initial message when it opens, then listens for incoming message events, prints them to the console and then logs if the connection closes or has errors. i’ve added a `sendmessage` function so that you can call this function and send messages as the user types for example on a chat app.

now, the real issues show when dealing with things like connection drops. your client needs to know how to try and reconnect if the connection goes down. this includes exponential back-off to prevent overwhelming your server. if it disconnects because the server crashed, and you try to immediately reconnect and then keep trying, you can bring the server down again, so we need to handle this.

here is a more robust client with reconnect logic:

```javascript
const websocketUrl = 'ws://localhost:8765';
let websocket;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 2000; // initial delay in milliseconds

function connectWebSocket() {
    websocket = new WebSocket(websocketUrl);

    websocket.onopen = () => {
        console.log('connected to websocket server');
        reconnectAttempts = 0; // Reset attempts on successful connection
        // Initial message or handshake here
        websocket.send(JSON.stringify({type: "chat", msg: "hello server"}));
    };

    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            console.log('received message:', message);
            // Handle incoming messages
        } catch (e) {
            console.error("error parsing message:", e);
        }
    };

    websocket.onerror = (error) => {
        console.error('websocket error:', error);
        handleDisconnection();
    };

    websocket.onclose = () => {
        console.log('websocket closed');
        handleDisconnection();
    };
}

function sendMessage(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({type: "chat", msg: message}));
    } else {
        console.log("cannot send message: websocket not open");
    }
}
function handleDisconnection() {
     if(reconnectAttempts < maxReconnectAttempts) {
        const delay = reconnectDelay * (2 ** reconnectAttempts); // exponential backoff
        console.log(`attempting reconnection in ${delay} ms`);
        setTimeout(connectWebSocket, delay);
        reconnectAttempts++
     } else {
        console.log("max reconnection attempts reached, giving up")
     }
}

// Initial connection attempt
connectWebSocket();
```

this more robust example includes exponential backoff logic, a function to handle disconnection events, and retries the connection until `maxReconnectAttempts` is reached. this is important if the connection drops for any reason the user won't just be disconnected, the app will try to reconnect for them.

session management also involves data management. since the connection is always on, you may have to decide how often you need to send messages. maybe you want to use some sort of change detection on the data that has changed or you might batch changes on the client side and send the updates as a bulk message to the server. for the server you need to understand what your clients are sending, what shape it is, what types of messages are there and then you may need to also batch your messages before sending them if you have a high number of concurrent users.

there's so much to consider here it really depends on what kind of app you're building. think about things like security. you might want to use wss instead of ws, which uses encryption. you also want to do some form of authentication on the websocket connection itself. consider looking into web socket authentication patterns, which might involve adding authentication headers when creating the websocket. also, there's a bunch of good resources to learn more about this stuff. i can recommend "high performance browser networking" by Ilya Grigorik, it covers a lot of this stuff in detail. also, you might like the rfc6455, which is the websocket protocol specification. these are way more in-depth, and they will tell you the "whys" instead of just the "hows".

finally, always remember to test your code. test for disconnections, test with multiple users, simulate high load, all those things. trust me, it’ll save you a lot of trouble down the line. i recall one time when we did a load test on our multiplayer game server and it collapsed under 50 concurrent users. luckily we did the test and found the problem (it was a silly bug with a race condition) and fixed it before going live.

i guess that’s everything i know about the topic, feel free to ask any further questions, unless you need help understanding why java always has too many classes, that is a mystery to even me :p.
