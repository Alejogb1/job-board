---
title: "How does LinkedIn's session-based architecture enhance data privacy and security? (Teaching point: Discusses architectural decisions in web application security.)"
date: "2024-12-12"
id: "how-does-linkedins-session-based-architecture-enhance-data-privacy-and-security-teaching-point-discusses-architectural-decisions-in-web-application-security"
---

My second suggestion is: Can you elaborate on the role of web sockets in real-time updates in platforms like Slack and Discord, and how it reduces the need for constant polling? (Teaching point: Explores real-time communication and efficiency in web applications.)
My third suggestion is: How might microservices contribute to the scalability of a system like Netflix, especially concerning the distribution of content across different regions and devices? (Teaching point: Examines the benefits of microservices architecture for large-scale applications.)

okay lets break this down you know like a deep dive into how these big platforms do their thing

first up linkedin and their session based stuff specifically how it helps with privacy and security think of it like this when you log into linkedin you're not just walking in with a blank face the system gives you a temporary key a session id stored usually in a cookie or a similar mechanism that's like your VIP pass for the site the server checks this id with each request you make its not storing your password or anything sensitive directly in your browser its like saying im user 12345 with the right access rights and the server says okay that checks out

the cool part is this session id is usually a random string of characters and numbers practically impossible to guess and its only valid for a limited time think of it like a day or a few hours after which it expires forcing you to log in again that way if somehow someone did get hold of your session id they wouldn't be able to do much because it wouldn't be valid for long it reduces the window of opportunity for bad actors its also server side session data means it's not visible on the client except as the id. This approach is way more secure than storing say your username or password directly because if that gets compromised it's game over session ids are also server side and the actual credentials are never directly exposed to the client reducing attack surfaces it's also often paired with other mechanisms like https which encrypts the data in transit between your browser and the server so even if someone did try to snoop they would just see garbled text.

its about limiting exposure limiting the blast radius if something goes wrong also think about the concept of stateless servers session handling is managed on the server side it doesn't need to remember who you are between each request it only cares about the current session id and that separation simplifies maintenance and scalability imagine that each request doesnt need to track a whole user history its all linked through a single session id. a server can get its id in the request and can be scaled horizontally for instance. you know stuff like load balancing is also easier as a stateless operation

next lets tackle web sockets and real time updates like in slack or discord imagine the old way without web sockets you want to know if a new message was posted you'd have to constantly ask the server hey anything new hey anything new thats polling its inefficient the browser makes numerous requests to the server that are mostly useless.
web sockets are different they're like an open line of communication between your browser and the server once the connection is established the server can push new data directly to your browser without you having to ask and the browser can send messages back to the server when you send a message in a chat its a continuous data stream not just a request response cycle this dramatically reduces the amount of network traffic and server load its just like having a persistent connection that never closes until you close it yourself or something breaks.

the persistent connection aspect is the game changer in terms of efficiency and real time its a duplex channel so data can flow both ways concurrently using this the server can push data and updates quickly without having to wait for clients to request it its like a radio transmitter broadcasting all the time and you only listen when something changes this is what enables those instant real time chats you see in slack or discord you see a message appear immediately rather than waiting for a reload or manual refresh

think about it its like tcp but persistent whereas http is a request response approach web sockets are a different protocol they use tcp and it's persistent with http its request response its very inefficient if you need constant updates and its perfect for streaming data.

```javascript
// example of client side web socket initialization
const socket = new WebSocket('wss://your-websocket-server.com');

socket.onopen = () => {
  console.log('WebSocket connection established');
  socket.send('Hello Server!'); //send a message to the server
};

socket.onmessage = (event) => {
  console.log('Message received from server:', event.data);
 // process the message
};

socket.onclose = () => {
  console.log('WebSocket connection closed');
};

socket.onerror = (error) => {
  console.error('WebSocket error:', error);
};

```

the above is an example of very high level javascript code on a browser connecting to a websocket url then it has callbacks for various events like opening receiving messages closing or an error.

finally lets get into microservices and netflix scaling imagine netflix as a giant monolith one huge application all bundled together it gets incredibly difficult to manage and update. this is where microservices architecture comes in think of it like breaking down that monolith into smaller independent services each service handles a specific piece of functionality like user authentication video encoding recommendations or billing now all of these are separate and independent each service can be scaled and updated without affecting the other services each service uses its own database and its own code

with netflix imagine video streaming is separated from user management or search engine each one can be scaled to meet its specific need user management or search engine each can be scaled up and down independently its about distributing the load across different servers so no single server is overloaded this lets netflix deliver videos to billions of users across the world and the ability to quickly deploy changes to just one part of the app without impacting the whole service. if something breaks in video encoding it doesnt take down the entire platform.

also think about deployments with microservices netflix can push code changes to video encoding while other services like recommendations or user authentication are still running smoothly without any downtime.

```python
# example of microservice interaction using a REST api
import requests

user_id = 123
video_id = 456
# call to a user management service
user_response = requests.get(f'https://user-service/users/{user_id}')
if user_response.status_code == 200:
    user_data = user_response.json()
    print(f"user data: {user_data}")
else:
    print(f"could not get user data: {user_response.status_code}")

# call to video encoding service to get encoding status
video_response = requests.get(f'https://video-encoding-service/videos/{video_id}/status')
if video_response.status_code == 200:
   video_status = video_response.json()
   print(f"video status: {video_status}")
else:
   print(f"could not get video status: {video_response.status_code}")

```

the above is a example of python code making calls to two different backend microservices getting user data and video data using http restful apis. each microservice has its own codebase and scales independently.

the following is a basic example of a go based microservice showing the setup

```go
package main

import (
        "fmt"
        "log"
        "net/http"
)

func videoHandler(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintln(w, "video status okay")
}
func main() {
        http.HandleFunc("/videos/status", videoHandler)
        log.Println("Starting server on :8080")
        if err := http.ListenAndServe(":8080", nil); err != nil {
                log.Fatal(err)
        }
}
```

this simple go microservice exposes a endpoint that displays a status.

microservices are about independent teams working on small pieces and scaling and deployment is easier its all about breaking things down into manageable chunks and the core concept is loose coupling and high cohesion think of each service focusing on doing one thing well and if one service fails it does not bring the whole system down this helps with scalability performance availability and maintainability of the entire platform.

for resources you might find "Designing Data-Intensive Applications" by Martin Kleppmann a great resource for overall system design and architecture concepts also “Building Microservices” by Sam Newman is a solid book focused on microservices and architectural patterns and for security a good starting point is to look at owasp website which has resources on common vulnerabilities and secure web app development. these resources should help to dive into the details further beyond this high-level overview.
