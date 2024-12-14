---
title: "How to do Grails push notification from one user to another?"
date: "2024-12-14"
id: "how-to-do-grails-push-notification-from-one-user-to-another"
---

alright, so you're looking at setting up push notifications in grails, going from one user to another. this is a pretty common ask when you’re building anything that needs real-time interaction, like a chat application or a social media feed. i've been down this road a few times, and it can get a bit intricate, but let’s break it down.

first things first, you're not going to be pushing notifications directly from your grails server to a user’s device. that's not how push notifications work. the standard approach is to use a push notification service – think firebase cloud messaging (fcm) or apple push notification service (apns) – they handle the heavy lifting of getting messages to the right devices.

here’s the general flow:

1.  **user a does something that triggers a notification.** maybe they post a comment, or send a message. your grails app is going to notice this event.
2.  **grails app decides user b needs a notification.** based on your application logic, you figure out who needs to get alerted.
3.  **grails app sends a message to fcm/apns.** instead of directly pushing to the user's device, you send a formatted payload to fcm or apns. this payload contains the message and the unique device token of user b (more on that later).
4.  **fcm/apns push the notification to the user b's device.** the push service takes care of routing the notification to the correct mobile operating system (ios or android) which in turn pushes the notification to the device.
5.  **user b sees the notification.** the user can interact with it, which could take them back into your application.

so, let's talk about the grails side.

the first key part is registering devices to users. when a user logs into your application from their mobile device you're gonna have to save a unique token associated with that user and device. this token is provided by the os when your app is installed on the device, and is usually done on the mobile client using fcm sdk (for android) or apns sdk (for ios). you’ll need to store this token in your database, associated with the user. it's a basic table setup really, let's call it `device` or whatever you like.

here’s a sample gorm domain class to get started:

```groovy
class Device {
    String token
    User user
    String platform // "ios" or "android"

    static constraints = {
        token unique: true, blank: false
        user nullable: false
        platform inList: ["ios", "android"], blank: false
    }
    
    static mapping = {
        user column: "user_id"
    }
}
```

this establishes a simple relationship to a user. the key thing to notice is the unique token, that's what fcm and apns use to identify the device to send notifications. make sure that user field is an association to your actual user domain class. the platform field is optional but helpful to determine how to send the notification.

now you need some way to send the push message. i usually use a service class to encapsulate the logic. here's a snippet for sending push to fcm with java, it's also easy to adapt it to groovy, but let's use java here. you'll probably want to adapt it to your project setup:

```java
package com.example.push;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.messaging.FirebaseMessaging;
import com.google.firebase.messaging.Message;
import com.google.firebase.messaging.Notification;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.FileInputStream;
import java.io.IOException;

@Service
public class FcmService {

    @Value("${fcm.credentials.path}")
    private String credentialsPath;

    private FirebaseMessaging firebaseMessaging;

    @PostConstruct
    public void initialize() {
        try {
           FileInputStream serviceAccount = new FileInputStream(credentialsPath);

            FirebaseOptions options = new FirebaseOptions.Builder()
                    .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                    .build();

            FirebaseApp.initializeApp(options);
            firebaseMessaging = FirebaseMessaging.getInstance();
        } catch (IOException e) {
            System.out.println("error initializing firebase");
        }
    }

    public void sendPushNotification(String deviceToken, String title, String body) {
         Notification notification = Notification.builder()
                  .setTitle(title)
                  .setBody(body)
                   .build();


        Message message = Message.builder()
                .setNotification(notification)
                .setToken(deviceToken)
                .build();

        try {
           String response = firebaseMessaging.send(message);
           System.out.println("successfully sent message: " + response);
        } catch (Exception e) {
            System.out.println("error sending message: " + e.getMessage());
        }
    }
}
```

what this does is:

*   it initializes the firebase app when the application starts.
*   it loads the credentials from the given path in the application properties.
*   it provides a `sendPushNotification` method to send a message using the device token.

remember to add the firebase admin sdk dependency to your gradle file:

```groovy
implementation 'com.google.firebase:firebase-admin:9.2.0'
```

you can adapt this to use apns by using apns libraries instead, this will give a good start with firebase though.

now how do you use this? basically, wherever you need to send a notification from grails you inject the `FcmService` and call the method. for example when a user receives a message on a chat.

```groovy
// inside your service that handles messaging
class ChatService {
    def fcmService
    def deviceService //assuming you have a service for the device table access

    def sendMessage(fromUser, toUser, messageContent) {
       // do stuff with your chat logic like save the message

       def device = deviceService.findActiveDevice(toUser)

       if (device) {
          fcmService.sendPushNotification(device.token, "New Message from ${fromUser.username}", messageContent)
       } else {
           println "no active devices for user ${toUser.username}"
       }
    }
}
```

i’m assuming you have a device service there, but you can use gorm directly if you want. and please check the user and device data, not to create a mess.

a common mistake i’ve seen is forgetting about handling device token updates. device tokens can change, so you should have a way to update a token on your server if the mobile client notifies you the token changed. maybe you should add an endpoint on your server, so if the device token changes, the mobile client can call this endpoint and update it for your user. this will save you some headaches later.

about resources, i’d recommend reading the official documentation for fcm and apns. it has all the specific details about payload formats, error handling, and best practices. for the underlying concepts about push notification i would point to books like “push notification principles” by michael k. gunderson if you want a general introduction to push notifications, even if the books might not be updated to the latest versions it will give you a good starting point.

and one more thing, it’s always good to have some kind of logging on your server to keep track of all notifications being sent, you will thank yourself later for this. this is not strictly necessary but really helps for debugging purposes. you know, it is like forgetting to close the curly braces, or forgetting the semicolons, it's just a matter of time when you will regret not logging.

this should give you a pretty good starting point. push notifications can be tricky at the beginning, but with a bit of work they’re definitely achievable. remember, always validate your input, log your requests, and always, always check the official documentation, or ask more if you still have doubts.
