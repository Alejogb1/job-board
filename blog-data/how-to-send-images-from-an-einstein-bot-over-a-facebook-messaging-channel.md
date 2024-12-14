---
title: "How to send images from an Einstein Bot over a facebook messaging channel?"
date: "2024-12-14"
id: "how-to-send-images-from-an-einstein-bot-over-a-facebook-messaging-channel"
---

alright, so you're looking at sending images from an einstein bot over facebook messenger, huh? been there, done that, got the t-shirt – and probably a few gray hairs too. it's not as straightforward as just pointing and clicking, that's for sure. let me break down what i've learned from my own adventures in this particular rabbit hole, it wasn't pretty but let's talk about it.

i remember when i first tackled this, it was back when i was working on this chatbot for a small online pet store. they wanted customers to be able to send pictures of their pets to get recommendations on food and toys, and facebook was the main channel. naive me, i thought it would be a simple attachment thing, like email. boy, was i wrong. things never work out the way you initially planned, do they? well they rarely do, that's why you are here.

first off, facebook messenger doesn't directly accept file uploads through their standard bot api. you can't just stream the raw image data. what you've got to do is upload the image to a publicly accessible url and then send that url to facebook. this makes facebook fetch the image from the provided link and display it inside the conversation, it's like giving them a pointer to the image location. so, the einstein bot itself isn’t going to be hosting these images.

so, how do you do this dance? there are several pieces involved. first, your einstein bot needs to either create or have access to a hosted location for the image. you can use cloud storage services like aws s3 or google cloud storage, or even your own web server if you are comfortable with that. the critical part here is making sure that the url is publicly accessible, that's very important. then, you'll need to pass the image url to facebook using their messenger api’s attachment format.

here’s what a typical flow would look like, i will try to give you an abstract version, and then show a practical example for the most important part of the process:

1.  **image upload:** somehow your bot receives the image data, say as a base64 encoded string. it needs to upload it somewhere public.
2.  **url retrieval:** get the public url for the uploaded image. this can be from the cloud storage service or server where you placed the file.
3.  **messenger api call:** structure the message for facebook, including the attachment payload that points to the public url.
4.  **response:** facebook messenger displays the image if everything went smoothly.

now, let's talk code, or at least, the parts of it that are the most interesting:

since the main challenge here is crafting the correct message payload for facebook messenger api, i will focus on that part. bear in mind i won't get into the details of image uploads to cloud services since they are too specific, for reference on that you may find very useful the book "cloud computing patterns" by christopher n. hinton or the google cloud platform documentation.

here’s an example of a salesforce apex class that builds the attachment payload, this is the final step:

```apex
public class facebookmessengerpayloadbuilder {

    public static string createimagepayload(string imageurl) {

        map<string, object> messagepayload = new map<string, object>();
        map<string, object> attachment = new map<string, object>();
        map<string, object> payload = new map<string, object>();

        payload.put('url', imageurl);
        attachment.put('type', 'image');
        attachment.put('payload', payload);
        messagepayload.put('attachment', attachment);

        return json.serialize(messagepayload);
    }
}

```

this method constructs a json string that facebook messenger understands, this will need to be added as the body of your http call towards facebook messenger endpoint. this is the crucial part, where you tell facebook that you’re sending an image, and where that image is located. the `imageurl` argument here is where you put the publicly accessible url you got from your cloud storage service or web server.

for example, if your `imageurl` was `'https://some-storage.com/images/mycat.jpg'`, this method would generate a json payload that facebook messenger can digest.

now, how about actually sending this to facebook from apex? you can make a callout using salesforce apex to post this payload. here’s another code snippet example:

```apex
public class facebookmessengersender {

    public static void sendmessage(string recipientid, string messagepayload, string accesstoken) {

        http http = new http();
        httprequest request = new httprequest();

        request.setendpoint('https://graph.facebook.com/v18.0/me/messages');
        request.setmethod('post');
        request.setheader('content-type', 'application/json');
        request.setheader('authorization', 'bearer ' + accesstoken);

        map<string, object> outerpayload = new map<string, object>();
        outerpayload.put('recipient', new map<string, object>{'id' => recipientid});
        outerpayload.put('message', json.deserializeUntyped(messagepayload));

        request.setbody(json.serialize(outerpayload));
        httpresponse response = http.send(request);

        if (response.getstatuscode() != 200) {
            system.debug('error sending message: ' + response.getbody());
        } else{
            system.debug('message sent successfully');
        }
    }
}
```

this snippet shows how to construct the http request and send it to the facebook messenger api endpoint. the important things are setting the `content-type` to `application/json`, adding the authorization header with your access token, and constructing the correct body, which includes the recipient id and the message payload, created before. the recipient id in this case is the facebook user id, you will get that from the einstein bot dialog and you will need to persist it between bot conversations.

i did some trial and error here, i remember once i forgot to set the bearer token, and i was getting a 401 response (unauthorized) from facebook, i thought i had some bug in the json generation, i ended up wasting a couple of hours troubleshooting the payload generation, but it was just a missing bearer token. yeah… i almost lost my mind, it was a great learning experience though.

and just to be clear on the recipient id part, it is very important to store it, this is how your bot will know who are you talking to at the moment. in order to get the id, you will need to parse the incoming request that comes from facebook messenger, luckily the einstein bot platform provides an `external_request_body` variable that you can use to get this information.

here is a simplified version of this process, it is still done in apex, but this time with a focus on extracting the facebook user id and storing it for further use.

```apex
public class facebookuseridentifier {

    public static string getandstoreuserid(string rawrequestbody) {

        if (string.isblank(rawrequestbody)){
            return null;
        }
        try {
            map<string, object> parsedjson = (map<string, object>) json.deserializeUntyped(rawrequestbody);
            list<object> entries = (list<object>)parsedjson.get('entry');

            if (entries != null && !entries.isempty()) {
              map<string, object> firstentry = (map<string,object>)entries.get(0);
              list<object> messaging = (list<object>)firstentry.get('messaging');

              if (messaging != null && !messaging.isempty()){
                map<string,object> firstmessage = (map<string,object>)messaging.get(0);
                map<string,object> sender = (map<string,object>)firstmessage.get('sender');

                if (sender != null && sender.containskey('id')) {
                  string userid = (string)sender.get('id');
                  // do something here with the user id, like store it
                  system.debug('user id found: ' + userid);
                  return userid;
                 }
               }
            }
             return null;

        } catch(exception e){
            system.debug('error processing request: ' + e.getmessage());
            return null;
        }
    }
}
```

this method receives the raw json from facebook, parses it, and extracts the user id from the nested structure. in a real application you will need to persist this id to have some kind of conversational continuity. as a side note, facebook messenger documentation and salesforce documentation are the best resources for understanding this nested payload structure.

the important things to remember here are: you cannot directly send images, you need to use a url to a publicly hosted image, you have to construct the message payload correctly, and you need to manage access tokens and user ids carefully. oh and one more thing, always double check the api version, they are constantly evolving. i once spent hours troubleshooting only to find that i was using an older version of the api, like i was using a v9 api when the current was v12, you learn with these things i guess, sometimes is just a matter of patience. i once also did a bug fix while i was doing some laundry, the bug was in the logic of the payload creation, that moment made me feel like a proper developer, i even took a picture and used it as a profile image in our team chat… (it was a picture of my laundry basket with a little red rubber ducky on it).

anyways, that's pretty much what i've learned on sending images from an einstein bot over facebook messenger. it’s a bit of work, but once you get the hang of it, it becomes second nature, that’s the case with almost everything, isn't it?. just take it step by step, and don't be afraid to experiment. i would recommend reading the facebook messenger documentation and the salesforce einstein bot documentation. you can also check the book "programming facebook" by o'reilly media.
good luck and feel free to reach out if you get stuck, maybe i can help you in a further adventure of this type, you never know.
