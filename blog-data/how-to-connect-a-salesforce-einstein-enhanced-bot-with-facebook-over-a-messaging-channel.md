---
title: "How to Connect a Salesforce Einstein Enhanced Bot with facebook over a messaging channel?"
date: "2024-12-14"
id: "how-to-connect-a-salesforce-einstein-enhanced-bot-with-facebook-over-a-messaging-channel"
---

alright, so you're looking to get your salesforce einstein enhanced bot chatting with folks on facebook messenger, huh? i've been down this road, and let me tell you, it's not always a walk in the park, but totally doable. i've built a few of these integrations in my time, mostly for startups trying to handle customer support with a little more automation, and it's generally a pretty interesting setup. the biggest problem is always the initial configuration.

first off, let's talk about the pieces you need. you've got your einstein bot, obviously, which you've hopefully already set up inside your salesforce org. this is where all your chatbot logic lives, the intents, the entities, and the dialog flows. it handles the natural language processing and decides how to respond to user input. then, you've got the facebook side of things, meaning your facebook page and the developer tools it comes with. that's where you'll get your page access token, which is essentially the key to let your bot talk to facebook messenger.

the magic connecting these two is typically going to involve a middleware service. while salesforce does allow for some direct channel connections, in my experience, especially for complex integrations with advanced bot features, using a middleware helps with things like custom authentication, payload modification, and error handling. it also offers greater flexibility and makes your setup easier to maintain in the long run. the middleware i've used most often is a simple node.js app, but python works too or serverless functions if you are into that. think of it as your translator, relaying messages back and forth between facebook and salesforce.

let’s sketch out how this flow generally works. a user sends a message to your facebook page. facebook sends a webhook notification to your middleware service. this notification contains the message text and other metadata. your middleware parses this, formats it to be salesforce friendly, and then sends it to your einstein bot api. the bot processes the message, creates a response, and sends that response back to the middleware. finally, your middleware forwards the bot’s response back to facebook, and the user sees it in messenger.

a common mistake i see folks make is trying to directly connect facebook to salesforce without any middleman. salesforce's platform events can handle a few things, but honestly, it’s more work to do it that way, especially once you move beyond super basic bot conversations. i remember one time i tried to do everything with platform events directly. took me nearly three days to debug a super strange authentication issue, it was such a pain. needless to say, after that, i was pretty much converted to the middleware approach.

now, let’s look at some code examples. this first snippet shows a basic express.js setup (node) for handling a webhook from facebook:

```javascript
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());

const facebookPageToken = 'your_facebook_page_access_token';
const salesforceBotEndpoint = 'your_salesforce_bot_endpoint';

app.post('/webhook', (req, res) => {
    const body = req.body;

    if (body.object === 'page') {
        body.entry.forEach(entry => {
            entry.messaging.forEach(event => {
                if (event.message && event.message.text) {
                    const messageText = event.message.text;
                    const senderId = event.sender.id;

                    // prepare payload for salesforce
                    const salesforcePayload = {
                        text: messageText,
                        userId: senderId,
                        channel: 'facebook'
                    };

                    axios.post(salesforceBotEndpoint, salesforcePayload)
                        .then(salesforceResponse => {
                            const botResponseText = salesforceResponse.data.text;
                            // send the response back to facebook
                            sendFacebookMessage(senderId, botResponseText);
                        })
                        .catch(error => {
                            console.error('error sending to salesforce', error);
                        });

                }
            });
        });
        res.status(200).send('event received');
    } else {
      res.status(404).send();
    }
});


function sendFacebookMessage(recipientId, text){

    axios.post('https://graph.facebook.com/v18.0/me/messages', {
        recipient: { id: recipientId },
        message: { text: text }
    }, {
        params: {access_token: facebookPageToken}
    })
      .catch(error => {
            console.error('error sending to facebook', error);
        });
}
// this is just for webhook verification during setup
app.get('/webhook', (req, res) => {
  const verifyToken = 'your_verify_token';
    const mode = req.query['hub.mode'];
    const token = req.query['hub.verify_token'];
    const challenge = req.query['hub.challenge'];

    if (mode === 'subscribe' && token === verifyToken) {
      res.status(200).send(challenge);
    } else {
       res.status(403).send();
    }
});

app.listen(3000, () => console.log('server running'));
```

this is a very basic node.js express server example. remember to install dependencies with: `npm install express body-parser axios`. you'd need to replace `'your_facebook_page_access_token'`, `'your_salesforce_bot_endpoint'`, and `'your_verify_token'` with your actual credentials. the `sendFacebookMessage` function sends the bot’s reply back to the user. also, note the webhook verification handler for setup from facebook.

next, let's peek at a python version using flask, because why not:

```python
from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

facebook_page_token = 'your_facebook_page_access_token'
salesforce_bot_endpoint = 'your_salesforce_bot_endpoint'

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if data['object'] == 'page':
        for entry in data['entry']:
            for messaging_event in entry['messaging']:
                if 'message' in messaging_event and 'text' in messaging_event['message']:
                   message_text = messaging_event['message']['text']
                   sender_id = messaging_event['sender']['id']

                   salesforce_payload = {
                         'text': message_text,
                         'userId': sender_id,
                         'channel': 'facebook'
                    }

                   try:
                     salesforce_response = requests.post(salesforce_bot_endpoint, json=salesforce_payload)
                     salesforce_response.raise_for_status()
                     bot_response_text = salesforce_response.json()['text']
                     send_facebook_message(sender_id, bot_response_text)
                   except requests.exceptions.RequestException as e:
                       print(f'Error sending to salesforce: {e}')

        return 'event received', 200
    else:
      return '', 404

def send_facebook_message(recipient_id, text):
    headers = {
        'Content-Type': 'application/json',
    }
    params = {'access_token': facebook_page_token}
    payload = {
        'recipient': {'id': recipient_id},
        'message': {'text': text}
    }

    try:
       response = requests.post('https://graph.facebook.com/v18.0/me/messages',
                                  params=params, headers=headers, json=payload)
       response.raise_for_status()
    except requests.exceptions.RequestException as e:
          print(f'error sending to facebook: {e}')



@app.route('/webhook', methods=['GET'])
def verify():
    verify_token = 'your_verify_token'
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode == 'subscribe' and token == verify_token:
        return challenge, 200
    else:
        return '', 403


if __name__ == '__main__':
    app.run(port=3000, debug=True)

```

install dependencies with `pip install flask requests`. like the node example it requires the same token and endpoint configuration. this snippet does the same as the first one, but in python, which is why i said it works with different middlewares. it processes the webhook from facebook, extracts the message and sender id, prepares the payload for salesforce, send it to your bot, gets the response from it and sends back the answer to the user over facebook.

and, just to cover some of the salesforce side of things, although the bot configuration is more gui than code, here’s a minimal json payload structure that your salesforce bot endpoint should be expecting, based on those snippets:

```json
{
  "text": "the user typed message here",
  "userId": "the unique facebook id from the user",
  "channel": "facebook"
}
```

this is a pretty simple format, but you can always extend this to pass more info to your bot. i've passed session identifiers and metadata before, which has been pretty useful for a few cases.

on the einstein bot side, make sure you create a new channel within the bots setup. this channel is essentially how salesforce identifies where your messages are coming from. in your bot’s dialog flow, you will need to access the 'channel' variable, or whatever you defined your channel identifier, to ensure that your bot is responding correctly in the proper context. for example, if facebook requests data on a certain format, and the bot receives information in a different format, the bot should be able to translate that properly. i once forgot about a date format mismatch between a website and my bot... it took me hours to realise it was the reason for a broken functionality.

as for reading material, forget those basic tutorial sites. i'd recommend checking out "speech and language processing" by daniel jurafsky and james h. martin for a deeper dive into nlp concepts if you want to understand more about how bots actually "understand" text and to have better ideas to extend your integrations; and "designing conversational interfaces" by nicolas rohrbach for a more in-depth discussion about the ux of chatbots. for more specific info about salesforce's bot api and features, the salesforce documentation, it has improved quite a lot in the past year, is pretty good and generally up to date and worth a check.

and, that’s pretty much it. remember the middleware is key, treat those access tokens with extreme care, always log stuff on your middleware (it will save you hours), and ensure your einstein bot is properly configured to work with external channels. it's also extremely useful to have a testing setup for this, where you can send messages without the facebook intermediary for debug. a final comment, i once got lost on the facebook graph api docs for a whole morning. it was like looking into a maze, that's why it's useful to keep your code as simple as possible. and... that's it, any questions?
