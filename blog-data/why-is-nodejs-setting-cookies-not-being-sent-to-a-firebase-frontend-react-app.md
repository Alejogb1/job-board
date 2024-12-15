---
title: "Why is Nodejs setting Cookies not being sent to a firebase frontend react app?"
date: "2024-12-15"
id: "why-is-nodejs-setting-cookies-not-being-sent-to-a-firebase-frontend-react-app"
---

alright, let's talk about why your nodejs backend isn't playing nice with your react firebase frontend when it comes to cookies. it's a classic headache, and i've definitely been there more times than i'd care to count.

so, you're sending cookies from your nodejs server, expecting your react app running in firebase hosting to happily accept them, but nothing’s happening, the client side doesn't seem to receive it. sounds familiar, the first time i encountered this, i swear i aged about five years in the space of a single afternoon. i had a similar setup back then, nodejs for the api, angular for the front end, and it was hosted, i thought everything should work out of the box, or so the tutorials said. let me walk you through some of the common pitfalls and how i eventually managed to solve them.

first off, the core issue usually boils down to cross-origin resource sharing (cors) and the way browsers handle cookies. when your nodejs server and your firebase app are on different domains (and they almost certainly are), browsers enforce strict policies for security. if your server is sending cookies without proper configuration, your browser will likely reject them, or worse, silently ignore them.

i’m assuming you're using something like express for your node server. let's start with the basics. if you haven't configured cors, there's a very high probability that it's the culprit. you should explicitly allow your firebase domain to access your api. here’s a simple example of how you would do this.

```javascript
const express = require('express');
const cors = require('cors');
const app = express();

const corsOptions = {
    origin: 'https://your-firebase-app-id.firebaseapp.com',
    credentials: true, // this is crucial for cookies
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
    allowedHeaders: ['Content-Type', 'Authorization'],
};

app.use(cors(corsOptions));

// rest of your app here...

app.get('/set-cookie', (req, res) => {
    res.cookie('mycookie', 'somevalue', {
        httpOnly: true,
        secure: true,
        sameSite: 'none', // important
    });
    res.send('cookie set!');
});

app.listen(3000, () => console.log('server running on port 3000'));

```

now, let's break down this snippet. `cors(corsOptions)` is the main player here. the `origin` is the url of your firebase hosting application. the `credentials: true` flag is absolutely essential – it tells the browser that your server should be able to use cookies. without it, the cookies simply won’t be included in the request. i remember when i forgot this. i was banging my head against the wall for hours. that tiny little setting caused me so much grief. the `methods` field specifies what http verbs you will allow to your endpoints and `allowedHeaders` specifies what headers are accepted in the request, for example `Content-Type` or `Authorization` for example if you are passing a bearer token in the header.

another crucial point is the settings of the `res.cookie()`. first `httpOnly: true` makes it so your cookies cannot be read by javascript client-side, adding a level of security. the `secure: true` ensures the cookie is only sent over https. the `sameSite: 'none'` flag is particularly important when your backend and frontend are on different domains. with `samesite: 'lax'` or `strict'` this won't work. without setting `samesite: 'none'`, browsers will likely refuse to send cookies to different domains. it’s one of those things that you wouldn’t think is so important but really is, once again, learned the hard way. in the example, i set to send the cookie when calling the endpoint `/set-cookie`.

moving to the client side, in your react app, you might also need to ensure that your requests are being sent with credentials as well. this is mostly relevant if you’re using something like `fetch` or `axios` directly, if you are using firebase authentication, then the cookies are handled by firebase itself. here’s a simple `fetch` request example.

```javascript
fetch('http://your-nodejs-server:3000/set-cookie', {
    method: 'GET',
    credentials: 'include',
})
.then(response => {
    console.log('response', response);
})
.catch(error => {
    console.log('error', error);
});
```

here, the `credentials: 'include'` option instructs the browser to send cookies with this request. sometimes, i used to forget this detail, and it just won't work. now i always check it when using fetch.

but, before going crazy checking your code, let me ask, are you sure that your nodejs is accessible from the firebase app? it sounds obvious but sometimes during development, the node server might be in a local environment and you are expecting that localhost is accessible by your hosted firebase app, this is not the case. i remember that i was using one of those tunnelling services, thinking that my nodejs was reachable from outside, and it wasn't. you should be mindful of this detail. also, make sure that the url you are hitting on the client side is the correct one.

finally, a tricky case that i encountered once, even with all this set correctly, is when you're setting up a reverse proxy. for example, if you are using a proxy server before hitting your nodejs instance, you will have to configure this proxy to forward the cookies. depending on the proxy you're using, the details can vary, but typically, you need to configure your proxy to pass along the `set-cookie` header in the response, and the cookie header in the request to your upstream server.

let me quickly describe a very particular error i had once. it had to do with subdomains. you can have multiple subdomains for your app, and i messed up the cors configuration in this case, because of the various subdomains of my backend api. it was very confusing because the error was still related to cors issues, but not the main domain of my application. it turns out the browser is very picky and even if you set the main domain in the cors configuration, it will still reject cookies from different subdomains, be mindful of this. i spent days on that error. i still remember the frustration.

also if you're running your nodejs server on a different port during development (like `localhost:3000`), but your firebase app is running on `your-firebase-app-id.firebaseapp.com`, the cors policies will block those requests. you should either configure your backend to respond to your firebase domain when you are on production or using a reverse proxy in production to redirect from the same domain to your backend. during development you could bypass this with proxy configurations in the `package.json` file of your react project, but this is only for development. this one also made me lose a lot of hair.

so, in summary, make sure:

*   you have correctly configured cors in your nodejs server, including the `credentials: true` setting.
*   the `samesite` attribute of the cookie is set to `none`.
*   the react client is sending requests with `credentials: 'include'` when using `fetch` or a similar method.
*   you are calling the right url from your client.
*   your backend url is accessible from your frontend app (firebase).
*   you are not having some issue with subdomains and if you are, make sure the cors configuration considers the subdomains.
*   if you have reverse proxy make sure it's configured to pass cookie headers.

if you've done all of that, you should be golden.

here’s another code snippet with the proxy example, in this case using `axios`.

```javascript
import axios from 'axios';

axios.defaults.withCredentials = true; // for cookies

const fetchData = async () => {
    try {
    const response = await axios.get('http://your-nodejs-server:3000/set-cookie');
    console.log('response', response);
    }
    catch (error) {
    console.error('error', error);
    }
};

fetchData();
```

this snippet shows how to use `axios`, if that is what you are using, to achieve the same result, i have used both methods in the past, both work. the difference with fetch and axios is that `axios.defaults.withCredentials = true` sets the `credentials` to include globally, so every call will carry the cookies.

one thing that used to confuse me is that browser's developer tools can make you think that the cookies are set, but the browser won't send them. it's crucial to also check the *request* headers on the browser developer console, to verify if the cookies were actually sent with the request. usually, if you don't see it there, it won't work. check both the response and the request on your browser.

for a deeper dive into the intricacies of cors and cookie handling, i would highly recommend looking at 'http: the definitive guide' by david gourley and brian totty. it's a bit of a hefty read, but it's the bible when it comes to understanding http. i had to study it back in the day.

and before you ask if it could be the firebase configuration, in my experience, unless you are using firebase functions as your backend, firebase hosting doesn't have much to do with this. the problem is almost exclusively between your browser, your nodejs configuration, and maybe any reverse proxy. i remember once i was sure that it was firebase fault, and i was wrong. haha.

anyways, that's about it. i hope this helps. good luck with your project! if none of this works, feel free to come back with more details, and we can see if we can help you with more specific cases.
