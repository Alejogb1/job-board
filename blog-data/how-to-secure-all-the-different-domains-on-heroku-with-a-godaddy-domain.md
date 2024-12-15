---
title: "How to secure all the different domains on Heroku with a Godaddy Domain?"
date: "2024-12-15"
id: "how-to-secure-all-the-different-domains-on-heroku-with-a-godaddy-domain"
---

so, you're looking at how to get your heroku app playing nicely with a godaddy domain, huh? been there, done that, got the t-shirt – and probably a few stress-induced grey hairs to show for it. it's a fairly common setup but the details can get a bit fiddly if you haven’t done it before. let’s break it down, less like a formal tutorial, more like a chat with someone who's fought the good fight and lived to tell the tale.

first off, let’s assume you have a heroku app up and running. it’s humming along at its heroku subdomain, something like `your-app-name.herokuapp.com`. you also have your godaddy domain, let’s say `your-domain.com`. the goal is to get `your-domain.com` (and maybe `www.your-domain.com`) pointing to your heroku app, and to make sure everything is secure, which, frankly, in this day and age, should always be the default.

the critical piece here is the dns (domain name system). dns is basically the internet's phonebook. it translates human-readable domain names into the ip addresses that computers actually use to find each other. we need to tell godaddy that requests for `your-domain.com` should be routed to the heroku servers hosting your app.

the easiest way to do this is by setting up what’s called a `cname` record with godaddy. a cname record essentially says, “hey, `your-domain.com` is really just another name for the thing at `your-app-name.herokuapp.com`”. this works great for the `www` subdomain as well.

here’s the process, step by step (and this is where it gets slightly technical):

1.  **heroku setup:** first, you need to add the custom domain to your heroku app. you’ll do this through the heroku cli or the heroku dashboard. in the cli it's a command like:
    ```bash
    heroku domains:add your-domain.com
    ```
    and:
    ```bash
    heroku domains:add www.your-domain.com
    ```
    this tells heroku, "expect requests for these domains, i'll be pointing them to you". heroku will then provide you with dns target. this target may be something like `your-app-name.herokuapp.com`, or in some cases, it may be a value for the `heroku ssl endpoint`. take note of this target (we’ll need it in a moment). if you happen to be setting up https for all your domains you would likely want to use the ssl target endpoint value for it. keep this heroku output handy; it’s your golden ticket.

2.  **godaddy dns settings:** now, head over to your godaddy account. find your domain, and then look for where you can manage the dns records. this is usually in a section called “dns management”, “manage dns”, or something similar. you'll need to add two new records here, one for your root domain and another for the `www` subdomain.
    *   **`www` cname record:** add a new dns record of type `cname`. set the “host” to `www`, and the “points to” or “value” field to the heroku target heroku provided you in the previous step, example `your-app-name.herokuapp.com`.
    *   **`@` or root cname record:** here’s the trickier one, the root domain. now the root domain, (`your-domain.com`) is also sometimes represented as `@` in godaddy dns record settings. godaddy (unlike some other registrars) does not support cname record at the root (@). this means you cannot create a cname for the base domain, `your-domain.com`, you need an `a` record for that. what `a` records do is point the domain to specific ip address which means that it needs to point to the ip address of the heroku platform where the app is running. to work around this and because heroku's ip addresses can change, you’ll need to use the `heroku ssl endpoint` or use their recommended workaround which is use an `a` record that points to the heroku’s alias hostname `proxy.heroku.com`. this alias host name contains heroku's ip addresses. create the record as type `a` record and point to `proxy.heroku.com`.

3.  **godaddy setup, a records alternative**: there is another alternative, which i find more useful if you are setting up multiple applications in heroku under the same domain. in that case we can create `cname` records for both `www` and also for root domain. to make this work you need to create a `cname` record for the root domain and point to heroku ssl domain endpoint which in our example could be something like `your-app-name.herokudns.com`.

here is an example of what the dns records in godaddy could look like.

```
 type    | host   |  value
-------------------------------------------
    a    |   @    | proxy.heroku.com
 cname   |  www   |  your-app-name.herokuapp.com
```

or as an alternative we can use:
```
 type    | host   |  value
-------------------------------------------
 cname    |   @    | your-app-name.herokudns.com
 cname   |  www   |  your-app-name.herokudns.com
```
choose the one that fits your needs more, it depends if you are using root domain only or if you are planning on creating several apps with different subdomain names, which in that case the second alternative makes more sense.

4.  **https setup:** finally, for that sweet, secure https, heroku offers automatic certificate management. heroku uses lets encrypt to generate the certificates for you if you are using a domain pointed to your application. lets encrypt is a free, automated, and open certificate authority. once the dns settings have propagated (which can take a little while – dns servers have their own agendas) heroku should automatically generate the tls/ssl certificates for you. heroku takes care of the complexities of certificate generation and renewal. this means you get the padlock in your browser, the secure connection, and all the good stuff.

now, a few things i learned the hard way:

*   **propagation time:** dns changes aren’t instant. it can take anywhere from a few minutes to a couple of hours for dns servers around the world to update. this is something that caused me great amounts of panic the first time i did this, but it is common and there is nothing you can do other than wait. be patient. if things don’t work immediately, don’t start changing things wildly – wait a bit and try again. there are websites that let you check dns records like `dnschecker.org`. use those to make sure the records have been propagated correctly.
*   **www vs non-www:** decide if you want users to access your site via `www.your-domain.com` or just `your-domain.com`. you can redirect one to the other if you prefer one format over another. in most cases its better to enforce the usage of only `www` or just the root domain and redirect the one that is not used to the other version.
*   **multiple domains:** if you need to secure a whole bunch of subdomains (like `api.your-domain.com` or `blog.your-domain.com`), you can add cname records for each of them to point to the same heroku app or even to different heroku apps depending on your requirement. the `a` record approach for the root domain might need to be changed to a `cname` pointing to heroku ssl endpoint and then create a record for each subdomain, for instance, if you wanted to have a `blog` app you would create a `cname` record, setting “host” to `blog` and the “points to” field to the heroku ssl endpoint of your blog app.
*   **godaddy interface:** godaddy’s interface can be a bit… let’s call it “unique”. if you’re not used to it, give yourself some time to figure it out. it changes frequently so it can be confusing to find things in the menus. i once spent about twenty minutes trying to find the “add record” button. i was so frustrated i was about to start my own registrar just so i could use a cleaner interface.

here is an example of how to redirect `non-www` version to `www` using express:

```javascript
const express = require('express');
const app = express();

app.use((req, res, next) => {
  if (req.headers.host === 'your-domain.com') {
    res.redirect(301, 'https://www.your-domain.com' + req.url);
  } else {
    next();
  }
});

// your other routes and middleware
app.get('/', (req, res) => {
  res.send('hello world!');
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`listening on port ${port}`));
```

you can also use the `heroku-ssl-redirect` module for handling the redirection:

```javascript
const express = require('express');
const sslRedirect = require('heroku-ssl-redirect');
const app = express();
app.use(sslRedirect());

// your other routes and middleware
app.get('/', (req, res) => {
  res.send('hello world!');
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`listening on port ${port}`));
```

for more in depth knowledge about the topics covered i recommend you check dns and tcp/ip books like “tcp/ip illustrated” by stevens, or “dns and bind” by albitz and liu. heroku's documentation is also very useful for getting up to date on its specific requirements, you should always go there first.

so yeah, that's about it. it's a bit of a dance between heroku and godaddy, but once you get the hang of it, it becomes fairly straightforward. just remember to be patient with dns propagation, double-check your settings, and try not to pull all your hair out. happy coding.
