---
title: "How to add security headers to the Apache Airflow Web UI?"
date: "2024-12-15"
id: "how-to-add-security-headers-to-the-apache-airflow-web-ui"
---

alright, so you're looking to beef up the security on your airflow web ui, huh? i've been there, trust me. it's one of those things that you often overlook at the beginning when you're just trying to get things off the ground, but it becomes critically important once you actually start running stuff in production. it's like building a house and forgetting to put locks on the doors. you will eventually need it.

i remember when i was first setting up a data pipeline for a small startup, we were so focused on just getting the etl process working that security was an afterthought. airflow was just kinda humming along, happily displaying all of our dag configurations and variable values to, well, anyone who happened to stumble upon the web interface. i didn't have anything super sensitive, but it was still not something you want out in the open. it was quite a stressful situation. this was back in 2018, airflow was still in its early days. i was a young padawan with a lot to learn and had a long way to go. i was just trying to survive the data pipelines trenches at that point.

so, let’s get down to the specifics. you want to add security headers, which essentially are http response headers that provide instructions to the browser on how it should behave when dealing with your website. they're pretty key for things like preventing clickjacking, cross-site scripting (xss) attacks and other common web vulnerabilities. the airflow web ui runs as a wsgi application, it's basically a python application served over http. so, the way to add security headers is by basically customizing how airflow does http.

the most common way to handle this is by using a reverse proxy server that sits in front of your airflow instance. this proxy server handles the http requests first, adds all the necessary headers, and then forwards the request to the airflow web server. this approach lets you centralize all your security configurations at one place, instead of trying to embed security into the airflow application itself, which is not really the intended use case. plus, it's much easier to manage.

i'd recommend using nginx or apache for this. both are solid reverse proxies and have tons of resources and documentation available. i'm a nginx guy myself, so i'll show you the nginx way for an example, but the same concepts translate easily to apache.

first, let’s start with a basic nginx configuration. you'll want to set up a server block that listens on the port you want to access airflow on (usually 80 or 443). then, you'll need a `location` block that proxies the requests to the airflow web server. this part is standard and simple. something like this:

```nginx
server {
    listen 80;
    server_name your_airflow_domain.com;

    location / {
        proxy_pass http://your_airflow_webserver_address:8080; # use your airflow url and port
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

```

in the code above remember to replace `your_airflow_domain.com` with the actual domain or ip address of your airflow instance, and also make sure to replace `your_airflow_webserver_address` with the address of your airflow web server, if they are not in the same machine. the usual port is 8080, but this can be configured in airflow.cfg, so please double check that first.

now here comes the meat of what we want: adding security headers. to do that, you use the `add_header` directive inside the `location` block. here is a good set of security headers to get you started. this code can be added within the location block of your nginx configuration:

```nginx
        add_header X-Frame-Options "DENY";
        add_header X-Content-Type-Options "nosniff";
        add_header X-XSS-Protection "1; mode=block";
        add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:;";
        add_header Referrer-Policy "no-referrer";
        add_header Permissions-Policy "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

let me break down these headers really quick:

*   `x-frame-options: deny;` this prevents clickjacking attacks by preventing your site from being embedded in an iframe.
*   `x-content-type-options: nosniff;` this prevents browsers from trying to guess the content type of a resource, mitigating some forms of xss attacks.
*   `x-xss-protection: 1; mode=block;` this enables the browser’s built-in cross-site scripting filter.
*   `content-security-policy` this is a powerful header that lets you specify from which sources the browser is allowed to load resources. here i've set a fairly basic policy that allows everything from the same domain, but allows in-line javascript and styles for now. you can fine-tune this later.
*   `referrer-policy: no-referrer;` this controls how much information is sent in the referrer header. setting it to `no-referrer` prevents sensitive data from leaking.
*   `permissions-policy`: this header is used to control which browser features your site is allowed to use.
*   `strict-transport-security`: this forces the browser to use https for all requests, even if they initially started over http. the `always` argument ensures this header is always included in the response, even for error pages.

note that the `content-security-policy` header is complex, and may require some adjustments, depending on your specific airflow configuration and which resources you are using. you may find that some airflow plugins or custom code you might have introduced requires changes to this header. the main goal here is to start with something secure and less restrictive, and then you can progressively tighten it as you go on. a good way is to use reports on the browser console to check for csp violation and change the policy.

after making these changes to the nginx config, just restart your nginx server with `sudo systemctl restart nginx` or equivalent.

another important thing you should look into is setting up https with a valid certificate. let's face it, if you're setting up security headers, you should absolutely use https. you can get a free certificate from let's encrypt with tools like certbot or equivalent tools that can help you with the automation and generation of this.

here is a simple example on how to integrate this on a nginx setup, inside the server block of your domain:

```nginx
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/your_airflow_domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your_airflow_domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3; # configure modern protocols
    ssl_ciphers  ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    ssl_prefer_server_ciphers on;

    location / {
    #proxy pass and all headers from above goes here
    }
```
replace `/etc/letsencrypt/live/your_airflow_domain.com/` with the actual path where your certificate files are. also, make sure the protocols used in `ssl_protocols` and `ssl_ciphers` are updated. there are online generators to help you with that. it’s not good to use older protocols and ciphers, so you should always go for the strongest ones. i did hear a joke a while ago that only the most secure protocols should be used, and that the weaker ones should just take a hike, but that's neither here nor there, we should focus on security.

remember, this is just a starting point. you should consult with a security professional to do a proper risk assessment for your airflow instance. a good resource for more information on http security headers would be the owasp secure headers project. you can also dive deep into the internet engineering task force (ietf) standards and publications for even more specifics on http security headers. those papers contain a wealth of knowledge to fully understand them. there are also great books out there like “bulletproof ssl and tls”, by ivan ristić, if you want a more detailed and in depth look at the transport security level.

finally, remember to keep your airflow version up to date with the latest patches and security fixes, as well as all your packages. security is an ongoing process, not a one-time task. it's like a garden, you always need to tend to it. i learned that the hard way.
