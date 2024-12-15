---
title: "How to add security headers to an Apache Airflow Web UI?"
date: "2024-12-15"
id: "how-to-add-security-headers-to-an-apache-airflow-web-ui"
---

alright, so you're looking to beef up the security on your airflow web ui, understandable. it's not exactly a fortress straight out of the box, and adding those security headers is a pretty standard way to get a bit more peace of mind. i've definitely been down this road before, and it's a good habit to get into with web-facing stuff like this.

let me tell you a story. back in the day, i had this airflow setup running on a cloud instance for a small project, nothing too crazy. one day, the security team at the company i was working with ran a scan and came back with a bunch of flags about missing headers. i felt like a noob; hadn't even thought about it properly. it was one of those ‘learn by doing’ moments, and i spent a good part of a weekend figuring out how to configure it properly. i'll try to help you skip that specific step of the learning curve with what i learned.

first things first, what headers are we even talking about? we want to add a few key ones to get good basic protection. headers like `x-frame-options`, `x-content-type-options`, `strict-transport-security` (hsts), and `content-security-policy` (csp) are your usual suspects. they all do slightly different things, but combined they create a good defensive layer against a bunch of common attacks.

apache airflow uses flask under the hood for its web ui, so the way you tackle this is by adding those headers at the webserver level. since airflow usually runs behind a reverse proxy like nginx or apache2, that’s where we will focus, in my example case, i am going with apache2 because i tend to use apache a lot more than nginx since it is easy to configure.

here's the basic gist: you'll need to tweak your apache config file to include these headers. usually, this is somewhere in your virtualhost config. let's take a look at an example, and we will also touch on some things you might want to consider further:

```apache
<virtualhost *:80>
    server_name your_airflow_domain.com

    # redirect http to https
    redirect permanent / https://your_airflow_domain.com/

    # add some nice logs
    errorlog ${apache_log_dir}/airflow-error.log
    customlog ${apache_log_dir}/airflow-access.log combined

</virtualhost>

<virtualhost *:443>
        server_name your_airflow_domain.com

        # ssl configuration goes here

    # this is important
    # header settings
        header set x-frame-options "sameorigin"
        header set x-content-type-options "nosniff"
        header set strict-transport-security "max-age=31536000; includeSubDomains"
        header set content-security-policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:;"


        # proxy pass to airflow's gunicorn server
        proxyPass / http://localhost:8080/
        proxyPassReverse / http://localhost:8080/

        # log settings
        errorlog ${apache_log_dir}/airflow-error.log
        customlog ${apache_log_dir}/airflow-access.log combined

</virtualhost>
```

in this config, i'm setting `x-frame-options` to `sameorigin` to prevent clickjacking, `x-content-type-options` to `nosniff` to stop browser mime-sniffing exploits, and `strict-transport-security` with a `max-age` to force the use of https for a year, including subdomains.  the `content-security-policy` is where things get a bit more involved; i've set a basic one there that is good for starters allowing only content from same origin as the website, allowing unsafe-inline styles, some analytics and images as data: or self-hosted resources.

now, here's the deal: the `content-security-policy` is a beast. it's incredibly powerful but also very finicky. if you get it wrong, things on your airflow web ui might stop working, so be very careful to test things out and start very relaxed with the policy rules, then as you get more knowledge, get stricter with it. the example i put up there should do for the most part, but you might need to tweak it depending on what external libraries or services your airflow setup is using. for example, i had to allow google analytics because i was using it in a dashboard view. also, notice the 'unsafe-inline' keywords in the csp? that's a no-no if you want to be super strict, and you may need to adjust your site to not use inline styles or scripts, but for airflow, it is easier to allow them as for a beginning setup, and it is better to be a little bit secure than completely insecure.

after you make changes to apache's config, don't forget to reload apache2: `sudo systemctl reload apache2`.

also, i like to add the below block of code in the apache config, usually in the beginning of the `<virtualhost *:443>` block, and that allows me to control even more parameters related to headers, and also make the apache system more secure:

```apache
    # turn off server tokens
    serverTokens Prod

    # additional security header
    header always set referrer-policy "no-referrer-when-downgrade"

    # prevent mime sniffing
    header set x-content-type-options "nosniff"

    # protect against xss attacks
    header set x-xss-protection "1; mode=block"
```
the `serverTokens Prod` setting will turn off the server signatures which may give away more information that it is necessary, the `referrer-policy` sets rules when and what data to send in the header, making it safer in the process, and the `x-xss-protection` will mitigate against cross site scripting attacks.

now, let's discuss that content security policy in more detail, i mean, it deserves some special attention, right? you see the `default-src 'self'`? this one says that, unless stated otherwise, the website will only allow loading resources from the same origin, that is your domain or subdomain. then the `script-src 'self' 'unsafe-inline' https://www.google-analytics.com` allows scripts from the same origin, unsafe-inline scripts, and google analytics specifically. the `style-src 'self' 'unsafe-inline'` does the same but for styles, and the `img-src 'self' data:` is for images, allowing local images and data images. and `font-src 'self' data:` the same but for fonts, allowing local fonts and data fonts. those are common to have, but you may need to adapt it if you need to have some other external dependencies or integrations.

a quick note: if you're dealing with websockets, you'll likely need to add a `connect-src` directive to your csp to include the allowed origins for websocket connections. you need to add that if for instance you are using an external web app to render the progress of a task or something like that.

also, this is a good moment to remind you that https is critical.  if you are exposing your apache airflow to the internet without it, you are essentially broadcasting your data in clear text, which is like writing all your passwords on a public board for anyone to see and steal. make sure you have a valid ssl/tls certificate set up correctly.

here is another snippet to configure tls/ssl in apache:

```apache
        sslcipheresuite ecdhe-rsa-aes128-gcm-sha256:ecdhe-ecdsa-aes128-gcm-sha256:ecdhe-rsa-aes256-gcm-sha384:ecdhe-ecdsa-aes256-gcm-sha384:kdh-rsa-aes128-gcm-sha256:kdh-rsa-aes256-gcm-sha384:aes128-gcm-sha256:aes256-gcm-sha384
        sslprotocol all -sslV2 -sslV3 -tlsV1 -tlsV1.1
        sslhonorcipherorder on
        sslcertificatefile /path/to/your/ssl/certificate.crt
        sslcertificatekeyfile /path/to/your/ssl/private.key
        sslcertificatechainfile /path/to/your/ssl/ca_bundle.crt
```
the `sslcipheresuite` will only allow very secure ciphers, the `sslprotocol` will make sure that old and insecure protocols are not accepted, and the `sslhonorcipherorder` makes sure that the cipher you picked is used and not another that is maybe less secure, and the last 3 ssl settings are to load your keys and certificate information.

and since we are talking about apache config, let's just do this here, because it is important: consider tightening up your apache config generally. things like disabling unused modules, limiting the apache user's permissions, and keeping the software updated are all worthwhile steps to take. security is not a single thing, it is layered, and all layers matter.

for more in-depth knowledge, i always recommend the owasp (open web application security project) guides, they are very complete and will give you more understanding about the attacks that can happen and how to prevent them, and also the book "bulletproof ssl and tls" by ivan ristić can give you more guidance on ssl related topics. the ietf documentation for the headers (those are usually rfc documents) are also super useful when you need to understand better how those technologies work, and also the mozilla's web security documentation gives a good overview and understanding. there are a bunch of other books too, like “the tangled web” by michał zalewski, it’s an old one, but it teaches a lot of the history and context of security in the web, really useful to learn why things are the way they are today.

now, one more important thing: after implementing these headers, use tools like the web page test or the mozilla observatory to double-check that they are correctly applied. also, use the built-in developer tools of your web browser to examine the headers for the requests, and check if they are showing up. it's always good to verify that what you expect is what you are actually getting. think of it like unit testing but for security headers. you will save yourself the head scratching and endless testing when you find that something is not working the way it should.

and here’s a bad joke to finish up: why did the security expert break up with the network engineer? because they had too many firewalls between them.

anyway, these measures are a start, not an end. security should be a continuous effort. keep learning, keep testing, and don't get complacent. i hope this gives you a good starting point. let me know if you have further questions.
