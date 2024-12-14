---
title: "Do I use the DNS Target or the heroku app domain for my CNAME Data?"
date: "2024-12-14"
id: "do-i-use-the-dns-target-or-the-heroku-app-domain-for-my-cname-data"
---

alright, let's talk cname records and heroku. i've been down this road a few times, and it can be a bit confusing, particularly when you're juggling cloud providers and dns settings. so, i'll try to keep it straightforward.

the core issue revolves around where you're pointing your cname record. it comes down to whether you should target the specific heroku application’s domain (*your-app*.herokuapp.com) or the general heroku dns target, which is usually something like *something*.herokudns.com. let me break it down:

**the heroku app domain (*your-app*.herokuapp.com)**

pointing a cname directly at *your-app*.herokuapp.com is the seemingly obvious route, and it might work for some simple setups. the heroku app domain points to a very specific server or cluster of servers.

**why it's usually not the best approach:**

*   **ip address changes:** heroku's infrastructure is dynamic. if your app gets moved to a new server (and that happens all the time behind the scenes), the ip address associated with *your-app*.herokuapp.com can change. if this happens, your cname will point to a defunct ip, and your site will stop working until your dns configuration is fixed. it means a manual update which is very uncool.
*   **scalability concerns:** heroku can scale your app across multiple servers. a cname directly to the app domain doesn't account for this and is not dynamic.

**the heroku dns target (*something*.herokudns.com)**

heroku provides a special dns target to avoid the problems i mentioned. this target, something like *something*.herokudns.com, is an intermediary. it’s always kept up to date with where your app is actually located. think of it as a load balancer.

**why it's the best approach:**

*   **dynamic resolution:** the herokudns target automatically resolves to the correct ip address(es) for your app, even when the backend infrastructure shifts. you dont have to worry about changes.
*   **scalability handled:** it’s designed to work with heroku's dynamic scaling and multiple servers. your cname will always point to a live app, no manual change.
*   **reliability:** using the herokudns target is the standard best practice, it prevents downtime due to backend changes.

**in short:** always use the heroku dns target, not the heroku app domain directly.

**my personal journey with this (the hard way)**

i remember this one time when i was setting up a custom domain for a side project. i was trying to be clever, and like many, i figured pointing the cname straight to the app’s domain was good enough. it worked for a few weeks, and i happily forgot about it. then, one day, the site went completely offline. it turned out heroku had migrated my app, and my cname was now pointing to a blackhole. a long debugging session and some panicked googling made me learn about the herokudns target the hard way. since then, i’ve stuck with the recommended method. i hope that can be useful for you. and not just a fun anecdote.

**code snippets to illustrate (and demonstrate some dns lookups)**

here are some snippets that show how to look up the heroku dns target for an app and how a cname record might look in a dns zone file. note that the `dig` command is a network utility, if you are using windows use `nslookup`.

**1. using `dig` to look up the herokudns target**

```bash
dig +short <your-app>.herokuapp.com cname
```

this command will reveal the herokudns target that you should use. the output may look like something like this:

```
<random-chars>.herokudns.com.
```

**2. a sample dns zone file showing the cname record**

```
; cname for www.example.com pointing to the heroku dns target.
www             IN      CNAME   <random-chars>.herokudns.com.

; optionally, a cname for example.com to www
example.com     IN      CNAME   www.example.com.
```

**3. checking the resulting records using dig (or nslookup)**

once the dns changes propagate, we can verify that the records resolve correctly.

```bash
dig +short www.example.com
dig +short example.com
```

these commands should show the corresponding herokudns target.

**important points to note:**

*   **dns propagation:** dns changes take time to propagate across the internet. you may need to wait a few minutes to a few hours after making changes.
*   **domain registrar setup:** the exact method for adding cname records will vary between domain registrars. usually, it's a simple form where you enter the subdomain/name and the target (the herokudns domain, not the app domain).
*   **root domain cnames:** you can't usually point the root domain (e.g., example.com) directly to a cname, usually you need to point it to a `www` subdomain and configure a cname redirect to work. some services like cloudflare offers `cname flattening` to solve this, but that's another topic.
*   **https/ssl certificates:** you'll probably want an https certificate for your custom domain. heroku offers a managed ssl certificate feature that can be setup to work with custom domains. this involves adding your custom domain to your app and then enabling ssl. check the heroku documentation for the most up-to-date process.
*   **heroku documentation:** always check the latest heroku docs. they are usually spot on and provide the most accurate instructions, it has been my go to place when i hit a wall.

**recommended resources**

*   **"dns and bind" by paul albitz and cricket liu:** a classic in the field, providing a really complete and technical deep dive into dns. a very good read if you want to really get into the specifics.
*   **heroku documentation:** heroku's official documentation is always a good place to start, they usually keep it up to date with the latest best practices.
*   **rfc 1034 and rfc 1035:** these are the original specifications for dns. they are very technical, but useful if you need to look for definitions and very low level details.
*   **"understanding linux network internals" by christian benvenuti:** this book is focused on linux, but the networking fundamentals that talks about are relevant for any operating system, including dns.

that's about it. use the herokudns target for your cname. it’s the right way, the secure way, and the one that will save you a ton of headaches down the line. you are now slightly less likely to end up staring at your browser trying to figure out why your web app is not online.
