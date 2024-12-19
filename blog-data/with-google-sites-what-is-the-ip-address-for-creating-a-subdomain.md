---
title: "With Google Sites, what is the IP Address for creating a subdomain?"
date: "2024-12-15"
id: "with-google-sites-what-is-the-ip-address-for-creating-a-subdomain"
---

hey, so you're diving into google sites and subdomains, right? it's a bit of a twisty area, not as straightforward as setting up a regular website. let me break down what i’ve learned over the years of messing with this kind of stuff, specifically regarding the whole "ip address for a subdomain" thing with google sites, because that's not how it works, and that's probably where the confusion is coming from.

first off, and this is key, google sites doesn't really work with ip addresses for subdomains in the traditional sense. unlike having your own server where you'd point a subdomain to a specific ip address, google sites manages all the hosting and infrastructure for you. think of it as a highly managed environment where they take care of all the server details. when you create a site, it lives on their infrastructure, not on an ip address you can directly configure.

instead, what you'll be doing is using cnames (canonical name records) in your dns settings to point your subdomain to google's servers. it's a crucial difference. you’re essentially telling your domain provider "hey, when someone asks for 'subdomain.yourdomain.com', send them to google, where they will find the google site i created there."

let me walk you through how this played out for me way back when, i had this personal portfolio site running on google sites, i wanted a blog, and naturally, i wanted it at 'blog.my-portfolio.com'. it wasn’t my first rodeo with dns settings, but i did expect to see an ip address somewhere i could use, and that’s where i got stuck for a while. i was searching for an ip to punch into my domain settings, but all i was seeing was a bunch of google's domain names. it took me some hours, reading through google support pages and other resources, before i realized the whole dns mechanism around the whole thing.

so, you are probably seeing the same, right? no ip address to play with. instead, you need to set up a cname record in your dns zone file at your domain registrar. here's what that usually looks like in a simplified example, and this is how it typically manifests itself at most registrars:

```
type    name           value
cname    blog          ghs.googlehosted.com.
```

this assumes your host supports cnames. the 'type' is set to 'cname', 'blog' represents the subdomain, and the 'value' points to google’s servers via ‘ghs.googlehosted.com’. notice the final dot, that’s not a typo, and very important. it is a fully qualified domain name.

if you're wondering where that ‘ghs.googlehosted.com’ comes from, that’s google’s way of directing your traffic to their servers where your site is actually hosted. it's their infrastructure working under the hood, and you’re not directly interacting with it. think of it as a postal forwarding address rather than a street address.

now, after you do that, you might need to go back to google sites and point your site to your custom domain via the settings. google needs to know what address the user typed and which site you want to show at that address. here’s how you typically set this up inside of google sites itself:

1. open your google site.
2. click settings, usually a gear icon at the top right.
3. navigate to 'custom domains'.
4. click 'start setup'.
5. enter your subdomain, something like 'blog.yourdomain.com'.
6. google will guide you through verifying that you own that domain.
7. google might take a few hours to update and get everything connected.

the time it takes to propagate can vary depending on your domain registrar, so be patient. sometimes, i found that the settings update and it's instant, and sometimes it may take a while. you'll start to see the changes when you type the subdomain into your browser and your google site shows up.

a word of caution though. during this whole process, make sure you've correctly setup the cname record. if you make a mistake, the connection wont work as expected and sometimes diagnosing that can be a bit of a pain, so double check your spelling and that you included the ‘.’ after googlehosted.com. i had a friend that spent hours figuring out a simple typo in the value, just a missing '.'.

here's another example of what this could look like at your dns registrar, this time with a "www" subdomain, just in case you want that, again, no ip addresses here:

```
type    name        value
cname    www        ghs.googlehosted.com.
```

so far, we have discussed only cname, but there is more, the 'a' and 'aaaa' records. there is a different mechanism, when you point your naked domain (yourdomain.com) to google sites, you'll likely be working with something called 'a' records instead. these do point to an ip address, but not an ip address you configure. google provides these ip addresses, and you should follow google’s current recommendations because they may change them at any moment, and you will then need to update your dns settings. let's see how that looks:

```
type     name      value
a      @         216.239.32.21
a      @         216.239.34.21
a      @         216.239.36.21
a      @         216.239.38.21
```

these 'a' records for your root domain tell your registrar to send users requesting your main domain to one of google's servers. the '@' symbol usually represents your root domain. sometimes you also will have 'aaaa' records as well, these are for the ipv6 protocol, and follow a similar principle. again, these values must be what google suggests at the time of your configuration.

so, when it comes to subdomains and google sites, ditch the hunt for an ip address. that’s not the way google sites rolls. think cnames, think pointing your subdomain to google, and think google hosted services handling the rest.

i've noticed that sometimes, when you get stuck with these problems, the solution is not always obvious at first. there are some good resources out there, like “dns and bind” by paul albitz and cricket liu. that's where i got a lot of my deeper dns understanding. it dives into how dns works at a low level. the book is a bit older, but the basics remain relevant. also, check out the official dns rfc (request for comments) documents, specifically rfc 1034 and rfc 1035. i’m not saying go read every single word, but at least check the main concepts and some sections as they provide an authoritative explanation of dns concepts. i used to skim the official standards to get some clarity.

one last important point, this isn’t a perfect world, and sometimes, your browser or your local computer can store old dns settings, so when you are testing if things are working, be sure to clear your browser cache or use a different browser. also, you might want to use online tools like dns propagation checkers to see if your changes are visible to the world.

and for a bit of humor. why did the programmer quit his job? because he didn't get arrays.

i hope that clarifies how subdomains and google sites work. it's a bit quirky, but once you get the hang of cnames and google hosted, it gets way easier. keep your cname records in check and don't lose your head looking for an ip address. remember that's not how it's done with google sites.
