---
title: "How can I setup private nameservers with an Amazon LightSail Instance with WHM / cPanel?"
date: "2024-12-15"
id: "how-can-i-setup-private-nameservers-with-an-amazon-lightsail-instance-with-whm--cpanel"
---

alright, let's talk about setting up private nameservers on a lightsail instance running whm/cpanel. i've been down this road a few times, and it's definitely something that needs careful steps, and it's not always intuitive, especially with all the moving parts involved.

so, the core idea here is that you want your own domain’s nameservers (like ns1.yourdomain.com and ns2.yourdomain.com) to resolve *directly* to your lightsail server's ip, instead of relying on your registrar’s dns service, or something like cloudflare. basically you become your own dns authority for your domain and anything using your nameservers. this gives you a lot more control, but it also means more setup and potential headaches if not done well. i remember my first time trying this, back when i was mostly using bare metal servers, and had to learn a *lot* about bind. the process is a bit similar even if we are using lightsail.

first, let's deal with the obvious: your lightsail instance needs a static ip. you cannot be changing ips or else you will have to update your dns records in every instance, and that would be a bad practice. you will have to associate your instance with a static ip in lightsail console. once you've done that, grab that static ip because you will need it in the next steps. keep it handy.

now, regarding whm/cpanel, thankfully, it handles the core dns server bit, that's bind, more or less transparently. but, you will have to do a bit of manual configuration on both cpanel and your domain registrar so here is what you have to focus on, mostly the registrar side.

you're going to need to create "glue records" at your registrar. these aren't dns records like a or cname you're probably used to. instead, these records associate your nameserver hostnames (like `ns1.yourdomain.com`) with the static ip of your lightsail instance. let me show you a small example. let’s assume your domain is `yourdomain.com` and your static ip is `192.0.2.100` and `192.0.2.101`. you should create two glue records:

*   `ns1.yourdomain.com` pointing to `192.0.2.100`
*   `ns2.yourdomain.com` pointing to `192.0.2.101`

the exact interface for doing this varies wildly depending on your registrar, but most will have a spot to manage glue records or sometimes they are called “child nameservers” or “hostnames”. note you need to do this for *each* nameserver. without it, your nameservers will not resolve to anything, and well, no one will find your website. it's like having a phone number but no telephone exchange, it just won't work. i've spent hours troubleshooting issues before because i forgot about these. never again. and now, we are ready to configure the cpanel side of things.

on the whm side of things:

1.  login to whm as root.
2.  navigate to "dns functions" > "edit dns zone".
3.  select your domain `yourdomain.com`
4.  add “a” records for each of your nameservers pointing to the static ip of your lightsail instance. in our example case:

    `ns1  a  192.0.2.100`
    `ns2  a  192.0.2.101`
5.  if you have a second ip address on your light sail instance add the second `ns2` to that ip address. if not add it to the same address.
6.  save the dns zone.

once this is done, go to "basic cpanel & whm setup" in whm, and configure "nameservers" section to use your newly created nameservers `ns1.yourdomain.com` and `ns2.yourdomain.com`. this is important, without it, cpanel won’t *tell* bind to start serving dns for your domain using these nameservers.

now, this might sound straight forward but here is the thing, dns propagation is not instant and this is important. it may take several hours (or even up to 48 hours) for the changes to propagate globally. you can use online dns propagation checkers, like `whatsmydns.net`, to see if your new nameservers are resolving correctly from different locations around the globe. patience is key here. i once panicked because my site was inaccessible immediately after changing nameservers, but it was just propagation lag. learned my lesson there.

one common error when doing this is when you use non-standard ports to access cpanel/whm. remember that you may need to adjust firewall rules (both in lightsail and maybe on your server itself, using for example `iptables` or `ufw`) to allow dns traffic (port 53) on both tcp and udp, but also you may want to check the port you have configured for whm and cpanel, and ensure you are allowing the new ports on the instance and also in whm/cpanel settings. this is a common issue, specially when someone configures whm to use a non standard port. i’ve done this and had to spend some time debugging it.

now, to be absolutely sure you have correctly set it up, you can use `dig` command (or `nslookup` if you prefer) to query your domain using your nameservers directly, before switching all your domains to them. for example, use:

```bash
dig @ns1.yourdomain.com yourdomain.com
```

if that command returns the correct ip address then you are golden, otherwise something is not resolving correctly, go back and double check each step. there is also, a great tool on whm, `named configuration`, you can see if there are any errors with your dns configurations. i recommend you take a look if you have any problems.

now, here's a bit of a gotcha that tripped me up once. if you plan to use your private nameservers for other domains hosted on that lightsail instance, you'll need to repeat the ‘a’ record step for every single domain inside whm, so you need to add these records on the zone configuration of every single one, you don't need to do anything on your domain registrar side, you only need to do it once, for the private nameservers of `yourdomain.com`. this caught me out when i was migrating a bunch of sites. i mean, i guess it makes sense, but it's easy to overlook, and i hate repeating myself.

here is an example of an actual dns zone configuration in bind zone format, this should give you an idea on the format you should see in the cpanel zone editor:

```
$ttl 86400
@   IN  SOA ns1.yourdomain.com. admin.yourdomain.com. (
    2023102701 ;serial
    7200 ;refresh
    3600 ;retry
    1209600 ;expire
    86400 ;minimum
)

@       IN    NS    ns1.yourdomain.com.
@       IN    NS    ns2.yourdomain.com.
@       IN    A    192.0.2.100
www  IN A   192.0.2.100
ns1  IN A  192.0.2.100
ns2  IN A  192.0.2.101
mail  IN A  192.0.2.100
```

and another snippet, an example using the same format, but with a different domain which is hosted on the same server:

```
$ttl 86400
@   IN  SOA ns1.yourdomain.com. admin.anotherdomain.com. (
    2023102701 ;serial
    7200 ;refresh
    3600 ;retry
    1209600 ;expire
    86400 ;minimum
)

@       IN    NS    ns1.yourdomain.com.
@       IN    NS    ns2.yourdomain.com.
@       IN    A    192.0.2.100
www  IN A   192.0.2.100
mail  IN A  192.0.2.100
```

notice in the second example that the nameservers used are still the ones from the `yourdomain.com` even if it’s another domain. now here’s a slightly more complex example, where we add also a subdomain:

```
$ttl 86400
@   IN  SOA ns1.yourdomain.com. admin.yourdomain.com. (
    2023102701 ;serial
    7200 ;refresh
    3600 ;retry
    1209600 ;expire
    86400 ;minimum
)

@       IN    NS    ns1.yourdomain.com.
@       IN    NS    ns2.yourdomain.com.
@       IN    A    192.0.2.100
www  IN A   192.0.2.100
ns1  IN A  192.0.2.100
ns2  IN A  192.0.2.101
mail  IN A  192.0.2.100
subdomain IN A 192.0.2.100
```

that’s it, the important thing to understand here is that once you have successfully setup the first domain name with the private nameservers, the rest of the process is to configure the nameservers to point to the initial domain.

one thing that might be a good thing to consider, and not related to this problem, but useful when you are dealing with dns is that dns records usually come with a ttl or time to live value. this value tells how long a record should be cached, and if you are making changes you may want to lower the ttl to speed up propagation, after everything is working you can set it to a higher value.

for further details i recommend taking a look at rfc1035, rfc1034, which describe the dns protocol itself and how it works. and also the dns and bind book by paul albitz, which is considered an authority on the subject of dns and bind servers, it is very useful if you are planning to get deep into dns.

and finally, if you see your website suddenly start working in one country, but not another, dont panic, probably you are still having some propagation issues. go grab a cup of coffee, wait a bit longer and it should be fine. i mean, you wouldn’t ask a snail to be a race car, *would you?*

in short, setting up private nameservers with lightsail and cpanel is all about glue records and consistent configuration. make sure to give time for propagation, or your site may experience some weird downtime issues. good luck, you will make it.
