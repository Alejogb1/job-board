---
title: "How to remove a custom domain from my github repository?"
date: "2024-12-15"
id: "how-to-remove-a-custom-domain-from-my-github-repository"
---

so, you want to detach a custom domain from your github repository. i've been there, done that, got the t-shirt and probably a few scars from similar situations. it’s actually a pretty common scenario, especially when you're juggling multiple projects or switching providers. i remember back in '09, i had this whole personal blog thing going on, hosted on github pages, with a fancy custom domain i'd painstakingly set up. well, that blog turned into a portfolio site, and the old domain just didn't fit anymore. i went through the process you’re facing, and it wasn't entirely smooth sailing.

let's break down the steps, and i’ll throw in some of the quirks and gotchas i've stumbled upon along the way.

first off, there isn't a single magic button in github that just says "detach domain." it's more about dismantling the configuration pieces. think of it like unhooking a series of pipes – you need to trace back where the domain is connected and disconnect each point.

the main thing to tackle first is the domain’s configuration at the domain registrar. that’s where you initially pointed your domain name to github pages. i'm assuming you have access to your domain registrar control panel, whether that's godaddy, namecheap, google domains, or whatever.

within your registrar’s settings you'll need to find the dns configuration. you're looking for records like `a records` or `cname records`. the specific record pointing to github will vary but, in general you'll be looking for something like this:

    type: a
    name: @ (or your domain name, like `example.com`)
    value: 185.199.108.153 (this is one of github's ips, might be another one)

or this:

    type: cname
    name: www (or your subdomain, like `www.example.com`)
    value: yourgithubusername.github.io

these records basically tell the internet "hey, when someone asks for `example.com` or `www.example.com`, go look at this specific github server." so, to remove the connection, you'll need to remove these entries.

here is the code snippet example in a `bind` zone file:

```
; zone file for example.com

$ttl 3600
@   IN  SOA ns1.yourdomain.com. hostmaster.yourdomain.com. (
                2023102701 ; serial
                3600       ; refresh
                1200       ; retry
                1209600    ; expire
                3600       ; minimum
)

@   IN  NS  ns1.yourdomain.com.
@   IN  NS  ns2.yourdomain.com.

; remove the following record to detach your github pages configuration:
; @    IN A     185.199.108.153

www    IN CNAME yourgithubusername.github.io.
```

once you've deleted those records at the registrar, the domain will eventually stop pointing to your github pages site. this change isn't instant. dns propagation takes some time (usually minutes, could even take up to 48 hours in rare cases), so you might see a brief window where the domain works intermittently.

that's the registrar side of things. now, let's talk about the github side.

next you need to head over to your github repository’s settings. navigate to the "pages" section. here you’ll see the custom domain setting. it will likely show the domain you want to remove. you'll see a textbox there, this is where you added your domain previously. you can clear the content of that box and press the save button. if the domain is not present, that is a sign you already removed it from github. you’re basically telling github "hey, i no longer want to use this domain for this repository."

after this github process, the github side of the disconnection is done too, you are now unhooking your github configuration from your domain. this is important, even if you remove it from your registrar records and this one is not done, github will still have the configuration stored.

now, keep in mind something i learned the hard way: if you’re planning on re-using the domain on github pages for another repository, github might still be thinking about the old setup. it's an odd behavior i encountered a few times in my early days, where i moved a domain from one repo to another and it was still pointing to the old one. you can fix this by just waiting and allowing the previous settings to propagate in the dns servers around the globe, or using other services as mentioned in the resources below.

here's a code snippet of a python script that you can use to test your dns, it uses the `dnspython` library, which you might need to install before running it using `pip install dnspython`:

```python
import dns.resolver

def check_dns_record(domain_name):
    try:
        resolver = dns.resolver.Resolver()
        answers = resolver.resolve(domain_name, 'A')
        print(f"a records for {domain_name}:")
        for rdata in answers:
            print(rdata)
    except dns.resolver.NoAnswer:
        print(f"no a record found for {domain_name}")
    except dns.resolver.NXDOMAIN:
        print(f"domain {domain_name} not found")

    try:
        resolver = dns.resolver.Resolver()
        answers = resolver.resolve(domain_name, 'cname')
        print(f"cname records for {domain_name}:")
        for rdata in answers:
            print(rdata)
    except dns.resolver.NoAnswer:
        print(f"no cname record found for {domain_name}")
    except dns.resolver.NXDOMAIN:
        print(f"domain {domain_name} not found")

if __name__ == "__main__":
    domain_to_check = input("enter the domain to check: ")
    check_dns_record(domain_to_check)
```

run this script, and type your domain when asked, it will print the current configuration that your domain has. it might show you an old configuration that needs to expire, allowing you to monitor the propagation.

one more point from my experience, the `www` subdomain might be set up as a `cname` record pointing to your `yourgithubusername.github.io`. if that's the case, you’ll want to remove that cname record as well in your dns settings in your registrar.

as for resources, i’d really suggest checking out "dns and bind" by paul albitz and cricket liu. it's an old book, but the core concepts haven't changed and it explains the dns fundamentals well. for a more modern take, "cloudflare dns" documentation is great, they're one of the fastest dns providers, and they explain the concepts of dns very well.

finally, a little bit of a joke, why did the dns server go to therapy? it had too many unresolved issues. :)

this should be it. after this process you are removing your domain from your github repository. if there are any remaining issues, it is usually a matter of waiting and allowing dns propagation to do its thing or perhaps double checking that the dns records were removed correctly from both the domain registrar and the github pages settings. remember, removing a custom domain from a github repo isn't rocket science, it is mostly about understanding the configuration pieces and where they live.
