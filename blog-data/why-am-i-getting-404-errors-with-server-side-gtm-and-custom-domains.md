---
title: "Why am I getting 404 errors with Server-side GTM and custom domains?"
date: "2024-12-16"
id: "why-am-i-getting-404-errors-with-server-side-gtm-and-custom-domains"
---

Okay, let's unpack this 404 situation with server-side google tag manager (sgtm) and custom domains. It's a surprisingly common hiccup, and over the years, I’ve debugged this scenario more times than I care to count. In my experience, these 404s aren't usually about the server actually missing files, but rather a misconfiguration that creates an impedance mismatch between your custom domain and the sGTM server container. Let's break it down into the key areas, focusing on practical troubleshooting and configurations I’ve found helpful.

The core issue often lies in how your custom domain's dns records, your reverse proxy configuration (if you have one), and your sGTM server container's settings interact. Remember, server-side tagging isn't just about firing tags server-side; it's also about correctly handling requests routed through your domain. Think of it as a pipeline where each component needs to be precisely aligned. The most frequent offenders, from what I've seen, tend to fall into these three categories:

1. **dns and domain misconfiguration:** The very first step is always verifying that your dns records are set up perfectly. When you use a custom domain for your sGTM server container, you're essentially telling the world to direct traffic intended for that domain to the specific server where your container is running. If the dns a-record or cname isn't pointed correctly, the browser won’t be able to resolve your domain to the IP of the machine hosting the sgtm container, or the reverse proxy in front of it. If this happens, it will result in a 404.

    It’s not always as simple as pointing an `a` record to an ip. Sometimes, you’re working with load balancers, which means you need to ensure the domain is properly configured in the load balancer’s configurations, and not just directly at the dns record level. I remember working on a complex setup involving an aws elastic load balancer, where the dns entries were perfect, but the load balancer itself wasn't routing the traffic to the correct instance, causing 404s.

2. **server-side container url configuration within gtm:** this is where many of the gotchas happen. Within the sGTM container’s interface, you specify the “container url”. This url must exactly match the custom domain you're using (and also the `x-forwarded-host` headers, as explained later). I’ve found that a minor variation, even adding an extra forward slash at the end of a domain, can result in 404 errors. GTM, at its heart, is pretty literal. It's expecting the incoming requests to match the defined url perfectly. I’ve spent many frustrating hours tracking these differences down, sometimes just a missing "www". Subdomain issues fall into this category too, make sure there are no missing subdomains like `www.` if your setup requires it.

3. **reverse proxy headers and routing problems:** if you’re using a reverse proxy (like nginx or apache) in front of your sgtm server, you'll need to configure it to correctly pass through the necessary headers, especially the `x-forwarded-host` header. When a request comes to a reverse proxy, the original host header is often overwritten or not forwarded correctly. The sGTM server relies heavily on this header to determine the container it should use for processing the request. If that header is missing, incorrect, or contains the load balancer’s host instead of the actual custom domain, sGTM will likely reject the request with a 404. It's imperative to properly manage these headers. I saw a case involving a client using cloudflare where cloudflare wasn’t properly forwarding these headers, resulting in a frustrating cascade of 404 errors.

Let’s look at some code examples to make these points clearer.

**example 1: nginx configuration for correct header forwarding**

Let's imagine you are using nginx as a reverse proxy. Here's a typical configuration snippet that addresses the `x-forwarded-host` and routing issues:

```nginx
server {
    listen 80;
    server_name your-custom-domain.com;

    location / {
      proxy_pass http://your-gtm-server-ip:8080; # or the internal url
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_set_header X-Forwarded-Host $host;  # This is crucial for sgtm
      
      
      #optional if you are using websockets
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
    }

}
```

in this example, `proxy_set_header X-Forwarded-Host $host;` is absolutely vital. It makes sure the original host that was used to make the request is correctly passed to the sGTM server. Without this line, sGTM is almost guaranteed to have problems, resulting in 404s. Additionally, you can see that the `proxy_pass` line is pointing to the backend server where the sgtm container is hosted. This needs to match the actual backend url. Notice the use of `$scheme` in `X-Forwarded-Proto` which captures the `http` or `https` to ensure this information is also forwarded.

**example 2: dns verification (using command line tools)**

Sometimes it's just good to double check your dns setup. You can use command-line tools such as `dig` or `nslookup` to query dns records. On linux or macos, you can do this:

```bash
dig your-custom-domain.com a

# or to see the full chain if using cname
dig your-custom-domain.com cname +trace
```

this command will show the resolved ip address for your domain, or the chain of cname records. In the output, you should see your domain’s `a` record pointing to the ip of the server or load balancer you expect. If it’s not, that’s your problem. On windows, you could use `nslookup`. This command checks if the dns resolution is happening as expected. Compare the output from these tools with your dns settings on your domain provider's website.

**example 3: sgtm server container url check (within the GTM interface)**

This one is not code, but shows a critical verification step. Go to your gtm interface. Make sure, within the "admin" section of your server container, the defined "container url" exactly matches the `server_name` specified in your nginx config and also the actual custom domain you intend to use. Double check for any typos or discrepancies and include `https://` if needed.

Remember, in my experience, even seemingly insignificant variations cause problems. If your server is running behind an https endpoint, then you must use the https protocol in the sgtm config. Also, ensure that all of your traffic is actually routed through the custom domain url, because if traffic is bypassing it, 404s can be a symptom.

To further improve your understanding, i would suggest exploring the following resources: “high performance browser networking” by ilya grigorik, which contains extremely detailed information about all these concepts regarding networking and dns. Also, nginx's official documentation is indispensable if you use nginx as a reverse proxy. Understanding concepts like `proxy_set_header` are key to solving these issues.

These are just a few examples, but i have found these three general areas and the solutions to be extremely effective at dealing with 404s. The key is systematic troubleshooting, double checking the dns configuration, your container url, and especially your reverse proxy header configurations. The issues, more often than not, are located in one of those areas. By focusing on these areas, you should be able to diagnose and resolve the 404 errors and get your server-side gtm setup running smoothly.
