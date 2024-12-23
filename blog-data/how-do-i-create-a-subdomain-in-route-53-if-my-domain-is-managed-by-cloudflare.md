---
title: "How do I create a subdomain in Route 53 if my domain is managed by Cloudflare?"
date: "2024-12-23"
id: "how-do-i-create-a-subdomain-in-route-53-if-my-domain-is-managed-by-cloudflare"
---

,  I've been down this road more times than I can count, and dealing with DNS providers while juggling multiple hosting platforms can certainly feel a bit like a tightrope walk. You've got your domain managed in Cloudflare, and now you need a subdomain in route 53. It's a common scenario, and thankfully, entirely solvable without too much drama. It essentially comes down to delegation. I remember when we first migrated some infrastructure from an on-prem setup to AWS; the DNS dance took some getting used to.

The key thing to understand is that you don't want to completely transfer your domain from Cloudflare to Route 53. Instead, we're looking to delegate the subdomain to Route 53. This means that while Cloudflare remains the authoritative name server for your main domain, it'll forward queries for your specific subdomain to Route 53's name servers.

Let's break it down into steps, and I'll weave in some code examples to make it concrete.

**Step 1: Create a Hosted Zone in Route 53 for Your Subdomain**

First, within the aws console, navigate to Route 53. You’ll need to create a new hosted zone – this acts as your ‘container’ for the DNS records of the subdomain. Let's say your main domain is `example.com` managed by Cloudflare and you want your subdomain to be `api.example.com`. In the route 53 console, you’d create a new hosted zone with the domain name `api.example.com`.

Once the zone is created, route 53 provides you with a list of nameservers (NS) records. These are the authoritative servers for your newly created zone for the subdomain. You'll need to keep these safe for the next step. they'll look something like `ns-xxxx.awsdns-xx.com`, `ns-yyyy.awsdns-yy.net`, etc.

**Step 2: Configure Subdomain Delegation in Cloudflare**

Now, you hop over to your Cloudflare dashboard. Locate your `example.com` domain. What you're going to do here is create a specific NS record for your subdomain. This record tells the internet that if someone asks for records relating to `api.example.com`, they should be directed to route 53's nameservers. It's like a forwarding address.

Here's the basic idea of what the DNS record should look like using the Cloudflare web interface or API:
    * **Type:** NS
    * **Name:** api
    * **Content:** *Each of the nameservers listed in the route 53 hosted zone, each entered as its own DNS record*.
    * **TTL:** Automatic.

You need to create one NS record for *each* of the name server values provided by route 53. It's critical to get these values exactly correct. Any typo here and your subdomain resolution will fail, and you'll have a headache to troubleshoot. It's a good habit to double-check these settings before saving to prevent problems down the line.

**Code Example 1: Using AWS CLI to get Route 53 Nameservers**
I've always preferred command line interfaces for these tasks, it's quicker and easy to script. Assuming you have the aws cli configured, here's how to extract the nameserver records:

```bash
aws route53 get-hosted-zone --id /hostedzone/YOUR_HOSTED_ZONE_ID --query "DelegationSet.NameServers" --output text
```
(Replace `YOUR_HOSTED_ZONE_ID` with the actual ID provided when you created the `api.example.com` hosted zone.)

This command will output the four (or however many route 53 provided) nameserver records you'll be adding to Cloudflare.

**Step 3: Verify Propagation**

DNS changes aren't instantaneous. You'll need to allow for propagation, which is the process of these changes being distributed across the global DNS system. You can use online tools like `dig` (on Linux or Mac) or `nslookup` (available on Windows) to check if your changes have taken effect.

**Code Example 2: Verify DNS Propagation using `dig`**

```bash
dig ns api.example.com
```

The output will show you the nameservers currently answering for `api.example.com`. You're looking to see the nameservers that route 53 provided. If you still see Cloudflare nameservers, just wait a bit longer (usually within minutes or an hour) and check again.

Once your change has propagated, route 53 is now in charge of the DNS records for `api.example.com`. You can now begin creating `A`, `CNAME`, and other DNS records within route 53 for this subdomain.

**Step 4: Create DNS Records in Route 53**

Now that `api.example.com` is delegated to Route 53, you add records there, just like you would for a regular domain. For instance, you might want to point `api.example.com` to a server. Let’s say it’s an EC2 instance with the public IPv4 address `192.0.2.10`. In Route 53, you’d create an 'A' record that looks like this:

* **Record Name:** (Leave blank - applies to the apex record, meaning `api.example.com`)
* **Record Type:** A
* **Value:** 192.0.2.10

You can also create other record types like CNAME if you have load balancers, or AAAA if your servers are configured with IPv6. This is also where you handle advanced DNS configurations, such as weighted routing or latency-based routing if your application's architecture requires it.

**Code Example 3: Using AWS CLI to Create Route 53 Record**

```bash
aws route53 change-resource-record-sets \
    --hosted-zone-id YOUR_HOSTED_ZONE_ID \
    --change-batch file://change-batch.json
```
This command uses a json configuration file (change-batch.json) to define your record:

```json
{
    "Changes": [
        {
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "api.example.com",
                "Type": "A",
                "TTL": 300,
                "ResourceRecords": [
                  {
                     "Value": "192.0.2.10"
                   }
                 ]
             }
         }
     ]
}
```

(Remember to replace `YOUR_HOSTED_ZONE_ID` with the correct value. The TTL is set to 300 seconds, which is a reasonable starting point for most applications)

That’s the gist of it. It seems involved, but once you understand that delegation is the critical piece, the rest tends to fall into place. The main thing is to be accurate with the nameserver values. In my past, I've seen issues arising from typos in those values, leading to hours of debugging.

For further reading and to deepen your understanding on the topic, I would highly recommend checking out *DNS and BIND* by Paul Albitz and Cricket Liu, a classic, and also the official AWS documentation on Route 53 which is quite comprehensive and well maintained. For a more theoretical understanding of networking protocols, *Computer Networking: A Top-Down Approach* by James Kurose and Keith Ross is a solid choice. These resources offer a depth of knowledge that’s beneficial for understanding the underlying principles, which makes the actual implementation much clearer, and helps prevent future troubleshooting when things don't work exactly as expected.
