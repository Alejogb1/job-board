---
title: "Why are DNS records missing with custom domain SSL?"
date: "2024-12-16"
id: "why-are-dns-records-missing-with-custom-domain-ssl"
---

Okay, let's tackle this. It's a problem I’ve certainly seen crop up a few times over the years, and it's often a cascade of seemingly disparate issues converging at once. The scenario you’re describing – missing DNS records when setting up SSL with a custom domain – isn't usually a straightforward case of one single point of failure. Instead, it tends to be a multifaceted problem involving the interplay between your domain registrar, your DNS provider, and your SSL certificate authority, sometimes with a reverse proxy thrown into the mix for good measure. Let’s break down the typical culprits and how to approach them.

First, let's acknowledge that the core issue boils down to miscommunication. When you're dealing with a custom domain and want to secure it with SSL, a specific set of DNS records needs to be in place for everything to play nicely. Primarily, you need an *a record*, or perhaps an *aaaa record* if you're using IPv6, to point your domain or subdomain to the correct IP address of your web server. But, for SSL, things get a little more intricate. The critical record that's often overlooked is the *caa record*. This *certificate authority authorization* record dictates which certificate authorities are permitted to issue SSL certificates for your domain. If that record is missing, incorrectly configured, or conflicts with the CA you're using, the certificate validation process may fail. And if the cert can’t validate, it won’t be issued correctly, which means your site won't be accessible via https.

From my experience, the most common scenario is that the caa record is either entirely missing or was not updated to include the specific certificate authority being used to generate the ssl cert. Think back to a project where I was migrating a web application to a new hosting provider. We transitioned the *a records* and everything seemed fine. But when we tried to obtain an ssl certificate for the new server, it kept failing silently. It took a while but eventually we realized that when the domain was originally set up the *caa record* only authorized a specific CA. The new provider used a different CA and because the *caa record* wasn't updated it prevented the certificate validation. A real head scratcher.

Here's a code snippet that demonstrates how a correctly formatted *caa record* might look in a bind-style zone file:

```
example.com. IN CAA 0 issue "letsencrypt.org"
example.com. IN CAA 0 issuewild "letsencrypt.org"
example.com. IN CAA 0 iodef "mailto:security@example.com"
```

This example allows *letsencrypt.org* to issue certificates for `example.com` and any subdomains (the `issuewild` directive). It also defines an email address for violation reporting, using the `iodef` (incident object description exchange format). Now if you were to use another certificate authority besides let's encrypt, or, were using a wildcard and wanted to use letsencrypt you would need to make sure the record was updated.

Let’s say you're hosting your application on a cloud provider like Amazon Web Services (AWS), which often involves using their load balancers or CloudFront. In such situations, you might not point the *a records* directly to your server's ip. Instead, they point to the load balancer's endpoint or CloudFront’s distribution. This indirection can sometimes create further confusion in the troubleshooting process. Your DNS records need to point to the *correct resource*, and that might not be the static IP address you'd traditionally expect, but rather a dynamic hostname or alias.

Another common issue I've observed stems from incorrect propagation. Changes to DNS records aren't instantaneous; they take time to propagate across the global dns network, and this can result in inconsistencies. When troubleshooting, always wait for adequate propagation before making further changes or assuming a problem exists. DNS propagation checks should be your friend during times of change. There are numerous free online tools for verifying DNS records. Tools like `dig` are your local resource for querying dns records.

Here's a basic command-line example of how you might query for *caa records* using `dig`:

```bash
dig example.com caa
```

The output will show the current *caa records* associated with your domain, giving you a quick snapshot of what's configured. If you see the records are missing or have incorrect entries, it is a direct place to focus on. It’s important to check the output of the dig command carefully to ensure the correct certificate authority is listed, especially if you have recently changed certificate providers.

Sometimes, it's not just the records themselves that are the issue, but how they are being managed. Especially when relying on cloud providers, the interface they provide can add layers of abstraction that make understanding what's being stored within the DNS system. Ensure your DNS configuration is not being overwritten by a separate service or automation system. If you're relying on infrastructure-as-code tools like terraform or cloudformation, you need to verify that your dns resource configurations are setup correctly. It's not unusual for a seemingly minor syntax error in your IaC configurations to result in missing or misconfigured DNS records.

Finally, let's not forget about the basics. Domain misconfiguration is a common cause; make absolutely certain you are modifying the DNS records for the correct domain name. It sounds obvious, but I’ve seen people spending hours troubleshooting issues, only to realize they were working on the wrong domain, especially with multiple domains and subdomains involved. I myself have spent far to long scratching my head only to realize I was modifying a record on the wrong domain.

Here is an example of how you might define dns records in terraform using the `cloudflare_record` resource

```terraform
resource "cloudflare_record" "example_caa" {
  zone_id = var.cloudflare_zone_id
  name    = "example.com"
  type    = "CAA"
  value = "0 issue \"letsencrypt.org\""
  priority = 0
}


resource "cloudflare_record" "example_a" {
  zone_id = var.cloudflare_zone_id
  name    = "example.com"
  type    = "A"
  value = "192.168.1.100" # Replace with your servers IP
  priority = 10
}

resource "cloudflare_record" "example_www_a" {
  zone_id = var.cloudflare_zone_id
  name    = "www"
  type    = "A"
  value = "192.168.1.100" # Replace with your servers IP
  priority = 10
}
```

The provided code snippets show three different records being defined in terraform for cloudflare. The first record creates the *caa record* to authorize let's encrypt to issue certificates for example.com. The next record shows how to define the *a record* for the root domain, and then for the www subdomain. Note that value should be your actual server IP. This is just an example of how you can define such records.

To solidify your understanding, I recommend looking into RFC 6844, which specifies the *caa record*, and dnssec which helps provide validation of dns entries. Also, "dns and bind" by Paul Albitz and Cricket Liu is a timeless and excellent resource for diving deeper into the technical details of dns. In short, missing dns records with custom domain ssl is often a combination of multiple small configuration issues which require a systematic approach to troubleshoot. Start with your *caa record* and then dig into the rest of the dns config. Take a step back and double check to make sure that all configuration changes are being made on the correct domain.
