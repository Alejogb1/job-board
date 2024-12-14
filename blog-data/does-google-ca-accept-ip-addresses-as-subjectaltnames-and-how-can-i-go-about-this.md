---
title: "Does Google CA accept IP addresses as SubjectAltNames and how can I go about this?"
date: "2024-12-14"
id: "does-google-ca-accept-ip-addresses-as-subjectaltnames-and-how-can-i-go-about-this"
---

alright, so you’re asking about google ca and using ip addresses in subject alternative names, eh? i've been around the block a few times with certificates, and this particular issue is one that's popped up quite a bit in my career.

first off, the short answer is, yeah, google certificate authority *does* accept ip addresses as subject alternative names (sans). it's a pretty standard thing in x.509 certificates nowadays, especially if you're dealing with internal services or devices that might not have a fqdn. but, and it's a big but, there are some gotchas you need to watch out for.

now, let me tell you a bit about where i first ran into this problem. it was back in 2015 i think, i was working on this large iot project - think thousands of sensors scattered around, each communicating back to central servers. at the time, we were trying to set up secure communications with mutual tls. we initially went down the route of only using domain names, but we soon realized that many devices didn’t have a consistent resolvable hostname, especially when they were deployed in varied network environments, we had to resort to ip addresses.

the first problem we faced was certificate generation. we tried initially with a free certificate tool and that didn't do the trick, it just flat-out rejected adding ip address sans. so, after a bit of hair-pulling, i did a deep dive into x.509 specs. this is where i learned the hard way that not all certificate generation tools are created equal when handling ip sans, some tools will just not allow that or will reject a certificate request if it contains them.

we then transitioned to openssl to be more flexible, and started crafting our csr requests manually. the first approach we had looked something like this (i’m paraphrasing from my notes, so it might not be exactly the same):

```bash
openssl req -new -newkey rsa:2048 -nodes -keyout my.key -out my.csr \
    -subj "/CN=my.internal.server.com" \
    -addext "subjectAltName=IP:192.168.1.100,IP:10.0.0.5"
```

this gave us the csr, but when we submitted it to the google ca, we got a certificate that didn't actually include the ip addresses in the san field. that was a major head-scratcher. turns out, the `-addext` flag, when used like that, does not automatically add subject alternative names correctly for many ca, the format is not correct. we had to use a config file instead.

after some trial and error we nailed it, using this setup with a configuration file:

```bash
# my.cnf
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]
CN = my.internal.server.com

[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
IP.1 = 192.168.1.100
IP.2 = 10.0.0.5
```

and then the request command becomes

```bash
openssl req -new -newkey rsa:2048 -nodes -keyout my.key -out my.csr -config my.cnf
```

that generated the csr with the ip sans correctly formatted.  finally, the google ca issued us a certificate that worked as intended. so the key takeaway here is: use a configuration file instead of the command line, it makes things more flexible and less error prone.

another thing i learned is that different ca’s have different validation policies, this is something to consider also. it’s a bit funny, but i remember an old colleague once jokingly said he thought ca’s used to check the ip addresses by actually pinging them. which is obviously not true. but i digress. google ca, from what i’ve seen, doesn't do anything crazy, they just check that the request is formatted correctly. however, remember that the ips that you put inside the san needs to be owned by you.

so when submitting to a ca for a certificate, ensure you check what information they need, and if you need to add specific parameters or fields inside the request. for example, some ca may require that you add other specific field in the request, such as specific extensions.

now, what if you’re using a program to manage all of this certificate stuff? well, you'll want to make sure your library or tool is handling ip sans correctly. i personally use python often for these tasks, if i’m doing some automation. for example i used to use the cryptography module for generating certificates. here’s a python snippet of a working example of a csr with ip sans using that module, just for demonstration purposes (i will remove some of the error handling so it looks more concise):

```python
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509 import IPAddress

def create_csr_with_ip_sans(common_name, ip_addresses):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    builder = x509.CertificateSigningRequestBuilder()
    builder = builder.subject_name(x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ]))

    san = []
    for ip in ip_addresses:
        san.append(IPAddress(ipaddress.ip_address(ip)))
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san), critical=False
    )

    request = builder.sign(key, hashes.SHA256())

    csr_pem = request.public_bytes(encoding=serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return csr_pem, key_pem

if __name__ == '__main__':
    ip_sans = ["192.168.1.100", "10.0.0.5"]
    csr, key = create_csr_with_ip_sans("my.internal.server.com", ip_sans)
    print("-----csr------")
    print(csr.decode('utf-8'))
    print("-----key-----")
    print(key.decode('utf-8'))
```
the code example above generates a csr and a private key with a defined list of ip sans. you would then need to take the csr and send it to the ca to obtain your certificates.

a few things to keep in mind when working with google ca. first, their pricing model, it can get expensive, especially if you are issuing tons of certificates, so if you’re doing this at a large scale, keep an eye on that. second, make sure that your environment is configured properly to obtain the certificates automatically. we’ve seen too often people using manual processes which is not ideal in real life. and lastly, always always, check your certificates. the easiest way is to use `openssl x509 -text -noout -in my.crt`, and verify that everything you requested, such as subject alternative names, are actually there.

for further reading, i would recommend having a look into the x.509 standard itself, for example the rfc 5280. also, a good book would be "bulletproof ssl and tls" by ivan ristevski, it provides a comprehensive overview of this subject. also, the openssl documentation itself is a really good source of information about certificates.

to sum it up, google ca accepts ip addresses in the subject alternative names, but you need to pay attention to how your csr is generated and make sure the tools that you use are handling it properly. manual or automatic, don't skip the checking, and always keep an eye on the costs of your operations. good luck with your certificate adventures!
