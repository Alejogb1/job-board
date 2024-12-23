---
title: "Do all peer nodes in Hyperledger Fabric store all user public keys?"
date: "2024-12-23"
id: "do-all-peer-nodes-in-hyperledger-fabric-store-all-user-public-keys"
---

Let's dive straight into it, shall we? The question of whether all peer nodes in Hyperledger Fabric store every user's public key is a common point of confusion, and rightfully so. Having spent a fair bit of time architecting and debugging Fabric networks, I can tell you unequivocally that the answer is: absolutely not. Fabric doesn't operate that way, and thank goodness for that, because the implications of such a design would be catastrophic for scalability and security.

The key concept here is the separation of identity management and transaction processing. Fabric leverages the concept of a Membership Service Provider (MSP), which is where user identities, including their public keys, are primarily managed. Instead of every peer node holding a comprehensive database of all user public keys, Fabric nodes receive necessary identity information on a need-to-know basis – specifically, alongside transactions that the peer must validate.

Think about it this way: Imagine a bank where every teller needs to personally know every single customer's signature on file. That's not how modern banking works. Instead, the tellers have a system that verifies signatures on a per-transaction basis, using a central registry. Fabric operates similarly. The "central registry" is, in essence, distributed across the MSPs, and the verification information is included within transactions.

When a client submits a transaction, it does so with a digital signature created using its private key. This transaction also includes the client's public certificate issued by the MSP. This certificate is not simply a public key, it's a more complete package including other identity information. Peers validating the transaction use the attached certificate to verify the digital signature, making sure the user indeed authorized the transaction, and also confirming that the certificate comes from a trusted MSP. Crucially, the peers do *not* store the certificates persistently beyond what's needed for on-chain validation.

Let me share a scenario from a past project, where we were implementing a supply chain solution. We had multiple organizations, each with its own MSP. If every peer in every organization had to store every other organization's user certificates, it would have been an unmanageable mess. Instead, the transaction payloads included the necessary credentials for the endorsing and committing peers to validate the transactions.

Now, let's solidify this with some illustrative code snippets. These are simplified representations to give you the gist; actual fabric code is naturally more involved.

**Snippet 1: Client-Side Transaction Creation (Illustrative)**

This snippet demonstrates the client acquiring its identity from the MSP and using it to sign the transaction proposal. This is akin to signing a check using a unique signature linked to an account held by a trusted institution.

```python
# Client.py - Simplified illustration of transaction creation.

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
import base64

# Assume we have loaded client's private key and certificate from the MSP.
# We're skipping loading mechanisms for brevity and assuming we have the objects already
client_private_key =  '<<pretend this is actual private key bytes>>'
client_certificate = '<<pretend this is actual certificate bytes>>'

# Lets use a dummy payload for the transaction
transaction_payload = b"Transfer 100 units to recipient_x."

# This function simulates how a transaction is signed.
def sign_transaction(private_key_bytes, payload):
    private_key = serialization.load_pem_private_key(
        private_key_bytes, password=None
    )
    signature = private_key.sign(
        payload,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


signature = sign_transaction(client_private_key, transaction_payload)


# Package up payload, signature, and certificate for transmission to the peers.
transaction_proposal = {
    "payload": base64.b64encode(transaction_payload).decode('utf-8'),
    "signature": signature,
    "certificate": base64.b64encode(client_certificate).decode('utf-8')
}

print(f"Transaction Proposal: {transaction_proposal}")

```

**Snippet 2: Peer-Side Transaction Validation (Illustrative)**

This snippet shows how a peer uses the accompanying certificate to verify the signature on the transaction. The peer doesn't need prior knowledge of the client's public key or private key, it only needs to trust the issuer of the certificate.

```python
# Peer.py - Simplified illustration of transaction validation

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography import x509
import base64


# Let's use the transaction data from the client.py example.
transaction_proposal = {
   "payload": "VHJhbnNmZXIgMTAwIHVuaXRzIHRvIHJlY2lwaWVudF94Lg==",
   "signature": "<<this is the base64 encoded signature from Client.py>>",
   "certificate": "<<this is the base64 encoded certificate from Client.py>>"
}

def verify_signature(certificate_bytes, payload, signature_bytes):
    certificate = x509.load_pem_x509_certificate(certificate_bytes)
    public_key = certificate.public_key()

    try:
        public_key.verify(
            signature_bytes,
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception as e:
         print (f"Signature verification failed : {e}")
         return False


certificate_bytes = base64.b64decode(transaction_proposal["certificate"].encode('utf-8'))
payload_bytes = base64.b64decode(transaction_proposal["payload"].encode('utf-8'))
signature_bytes = base64.b64decode(transaction_proposal["signature"].encode('utf-8'))
is_valid = verify_signature(certificate_bytes, payload_bytes, signature_bytes)

if is_valid:
    print("Transaction signature is valid.")
else:
    print("Transaction signature is invalid.")

```

**Snippet 3: MSP Interaction (Illustrative)**

While peers don't store all public keys, the MSP does. This snippet (highly simplified) shows conceptually how a user's certificate might be retrieved.

```python
# MSP.py - Conceptual representation of MSP functionality
# Notice how the "private" key is not available here
import json
import base64

# In a real system this would be an actual storage mechanism like a database.
# We are using a dictionary here for example only.
msp_storage = {
   "user_x" : {
        "certificate" : "<<base64 encoded certificate>>",
        "revoked" : False
   }
}

def get_certificate_by_user(user_id):
    user_data = msp_storage.get(user_id)
    if user_data and user_data.get("revoked") == False:
      return base64.b64decode(user_data.get("certificate").encode('utf-8'))
    else:
      return None


certificate = get_certificate_by_user("user_x")

if certificate:
    print(f"Certificate Retrieved: {certificate}")
else:
  print(f"Certificate Not Found or Revoked.")

```

These examples demonstrate the fundamental architecture of Fabric's identity management system. Peers are more like "verifiers" and rely on the MSPs for providing proof of identity rather than acting as a directory of all valid public keys. This design significantly improves both the scalability and security of the system. If each peer stored every user's public key (including that of users across different organizations), updating a single certificate revocation would require changes across all peers in the network, leading to serious performance bottlenecks and security vulnerabilities.

For a deeper dive into these concepts, I'd highly recommend exploring the *Hyperledger Fabric documentation*, specifically the sections on the Membership Service Provider (MSP) and transaction flow. Furthermore, "Mastering Blockchain" by Imran Bashir provides an excellent general overview with relevant details on Fabric. Also, the *"Hyperledger Fabric in Action"* by Manning publications can offer a more practice-oriented approach on concepts explained above. Finally, understanding the core cryptography concepts in *"Cryptography Engineering"* by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno is crucial for building robust decentralized applications.

In conclusion, Fabric doesn’t store all user public keys on all peer nodes. It utilizes the MSP to manage identities and provides necessary verification information with each transaction, ensuring a more secure and scalable network architecture. This division of responsibilities between peers and MSPs is a foundational element of Fabric’s design and is critical for its operation.
