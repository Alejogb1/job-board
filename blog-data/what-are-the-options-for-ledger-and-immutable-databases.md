---
title: "What are the options for ledger and immutable databases?"
date: "2024-12-23"
id: "what-are-the-options-for-ledger-and-immutable-databases"
---

Let’s talk ledgers and immutable databases. I've been around the block with these technologies, having firsthand experience with both the headaches and triumphs they can bring, particularly when building systems where data integrity is paramount. Over the years, I’ve seen their use cases grow far beyond simple financial transactions to become crucial in supply chains, healthcare, and even some surprisingly complex gaming architectures.

The core idea behind both ledgers and immutable databases is the same: to provide a reliable, tamper-evident record of data over time. This is typically achieved by designing the system in a way that prevents alterations or deletions after a record has been written. This characteristic differs quite a bit from standard relational database management systems (RDBMS) where data is frequently updated and even deleted as required. But, while the general idea is similar, the implementation details lead to some significant differences between ledger-specific solutions and more general-purpose immutable databases.

First, let's consider ledger databases. These solutions often focus on transaction-oriented data. Think of a traditional accounting ledger, where entries represent changes in accounts, each one building sequentially on the last. Ledger databases usually focus on maintaining an audit trail, meaning each transaction has a defined order and is linked to previous transactions. This approach makes it exceptionally easy to trace the history of any given data point or account. Key aspects here often include cryptographic hashing to link blocks of transactions and prevent tampering. The goal isn't just to prevent changes, but also to make it extremely easy to prove that no unauthorized changes have been made. My past work with a distributed micro-payment system heavily relied on this property. We used a ledger database to maintain a complete, verifiable history of every transaction across many different users and services. We ultimately settled on a custom implementation utilizing cryptographic hashing and a B-tree structure on top of a simpler data store, but plenty of off-the-shelf solutions exist now.

Now, let's switch gears to immutable databases. These systems are designed to preserve a historical record of data but are generally not as transaction-centric as ledger databases. Instead, they prioritize the immutability of records within a more general storage context. A good example might be tracking document versions, sensor readings over time, or configuration changes across a large distributed system. In these cases, while the data is immutable, a transaction isn't necessarily the central concept. Rather, the idea is that each version of data has a unique, identifiable state. Some database systems achieve this by using append-only storage, or by creating new versions of existing records each time an "update" is applied while leaving the past versions unmodified. They often also leverage checksums and other forms of data verification to maintain consistency. In one specific instance, I built a system for tracking infrastructure deployment changes using an immutable database approach; this proved invaluable when rolling back failed deployments or diagnosing configuration drift, because we always had a clear, immutable record of every change we had ever made.

To illustrate, let's look at three code snippets; the first one will be a very simplified Python example demonstrating an append-only ledger using basic hashing, the second will simulate a very simplified update mechanism in an immutable database using versioning, and the third will touch on a basic read operation that shows how to access historical states from an immutable data store. Keep in mind that this is very simplified pseudocode, and a production system would be significantly more complex and would leverage specialized data structures and security measures.

**Example 1: Simplified Ledger Database (Python)**

```python
import hashlib
import json

class SimpleLedger:
    def __init__(self):
        self.chain = []
        self.previous_hash = "0"  # Genesis hash

    def add_transaction(self, data):
        transaction = {
            "data": data,
            "previous_hash": self.previous_hash
        }
        transaction_json = json.dumps(transaction, sort_keys=True).encode('utf-8')
        current_hash = hashlib.sha256(transaction_json).hexdigest()
        self.chain.append({
            "transaction": data,
            "current_hash": current_hash,
            "previous_hash": self.previous_hash
            })
        self.previous_hash = current_hash

    def verify_chain(self):
        for i in range(1, len(self.chain)):
           current_block = self.chain[i]
           prev_block = self.chain[i-1]
           recalculated_hash_string = json.dumps({
                    "data":current_block["transaction"],
                    "previous_hash": prev_block["current_hash"]}, sort_keys = True).encode('utf-8')
           recalculated_hash = hashlib.sha256(recalculated_hash_string).hexdigest()
           if recalculated_hash != current_block["current_hash"]:
              print (f"Chain compromised at block index: {i}")
              return False
        return True

ledger = SimpleLedger()
ledger.add_transaction({"account": "A", "amount": 100})
ledger.add_transaction({"account": "B", "amount": -50})
ledger.add_transaction({"account": "A", "amount": 25})
print(ledger.verify_chain()) #will print True if valid

# Attempted tampering example
ledger.chain[1]["transaction"]["amount"] = 9999 #simulate tampering with transaction
print(ledger.verify_chain()) #will print that the chain is compromised


```
Here, each transaction gets hashed and includes the hash of the previous transaction, creating a linked chain and a very simple verification mechanism.

**Example 2: Simplified Immutable Database (Python)**

```python
class SimpleImmutableDB:
    def __init__(self):
        self.data = {} # key is the id and value is a list of versions

    def update_record(self, record_id, new_value):
        if record_id not in self.data:
            self.data[record_id] = []
        self.data[record_id].append({
           "version" : len(self.data[record_id])+1,
           "data": new_value,
        })

    def get_record(self, record_id, version):
         if record_id not in self.data:
            return None
         for record in self.data[record_id]:
           if record["version"] == version:
               return record["data"]

db = SimpleImmutableDB()
db.update_record("doc1", {"title": "Initial Title", "content": "First version"})
db.update_record("doc1", {"title": "Updated Title", "content": "Second version"})
db.update_record("doc2", {"data": "Initial Data"})
print(db.get_record("doc1",1)) #will print the first version of doc1 data
print(db.get_record("doc1",2)) #will print the second version of doc1 data
print(db.get_record("doc2",1)) #will print the first version of doc2 data

```
Here, updates create new versions of a record without altering older versions, demonstrating the core concept of versioning.

**Example 3: Reading Historical Data (Python) - Using the example above, accessing specific historical states:**
The get_record() function is already part of the above code snippet. Here is a very brief example again:

```python

print(db.get_record("doc1",1)) #Retrieves the first version of "doc1" data
```
This shows us that immutable database architectures allow specific historical points in time to be easily accessed.

For deeper reading, I strongly suggest exploring academic literature on Merkle trees and blockchains, which are fundamental concepts in many ledger implementations. A good starting point is "Bitcoin: A Peer-to-Peer Electronic Cash System" by Satoshi Nakamoto, even if you aren't specifically interested in cryptocurrencies; it explains many of the core concepts very well. For a more theoretical foundation on databases, "Database System Concepts" by Abraham Silberschatz, Henry F. Korth, and S. Sudarshan is invaluable, and includes discussions on concepts such as versioning and access structures that are highly relevant to building these types of systems. Another excellent resource is the book "Designing Data-Intensive Applications" by Martin Kleppmann, which includes a solid overview of database architectures and distributed systems. Finally, research specific databases or database extensions such as Amazon QLDB, LedgerDB in Azure, or immudb that specifically handle ledger or immutable requirements for more advanced techniques.

In summary, whether you need a transaction-oriented ledger or a more general immutable store, the crucial factor is to select the solution that best aligns with the precise needs of your system. Each of these approaches has its particular strengths, and your decision should always be dictated by practical considerations of performance, scalability, and the precise security requirements of your data. There is no single "best" option, rather a wide range of options each with its own tradeoffs. I hope this is helpful!
