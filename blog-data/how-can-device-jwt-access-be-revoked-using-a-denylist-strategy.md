---
title: "How can device JWT access be revoked using a denylist strategy?"
date: "2024-12-23"
id: "how-can-device-jwt-access-be-revoked-using-a-denylist-strategy"
---

Let's unpack device JWT revocation via a denylist, shall we? This is a scenario I’ve encountered several times, often in the context of IoT platforms where the lifecycle of a device is a bit more… dynamic than your average user account. A simple expiration policy on JWTs isn't always enough; you frequently need to forcibly invalidate a token *before* its natural expiration, particularly if a device is compromised or decommissioned.

The core idea behind a denylist strategy is straightforward: we maintain a list of JWT identifiers (typically, the `jti` claim) that are explicitly considered invalid. When a request arrives with a JWT, we must check that its `jti` is *not* present in the denylist before considering it valid. This provides an additional layer of control beyond the token's inherent expiration date. It does introduce a state management aspect that requires careful handling, of course, but it’s a robust approach for handling premature revocation.

From my experience, the implementation choices for a denylist vary based on scale and latency requirements. A naive approach might involve a simple in-memory set for very small, low-traffic systems, but that won’t cut it at scale. I’ve seen that quickly become a bottleneck. For more substantial systems, a persistent datastore with fast read access, like a Redis cache or a dedicated key-value store, is almost always the better solution. Consistency, also, is something to bear in mind. A distributed denylist needs a strategy for replication and consistent reads.

Let’s break down how we can approach this programmatically. I will show you examples in python, since that's a language many in the industry are comfortable with and it simplifies the concept.

Here's a basic example demonstrating how you'd integrate a simple in-memory denylist check. This is purely for illustrative purposes and *not* recommended for production.

```python
import jwt
import datetime

denylisted_jti = set()

def generate_jwt(payload, secret_key, expiry_minutes=30):
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiry_minutes)
    payload['jti'] = str(uuid.uuid4()) # Add a unique jti
    return jwt.encode(payload, secret_key, algorithm='HS256')

def validate_jwt(token, secret_key):
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        if payload['jti'] in denylisted_jti:
             return False, "token has been revoked"
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, "token has expired"
    except jwt.InvalidTokenError:
        return False, "invalid token"

def revoke_jwt(jti):
    denylisted_jti.add(jti)

# Example usage:
import uuid
secret = "my_super_secret_key"
token = generate_jwt({'user_id': 'device123'}, secret)
valid, payload = validate_jwt(token, secret)

if valid:
    print("token is valid:", payload)

jti_to_revoke = payload['jti']
revoke_jwt(jti_to_revoke)

valid, payload = validate_jwt(token, secret)

if not valid:
    print("Token is not valid:", payload)

```

In this snippet, I used a standard `set` to store the `jti`s that need to be revoked. The `generate_jwt` function has been modified to include a `jti` claim. The `validate_jwt` function then checks the `jti` before allowing the validation. As you can see, after we've added a `jti` to the `denylisted_jti` set, the token immediately fails validation. This demonstrates the core principle of how a denylist is used to revoke access.

However, as I stated previously, an in-memory denylist will not suffice in most real-world applications. Let's explore using Redis instead for our denylist, focusing on the necessary changes.

```python
import jwt
import redis
import datetime
import uuid

redis_client = redis.Redis(host='localhost', port=6379, db=0)


def generate_jwt(payload, secret_key, expiry_minutes=30):
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiry_minutes)
    payload['jti'] = str(uuid.uuid4())
    return jwt.encode(payload, secret_key, algorithm='HS256')

def validate_jwt(token, secret_key):
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        if redis_client.exists(payload['jti']): # Check Redis for denylisted jtis
            return False, "token has been revoked"
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, "token has expired"
    except jwt.InvalidTokenError:
        return False, "invalid token"

def revoke_jwt(jti):
    redis_client.set(jti, 'revoked', ex=60 * 60 * 24 * 7)  # Add to Redis denylist with a 7-day expiry
    # Add an expiration if needed, to purge old revoked tokens

# Example usage
secret = "my_super_secret_key"
token = generate_jwt({'user_id': 'device123'}, secret)
valid, payload = validate_jwt(token, secret)

if valid:
    print("token is valid:", payload)

jti_to_revoke = payload['jti']
revoke_jwt(jti_to_revoke)

valid, payload = validate_jwt(token, secret)

if not valid:
    print("Token is not valid:", payload)

```
Notice now that our denylist is stored in Redis. The redis connection initialization, the way we add to the denylist (`redis_client.set`), and how we check for membership during the token validation have changed. The `redis_client.exists` method is a fast lookup that avoids any manual iteration. Also, I’ve added an expiry when adding a `jti` to the denylist. This is a good practice, ensuring that our denylist does not grow indefinitely. Here, I have used an expiry of one week. You can adapt that for your needs.

Finally, let's consider a more elaborate scenario. Imagine needing to manage not just revoked tokens, but also the *reason* for the revocation, maybe to provide details to support teams. In that case, a denylist approach can still be useful.

```python
import jwt
import redis
import datetime
import uuid
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def generate_jwt(payload, secret_key, expiry_minutes=30):
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiry_minutes)
    payload['jti'] = str(uuid.uuid4())
    return jwt.encode(payload, secret_key, algorithm='HS256')

def validate_jwt(token, secret_key):
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        revocation_data_json = redis_client.get(payload['jti'])
        if revocation_data_json:
             revocation_data = json.loads(revocation_data_json)
             return False, f"token has been revoked, reason: {revocation_data['reason']}"
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, "token has expired"
    except jwt.InvalidTokenError:
        return False, "invalid token"

def revoke_jwt(jti, reason):
    revocation_data = {"reason": reason, "revoked_at": str(datetime.datetime.utcnow())}
    redis_client.set(jti, json.dumps(revocation_data), ex=60*60*24*7) # Store revocation data

# Example usage
secret = "my_super_secret_key"
token = generate_jwt({'user_id': 'device123'}, secret)
valid, payload = validate_jwt(token, secret)

if valid:
    print("token is valid:", payload)

jti_to_revoke = payload['jti']
revoke_jwt(jti_to_revoke, "device_compromised")

valid, payload = validate_jwt(token, secret)

if not valid:
    print("Token is not valid:", payload)

```
Here, the `redis_client.set` method is used to store a JSON object containing the revocation reason along with a timestamp, which can be helpful in debugging and auditing. We still use an expiry, and now `validate_jwt` is able to decode that JSON object to return the reason for revocation. This gives us a richer context on why a token was revoked.

For further reading, I would strongly recommend consulting "Designing Data-Intensive Applications" by Martin Kleppmann. It covers a lot of ground concerning distributed systems and provides a good understanding of the underlying problems when dealing with state management. Additionally, for a deeper understanding of JWT security best practices, the official IETF draft on JSON Web Token (RFC 7519) is a crucial resource. Lastly, looking into academic papers about distributed consistency models can offer a much more in-depth view about how a denylist strategy can fail under certain conditions. These resources can be invaluable for a complete understanding of the topic.

To summarize, a denylist for JWT revocation is a powerful technique, but requires careful consideration of data storage, consistency, and access patterns. Always test your code thoroughly, and understand the trade-offs you are making for performance vs consistency. Choosing the right strategy for a given situation is essential to a performant and secure system.
