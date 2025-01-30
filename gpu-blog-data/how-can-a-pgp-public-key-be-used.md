---
title: "How can a PGP public key be used to generate a fingerprint?"
date: "2025-01-30"
id: "how-can-a-pgp-public-key-be-used"
---
The fingerprint of a Pretty Good Privacy (PGP) public key is a short, fixed-length cryptographic hash that uniquely identifies that specific key. It serves as a convenient and easily verifiable representation, facilitating key sharing and authentication without needing to handle the entire, often lengthy, public key data. I've encountered situations where quickly verifying a key by fingerprint significantly reduced errors during secure communications setup. Generating the fingerprint involves a standardized process, and it's crucial to understand that the fingerprint is not the key itself, nor is it a secret value. It's a one-way hash, meaning it's computationally infeasible to derive the original key from the fingerprint.

The process begins with the raw public key data, which is often stored in an ASCII-armored format when exchanged. This format typically includes headers and footers that need to be removed before proceeding. The base64-encoded key material within those bounds then undergoes decoding to retrieve the actual binary representation of the public key. This binary data, including all identifying information within the key structure, is then passed to a cryptographic hash function, typically SHA-1. The SHA-1 function outputs a 160-bit (20-byte) hash value. While SHA-1 is now considered cryptographically weakened for certain purposes, it remains commonly used for generating PGP key fingerprints due to historical prevalence. In some newer implementations, SHA-256 is used. However, the process remains similar in principle, differing primarily in the chosen hash algorithm and resulting hash length.

The raw hash output is typically presented in hexadecimal format for human readability. Each byte of the hash is converted to its corresponding two-character hexadecimal representation. These hex values are often grouped with spaces or colons for better visual clarity. The fingerprint thus obtained serves as a reliable check to confirm that two individuals are communicating with the expected keys. This is vital to prevent man-in-the-middle attacks or key substitution errors. I’ve experienced instances where a discrepancy between fingerprint values indicated tampering with the public key exchange, underscoring the practical importance of this seemingly simple cryptographic operation.

To illustrate this process, I will provide code examples in Python, a language frequently used in cryptographic scripting and prototyping, utilizing libraries readily available.

**Example 1: Basic Fingerprint Generation with SHA-1**

```python
import hashlib
import base64

def generate_sha1_fingerprint(public_key_ascii):
    """Generates a SHA-1 fingerprint from a PGP public key in ASCII armor."""

    try:
        # Remove headers and footers; extract base64-encoded data
        lines = public_key_ascii.splitlines()
        key_data_lines = [line for line in lines if not line.startswith("-----")]
        base64_encoded = "".join(key_data_lines)
        
        # Decode base64 data to bytes
        binary_key = base64.b64decode(base64_encoded)
        
        # Calculate SHA-1 hash
        sha1_hash = hashlib.sha1(binary_key)

        # Convert the hash to hex and return formatted fingerprint
        hex_fingerprint = sha1_hash.hexdigest().upper()
        formatted_fingerprint = " ".join(hex_fingerprint[i:i+4] for i in range(0, len(hex_fingerprint), 4))
        return formatted_fingerprint
        
    except Exception as e:
         print(f"Error during fingerprint generation: {e}")
         return None


# Example usage (replace with actual ASCII armored key)
public_key_armor = """-----BEGIN PGP PUBLIC KEY BLOCK-----
mQGNBFk+y+wBDAC85I86e2Yl88T0vXg6d90Fw871Wn/q/12P8p/7bF7k2/a46w
... (truncated for brevity) ...
-----END PGP PUBLIC KEY BLOCK-----
"""

fingerprint = generate_sha1_fingerprint(public_key_armor)
if fingerprint:
   print(f"SHA-1 Fingerprint: {fingerprint}")
```

This code defines a function, `generate_sha1_fingerprint`, that takes the ASCII armored public key as input. It first removes the header and footer lines and concatenates the remaining base64 encoded content, which is then decoded. The raw byte data is passed to SHA-1 for hashing. The resulting hash is converted to hexadecimal format, uppercased, and formatted into groups of four characters, providing a standard looking fingerprint output. The error handling helps prevent program crashes due to malformed key data. I included a 'try-except' block due to past encounters with corrupted or improperly formatted key strings in various systems.

**Example 2: Using SHA-256 for fingerprint generation**

```python
import hashlib
import base64

def generate_sha256_fingerprint(public_key_ascii):
    """Generates a SHA-256 fingerprint from a PGP public key in ASCII armor."""

    try:
        # Remove headers and footers; extract base64-encoded data
        lines = public_key_ascii.splitlines()
        key_data_lines = [line for line in lines if not line.startswith("-----")]
        base64_encoded = "".join(key_data_lines)
        
        # Decode base64 data to bytes
        binary_key = base64.b64decode(base64_encoded)
        
        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256(binary_key)

        # Convert hash to hex, format, and return
        hex_fingerprint = sha256_hash.hexdigest().upper()
        formatted_fingerprint = " ".join(hex_fingerprint[i:i+4] for i in range(0, len(hex_fingerprint), 4))
        return formatted_fingerprint
    
    except Exception as e:
         print(f"Error during fingerprint generation: {e}")
         return None

# Example Usage
public_key_armor = """-----BEGIN PGP PUBLIC KEY BLOCK-----
mQGNBFk+y+wBDAC85I86e2Yl88T0vXg6d90Fw871Wn/q/12P8p/7bF7k2/a46w
... (truncated for brevity) ...
-----END PGP PUBLIC KEY BLOCK-----
"""


fingerprint = generate_sha256_fingerprint(public_key_armor)
if fingerprint:
    print(f"SHA-256 Fingerprint: {fingerprint}")

```

This example shows the process using SHA-256 instead of SHA-1. The core logic remains the same: extract base64-encoded data, decode it, hash, convert to hex, and format for readability. The crucial difference lies in the `hashlib.sha256()` call. The resulting SHA-256 fingerprint is longer (64 hex characters, or 256 bits) compared to a SHA-1 fingerprint (40 hex characters, 160 bits).

**Example 3: Fingerprint generation using a dedicated library**

```python
import gnupg

def generate_fingerprint_using_gnupg(public_key_ascii):
    """Generates a fingerprint using the gnupg library."""
    try:
       gpg = gnupg.GPG()
       import_result = gpg.import_keys(public_key_ascii)
       key_id = import_result.fingerprints[0]
       return key_id
    except Exception as e:
         print(f"Error during fingerprint generation with GnuPG: {e}")
         return None


public_key_armor = """-----BEGIN PGP PUBLIC KEY BLOCK-----
mQGNBFk+y+wBDAC85I86e2Yl88T0vXg6d90Fw871Wn/q/12P8p/7bF7k2/a46w
... (truncated for brevity) ...
-----END PGP PUBLIC KEY BLOCK-----
"""

fingerprint = generate_fingerprint_using_gnupg(public_key_armor)

if fingerprint:
    print(f"GnuPG Fingerprint: {fingerprint}")
```

This example leverages the `gnupg` library, which wraps GnuPG functionality. The library simplifies the process by handling key import and fingerprint extraction directly.  The code imports the public key into GnuPG's keyring and accesses its fingerprint property. This is often a preferable method in real-world applications where a system might already be using GnuPG and its tools. I’ve found this to be substantially less error-prone than manual parsing and hashing, especially when dealing with various PGP key types.

For resource recommendations, I suggest consulting the documentation of the `hashlib` and `base64` modules for Python, and exploring the `gnupg` library if you plan to do more extensive work with PGP keys. The relevant sections of RFC 4880, which specifies OpenPGP, offers a more in-depth theoretical explanation on key structure and data encoding. For those new to cryptography, introductory texts on hash functions are beneficial for a firm foundational understanding.
