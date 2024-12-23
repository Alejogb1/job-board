---
title: "How can bash scripts securely handle private keys?"
date: "2024-12-23"
id: "how-can-bash-scripts-securely-handle-private-keys"
---

Okay, let's talk about securely managing private keys within bash scripts. I’ve seen this go wrong in so many ways during my time, and it’s absolutely crucial to get this right. The temptation to hardcode credentials directly or pass them as command line arguments is strong, especially when you're under pressure. However, that’s a recipe for disaster. You expose the keys in your script's history, process listings, and potentially in the script file itself.

The core issue is that bash scripts are, by nature, often exposed to multiple users on the system. Any user with read access to the script, or even the ability to run `ps` to view command-line arguments, can potentially compromise your keys. This isn't just a hypothetical; I once debugged a compromised server where the root cause was an improperly written script that included a hardcoded private key for database access. It took hours to isolate, and the cleanup was messy, to say the least. So, a robust security model is paramount.

The key principle I stick to here is minimizing exposure. Never store the key directly in the script. Instead, we need to leverage a more secure method. Let's break down some effective strategies.

First, we can use environment variables. This approach moves the key out of the script itself, but still requires care in how the environment variable is set. You don't want it lingering or visible to other users. The crucial part here is ensuring the variable is not globally set or displayed in process listings.

Here's a simplified example of how you might use an environment variable in a bash script:

```bash
#!/bin/bash

# This script SHOULD NOT display the actual key in the process list
# Instead it retrieves the key from environment variables

KEY_VALUE="$PRIVATE_KEY"

if [ -z "$KEY_VALUE" ]; then
  echo "Error: PRIVATE_KEY environment variable not set." >&2
  exit 1
fi

echo "Key successfully retrieved from environment variable."
# Now you would use $KEY_VALUE in secure operation
# for example, decrypting a file

echo "Some sensitive operation that requires using private key might happen here..."
# openssl enc -d -aes256 -in file.enc -out file.dec -k "$KEY_VALUE"
```

The critical element is setting `$PRIVATE_KEY` before running this script. I typically recommend something like `export PRIVATE_KEY=$(cat /path/to/my/private.key); ./my_script.sh; unset PRIVATE_KEY;`. Notice the `unset` command; this removes the environment variable after the script finishes, reducing the window of vulnerability. This is an example that uses `cat` to read directly from a file, which is not optimal and should use an encrypted file instead, which we'll discuss next. But it showcases the base principle for a bash script with an environment variable.

The problem with the approach above is that anyone can see the value of the environment variable while the script is running (using `ps aux`). To further enhance security, I prefer storing keys in encrypted files. This approach introduces a layer of access control and ensures the key is not immediately readable, even if the file is exposed. We leverage tools like `openssl` for this.

Here's an example of how you can use `openssl` to encrypt a private key and then decrypt it in a bash script:

```bash
#!/bin/bash

# Pre-requisite: Key has been encrypted with openssl
# openssl aes-256-cbc -salt -in private.key -out private.key.enc

# Script to decrypt the encrypted private key using password
# IMPORTANT: This password should NOT be hardcoded into the script
PASSWORD_VAR="$PRIVATE_PASSWORD"
FILE_PATH="$ENCRYPTED_KEY_FILE"

if [ -z "$PASSWORD_VAR" ]; then
   echo "Error: PRIVATE_PASSWORD environment variable not set." >&2
   exit 1
fi

if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File '$FILE_PATH' does not exist" >&2
  exit 1
fi


DECRYPTED_KEY=$(openssl aes-256-cbc -d -in "$FILE_PATH" -k "$PASSWORD_VAR" 2>/dev/null)

if [ -z "$DECRYPTED_KEY" ]; then
  echo "Error: Could not decrypt the key. Please ensure the password is correct." >&2
  exit 1
fi

echo "Key successfully decrypted."
# Now use $DECRYPTED_KEY in secure operation
echo "Some sensitive operation that requires using private key might happen here..."
# ssh -i <(echo "$DECRYPTED_KEY") user@host
```

Here, we are passing a password via the `$PRIVATE_PASSWORD` variable to decrypt a previously encrypted key file, `$ENCRYPTED_KEY_FILE`. Make sure this variable is also handled with the same care as `PRIVATE_KEY` example. The `2>/dev/null` redirects standard error so that password prompting is not printed to the console, and ensures sensitive details aren’t exposed.

But, even using an encrypted key introduces another variable -- the password for the encrypted file. Storing the password as an environment variable improves upon the direct key storage but does not provide perfect safety. The third option would be to leverage a dedicated key management system (KMS) such as HashiCorp Vault, which is my preferred method. With Vault, the script can authenticate and request the key dynamically without ever needing to know the underlying password or stored private key value itself.

Here's a conceptual example with Hashicorp Vault. While the implementation details vary depending on your vault setup, the core concept remains the same:

```bash
#!/bin/bash

# Pre-requisite: Vault is installed and configured.
# The script needs to authenticate with Vault before retrieving the key.

# Assumes the user has a Vault authentication method setup and is authenticated before running
# Typically this is done through a login flow using token or other mechanisms.

VAULT_ADDR="https://your-vault-address:8200"
VAULT_TOKEN="your_vault_token"
SECRET_PATH="secret/data/my-app/private-key"
SECRET_KEY="private_key"


# Retrieve the key from Vault using vault cli command
DECRYPTED_KEY=$(vault read -format=json "$SECRET_PATH" | jq -r ".data.data[\"$SECRET_KEY\"]")

if [ -z "$DECRYPTED_KEY" ]; then
  echo "Error: Could not retrieve the key from vault." >&2
  exit 1
fi

echo "Key successfully retrieved from Vault."
# Now use $DECRYPTED_KEY in secure operation
echo "Some sensitive operation that requires using private key might happen here..."
# ssh -i <(echo "$DECRYPTED_KEY") user@host
```

This example uses the `vault` cli. Authentication is assumed done before execution of the script, and the script retrieves the key from the designated Vault secret path using `jq` to parse the JSON response. This approach ensures that no sensitive keys are exposed on the machine running the script and moves the control and access of the keys to the centralized vault instance. You can also use the Vault agent for authentication which can perform token renewals.

It is important to note that in all of these situations you still have to be careful that the keys in your bash scripts are only used by a specific user, or within a secure environment that only trusted parties have access. In addition to the strategies highlighted above, I highly recommend reading "Applied Cryptography" by Bruce Schneier for a deep dive into cryptographic principles. The OWASP guides, such as "OWASP Secure Coding Practices," and "OWASP API Security," are also indispensable resources that you should utilize to understand various attack vectors that might be employed against your system. These resources will provide a more in-depth understanding of secure practices and help you make informed choices.

In summary, never hardcode keys. Prefer using environment variables, encrypted files, and, for production systems, opt for a dedicated KMS like HashiCorp Vault, focusing on minimizing key exposure at every stage. Following these precautions makes your bash scripts significantly more secure. Remember, it's an ongoing process of layering protection, not a one-time fix.
