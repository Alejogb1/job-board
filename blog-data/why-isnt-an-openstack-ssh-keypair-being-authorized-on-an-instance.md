---
title: "Why isn't an OpenStack SSH keypair being authorized on an instance?"
date: "2024-12-23"
id: "why-isnt-an-openstack-ssh-keypair-being-authorized-on-an-instance"
---

Alright, let’s tackle this authorization puzzle. From my experience, when an openstack ssh keypair refuses to authorize correctly, it rarely boils down to a singular cause. It's typically a multi-faceted issue, requiring a systematic investigation. I’ve personally been through this more times than i’d care to count, often with a late-night troubleshooting session involved. Let’s break it down into the most likely culprits and how to address them.

Firstly, we need to consider that the issue might not actually be *with* the keypair itself, but rather with the propagation mechanism, the instance's configuration, or even the user's procedure. When I've faced this in the past, I've found it useful to start by verifying the basics, moving gradually toward the more complex scenarios.

Let’s begin with the keypair itself. I’ve seen numerous instances where users accidentally upload the public *and* private key as a keypair or, conversely, try to use a public key format that isn’t recognized by openstack. Openstack expects the key to be in a specific ssh-rsa format, often encoded in the traditional ssh public key format, starting with “ssh-rsa”, followed by the base64 encoded key material and, often, an optional comment.

Here's a quick snippet in Python demonstrating how you could extract a correctly formatted ssh public key if you happen to have the private one available using the `paramiko` library:

```python
import paramiko
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def extract_public_key(private_key_path):
  """Extracts the public key from an RSA private key and formats it for OpenStack.

  Args:
    private_key_path: Path to the private key file.

  Returns:
    A string containing the public key in OpenStack's format or None if fails.
  """
  try:
    with open(private_key_path, 'rb') as f:
      private_key_bytes = f.read()
    private_key = serialization.load_ssh_private_key(
            private_key_bytes,
            password=None
        )

    if isinstance(private_key, rsa.RSAPrivateKey):
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        return public_key_bytes.decode('utf-8')
    else:
        return None

  except Exception as e:
    print(f"Error processing key: {e}")
    return None


# Example usage:
private_key_file = 'path/to/your/private_key'  # Replace with your path
public_key_string = extract_public_key(private_key_file)

if public_key_string:
  print(public_key_string)
else:
  print("Could not extract a valid public key.")

```

This snippet is useful because it demonstrates the precise format that OpenStack expects for the public key. Ensure the public key you upload to OpenStack, either when creating the keypair or assigning it to the instance, conforms to this format. Pay special attention to the presence of ‘ssh-rsa’ at the beginning, as well as a correctly base64 encoded blob. Any deviations will cause issues.

Next, let’s address instance metadata. This is where a lot of my past problems have stemmed from. OpenStack relies on cloud-init (or similar tools) inside the instance to fetch the public key via metadata and install it. The public key is usually injected into the `.ssh/authorized_keys` file for the specified user, typically the default user (e.g., ubuntu, cloud-user, centos). If cloud-init fails, for instance, due to networking errors or due to conflicts with an improperly configured network setup, the key won’t be set up correctly.

To diagnose this, I recommend taking a look at cloud-init’s logs on the instance itself, assuming you have a temporary way of accessing it, perhaps via the console or another method if possible. These logs are usually found under `/var/log/cloud-init.log` or `/var/log/cloud-init-output.log`, or sometimes `var/log/cloud-init.log.0`. Inspecting these logs will offer important clues, such as errors in retrieving metadata or failures in key placement. A crucial aspect to look for are messages indicating that cloud-init encountered issues connecting to the metadata service or that there were errors parsing or applying the provided public key.

Here's a basic bash script that can help you check if the key is present in the `authorized_keys` file and verify that the correct user is associated with the key setup:

```bash
#!/bin/bash

USERNAME="ubuntu" # Replace with the actual user

if [ -f "/home/$USERNAME/.ssh/authorized_keys" ]; then
  echo "authorized_keys file found."
  if grep -q "ssh-rsa" "/home/$USERNAME/.ssh/authorized_keys"; then
      echo "ssh-rsa public key detected in authorized_keys."
      grep "ssh-rsa" "/home/$USERNAME/.ssh/authorized_keys"
  else
    echo "No ssh-rsa key found in authorized_keys."
  fi
else
  echo "authorized_keys file not found for user $USERNAME."
fi

echo "Checking cloud-init logs..."
sudo tail -n 20 /var/log/cloud-init-output.log
sudo tail -n 20 /var/log/cloud-init.log

```

This simple script checks for the presence of the `authorized_keys` file, looks for the 'ssh-rsa' pattern within it, and outputs the last twenty lines from the cloud-init logs. Running this directly within your instance will give you a quick indication of the state of key injection and cloud-init processing. Remember to adjust the `USERNAME` variable to match the user you expect to log in with.

Furthermore, instance networking is critical. If the instance cannot reach the metadata service to fetch the public key (typically at 169.254.169.254, sometimes via a dedicated metadata API endpoint), cloud-init will fail to set up the keys. Check that the instance has a valid network configuration, including DNS settings. Incorrect network configurations can prevent the instance from establishing an HTTP connection needed for retrieving the metadata. If using a custom network configuration, double-check your network setup. This also includes checking any access control lists or security groups that might be in the way.

Lastly, ensure there are no conflicts with user data settings that might be overwriting cloud-init’s key management steps. Some users inject their own user data, possibly inadvertently. If this data has instructions that conflict with cloud-init’s logic, or includes commands that remove or alter the authorized_keys file after the key has been injected, it will appear that the key authorization isn’t working.

For deeper insight into cloud-init internals and troubleshooting practices, consult the official cloud-init documentation. Also, the book "Mastering OpenStack" by Omar K. Al-Masri and colleagues provides comprehensive information on OpenStack internals, including the role of the metadata service and cloud-init. For a broader understanding of network configurations within OpenStack environments, "OpenStack Cloud Computing Cookbook" by Kevin Jackson is an excellent resource. The "Linux System Programming" book by Robert Love provides an invaluable deeper dive into the linux system’s internals that cloud-init runs on, for a deeper understanding. Careful study of these documents and a systematic approach when troubleshooting is often key to resolution.

In my experience, it is nearly always one of these three main areas that cause the problem. Start with the basics—key formatting, then move to instance configuration and network accessibility, and you'll typically uncover the root cause without too much hassle.
