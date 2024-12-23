---
title: "Can Hugging Face pipelines be used behind proxies on Windows Server?"
date: "2024-12-23"
id: "can-hugging-face-pipelines-be-used-behind-proxies-on-windows-server"
---

,  The question of using Hugging Face pipelines behind proxies on a Windows Server is a fairly common one, and it’s definitely something I've had to navigate in a few production environments. It's not inherently problematic, but it requires careful configuration of both the system and the Python environment. The key, really, boils down to ensuring that the necessary environment variables are correctly set and that your application—in this case, your Python script using Hugging Face—is aware of these settings. It’s not just about the `HTTP_PROXY` and `HTTPS_PROXY` variables, though they are foundational.

My experiences have taught me that there are multiple layers to this, and we shouldn't assume they all default correctly. One project, specifically, involved deploying a transformer-based sentiment analysis tool on Windows Server 2019, all nestled behind a particularly restrictive proxy. The standard approach of merely setting environment variables often fell short, leading to frustrating connection errors. This required delving deeper into both the Windows networking settings *and* the intricacies of Python's requests library, which is often a dependency for Hugging Face's operations. I recall a specific incident where the proxy required authentication using NTLM, and that introduced an entirely different set of challenges.

Here's a breakdown of what’s crucial and why, along with practical code examples.

Firstly, the foundation: environment variables. You need to set `HTTP_PROXY` and `HTTPS_PROXY`. This tells programs where to find the proxy server for outgoing http and https requests, respectively. Often, in corporate environments, these will be different URLs for internal and external access. If the proxy needs authentication, the variables should include the user and password. The correct format generally looks like this: `http://username:password@proxy_server:port` or `https://username:password@proxy_server:port`.

Now, where things often go sideways is when these variables are not correctly consumed by the underlying libraries. Python's `requests` library, widely used by many components in the Hugging Face ecosystem, usually respects these environment variables by default. However, this isn't always a guarantee. Additionally, some proxy servers may employ NTLM authentication, which adds another layer of complexity. If you encounter this, you'll likely need the `requests-ntlm` package.

Let's illustrate with code. Assume a basic Hugging Face pipeline attempting to fetch a model. Here is a basic scenario:

```python
from transformers import pipeline

try:
    classifier = pipeline("sentiment-analysis")
    result = classifier("This is a fantastic day.")
    print(result)

except Exception as e:
    print(f"Error encountered: {e}")

```

This code, if run without proxy settings, will likely fail if your machine cannot access the internet directly. To correct this, you would set environment variables in windows, perhaps by doing something like this via command prompt:

`set HTTP_PROXY=http://your_proxy_server:port`
`set HTTPS_PROXY=https://your_proxy_server:port`
`set NO_PROXY=localhost,127.0.0.1,.yourdomain.local`
(the last, `NO_PROXY`, lists hosts that should bypass the proxy)

And then re-run the python script. However, that may not be enough. If your server needs NTLM authentication, and a direct proxy URL isn’t being handled, you will need to install and use the `requests-ntlm` library and supply an adapter:

```python
import os
from transformers import pipeline
import requests
from requests_ntlm import HttpNtlmAuth

try:
    proxy_url = os.environ.get('HTTPS_PROXY')
    if not proxy_url:
        print("HTTPS_PROXY not set, skipping NTLM configuration.")
        classifier = pipeline("sentiment-analysis") #try without NTLM
    else:
        # Parse proxy URL
        proxy_parts = proxy_url.split("@")[1].split(":") if "@" in proxy_url else proxy_url.split(":")

        if len(proxy_parts) > 2: # Username and password
            proxy_username = proxy_url.split("@")[0].split("//")[1].split(":")[0] if "@" in proxy_url else None
            proxy_password = proxy_url.split("@")[0].split("//")[1].split(":")[1] if "@" in proxy_url else None
            proxy_host = proxy_parts[0]
            proxy_port = int(proxy_parts[1]) if len(proxy_parts) >=2 else None

            session = requests.Session()
            session.auth = HttpNtlmAuth(proxy_username, proxy_password)
            proxies = {
                "http": f"http://{proxy_host}:{proxy_port}",
                "https": f"https://{proxy_host}:{proxy_port}"
            }
            classifier = pipeline("sentiment-analysis", session=session, proxies=proxies)
        else:
             # Standard proxy use
            proxies = {"http": proxy_url, "https": proxy_url}
            session = requests.Session()
            classifier = pipeline("sentiment-analysis", session=session, proxies=proxies)

    result = classifier("This is a fantastic day.")
    print(result)

except Exception as e:
    print(f"Error encountered: {e}")
```

In this code, we first check if the environment variable `HTTPS_PROXY` (or `HTTP_PROXY`) is set. If it is, we parse it to extract the username, password (if present), host, and port. Then, we set up the necessary components to use NTLM if required, and pass this session to the pipeline initialization with a dictionary containing the proxy urls to use. Note the conditional logic here to account for NTLM enabled proxies versus standard proxies.

Furthermore, remember that some Windows proxy settings are also accessible at a system level, separate from environment variables. These can sometimes clash, or introduce unexpected behaviour. Be aware of the Internet Options control panel (inetcpl.cpl) and its "Connections" tab. Sometimes, ensuring that the system proxy settings are disabled or compatible with the environment variables is also essential.

Finally, if the system is running as a service, those service accounts often have their own profile, meaning those system-wide settings will not apply. This was a challenge I encountered where the python code ran perfectly locally, but failed when deployed as a windows service. Ultimately, we found that the user the windows service was running under had its own environment variables that had to be set. Setting it at a system level was insufficient.

For more in-depth learning, I'd recommend looking at the following:

* **The `requests` library documentation:** Specifically, the sections on proxies and authentication. This will give you the best understanding of how Python handles network requests.
* **"Network Programming with Python" by Brandon Rhodes:** This book covers network programming in Python, including using proxies, in significant detail.
* **"Windows Internals" by Pavel Yosifovich, et al:** While not directly about Python, understanding the internals of Windows, including how network settings work at a lower level, is very helpful for debugging challenging proxy issues, particularly with service accounts.

In conclusion, using Hugging Face pipelines behind proxies on Windows Server is entirely feasible. It requires a systematic approach, careful configuration of environment variables, and awareness of potential authentication complexities (like NTLM). Don’t hesitate to inspect the behavior of the underlying `requests` library. Remember, thorough troubleshooting is key. You'll find a lot of the headaches involved with this scenario disappear once you have these fundamentals solid. It’s not the most glamorous aspect of the development lifecycle, but doing it right avoids a lot of potential issues down the road.
