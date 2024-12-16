---
title: "How do I use separate ReCaptcha keys for dev and production?"
date: "2024-12-16"
id: "how-do-i-use-separate-recaptcha-keys-for-dev-and-production"
---

Alright, let's talk about reCaptcha keys across development and production environments. This isn't just academic; I've been down this road more times than I care to recall, and it’s a common pitfall that can lead to some surprisingly frustrating situations if not managed properly. Setting up distinct reCaptcha keys isn't solely about keeping things tidy—it’s about ensuring accurate testing and safeguarding your production environment against test-related traffic and potential key leakage.

From my experience, especially during the early days of building large-scale applications, I learned the hard way that using the same keys across all environments is a recipe for headaches. Debugging user validation flows when they're interwoven with test submissions is a pain, and exposing production keys in development builds is, well, a security risk we definitely want to avoid. The core idea here is that each environment should be a self-contained unit with its own set of configurations.

The problem stems from how reCaptcha verification works; a request needs a site key and secret key to validate a token. A single pair used everywhere will create a lack of proper testing, as validation success depends heavily on environment context which should be avoided.

There are several approaches to manage this separation, but I find the most maintainable one involves configuration management. Essentially, the application should obtain the reCaptcha site key from a configurable source, specific to the environment in which it's running. Let's unpack that further.

First, you need to generate separate reCaptcha keys. Go to the Google reCaptcha admin console and create one set of keys for your development environment and another for production. Name them clearly; something like 'my_app_dev' and 'my_app_prod' is a good starting point. The site key will be embedded in your frontend code, while the secret key is used on the backend for validation. Crucially, the backend should never receive the site key.

The following code examples will demonstrate how this can be achieved, using both javascript (frontend) and python (backend), and for a simple configuration management. This approach is generic, and similar logic could be implemented using other languages/frameworks.

**Example 1: Frontend (JavaScript with Environment Variables)**

```javascript
// Assume your application is bundled with a process that substitutes
// environment variables during build. For example, Webpack or similar.
// The `process.env.REACT_APP_RECAPTCHA_SITE_KEY` is set differently for each
// build.

function loadRecaptcha() {
  const siteKey = process.env.REACT_APP_RECAPTCHA_SITE_KEY;
  if (!siteKey) {
    console.error('reCaptcha Site Key not provided. Please configure environment variables.');
    return;
  }

  // Dynamically create and append a script tag to load the reCaptcha API.
  const script = document.createElement('script');
  script.src = `https://www.google.com/recaptcha/api.js?render=${siteKey}`;
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

function executeRecaptcha(action) {
    return new Promise((resolve, reject) => {
        if(typeof grecaptcha === 'undefined') {
            console.error('reCaptcha API not loaded yet. Ensure you have a network connection');
            reject('reCaptcha API not loaded yet.');
        }

        grecaptcha.ready(() => {
          grecaptcha.execute(process.env.REACT_APP_RECAPTCHA_SITE_KEY, { action: action }).then(token => {
              resolve(token);
          }).catch(err => {
            console.error('Error obtaining reCaptcha token', err);
            reject('Error obtaining reCaptcha token');
          })
        });
    });
}

// Call loadRecaptcha on initial application mount.
loadRecaptcha();
// executeRecaptcha can be called before submitting a form.
```

In this frontend example, you can see that the site key, the one exposed within the html, is managed by an environment variable. Depending on the environment where the application is running, the correct key will be picked up. This ensures there is no hard coded key and no leakage can be made.

**Example 2: Backend (Python using environment variables and .env files)**

```python
# Assuming you have a library like `python-dotenv` installed
import os
from google.oauth2 import service_account
from google.auth.transport import requests
from google.auth import credentials
import json
import requests

from dotenv import load_dotenv
load_dotenv()


def verify_recaptcha(token: str) -> bool:
    secret_key = os.getenv("RECAPTCHA_SECRET_KEY")
    if not secret_key:
      print('ReCaptcha Secret Key not found. Please provide within the environment')
      return False

    try:
      data = {
        "secret": secret_key,
        "response": token
      }
      response = requests.post("https://www.google.com/recaptcha/api/siteverify", data=data)
      response.raise_for_status()

      result = response.json()
      if result.get('success'):
          return True
      else:
          print(f'ReCaptcha validation failure, {result.get("error-codes")}')
          return False
    except Exception as e:
      print(f'ReCaptcha exception: {e}')
      return False


# This would be called after receiving a reCaptcha token from the frontend
#  response = verify_recaptcha(token_received_from_frontend)
```

For the backend code, the `RECAPTCHA_SECRET_KEY` should be an environment variable configured in your deployment environment. The python script reads it using `os.getenv`. This separates the secret key from being hardcoded in the code which is the basic security principle. In this case, a .env file is used locally to help development.

**Example 3: Backend (Python using a configuration file)**

Alternatively, if using a framework that encourages using config files, one can leverage those mechanisms.

```python
import json
import requests
import os

def load_config(env: str) -> dict:
    try:
        config_file = f"config_{env}.json"
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file for {env} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid config file format for {env}")
        return {}

def verify_recaptcha(token: str) -> bool:
    env = os.getenv("APP_ENV", "dev") # or from any other environment configuration mechanism
    config = load_config(env)

    secret_key = config.get("recaptcha", {}).get("secret_key")
    if not secret_key:
      print(f'Recaptcha Secret key not found within {env} config')
      return False

    try:
      data = {
        "secret": secret_key,
        "response": token
      }
      response = requests.post("https://www.google.com/recaptcha/api/siteverify", data=data)
      response.raise_for_status()

      result = response.json()
      if result.get('success'):
          return True
      else:
          print(f'ReCaptcha validation failure, {result.get("error-codes")}')
          return False
    except Exception as e:
      print(f'ReCaptcha exception: {e}')
      return False
```

In this example, each environment has its own json configuration file (e.g., `config_dev.json` , `config_prod.json`). The desired environment is extracted via the environment variable `APP_ENV`, and a specific configuration is loaded. This approach is useful when you want to manage other configuration parameters within your application.

These snippets show the core principles: environment-specific configuration and the separation of site and secret keys. The frontend only receives the site key, while the backend handles the verification via the secret key. Both utilize environment variables or a configuration file to keep these secrets separate.

For further insights into secure configuration management, I recommend reading "The Twelve-Factor App" guidelines. Though not directly focused on reCaptcha, it offers a great overview of best practices for application configuration and environment management. "Building Microservices" by Sam Newman offers further perspective on how to organize your application for multi environment deployment. Finally, understanding the details of the reCaptcha API itself is essential; Google’s official documentation is the most reliable source for that. The google cloud documentation itself has many resources that help in understanding the concepts of different environments.

Managing different keys is a foundational practice for robust and secure application deployment. Setting this up correctly upfront will avoid significant headaches further down the line, especially as your application scales.
