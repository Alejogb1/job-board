---
title: "How can I correctly configure encryption in Rails 7?"
date: "2024-12-23"
id: "how-can-i-correctly-configure-encryption-in-rails-7"
---

Let's tackle encryption in Rails 7, a topic I've spent quite a bit of time on over the years, especially back in my days working on that financial application for StellarCorp. There, ensuring data privacy was not just a best practice; it was a core requirement, and we had to get it absolutely nailed. Rails 7, thankfully, provides some excellent built-in tools, but correct configuration goes beyond simply flipping a switch. It's a layered approach, and let's break it down focusing on active storage encryption, encrypting model attributes and secrets management.

First, understand that Rails employs a few key encryption mechanisms. For data at rest, we primarily leverage Active Storage's encryption capabilities for files and Rails' built-in encrypted attributes. Data in transit, which is handled via https, isn't within the scope here – I'm assuming you've got that covered already, but it is vital!

**Active Storage Encryption**

Let's start with Active Storage. This is critical for securing any uploaded files, and it’s something I remember being quite glad to see integrated into Rails, saving us having to implement it ourselves. In Rails 7, the encryption features are available out-of-the-box. To enable it, you’ll need to configure your `config/storage.yml` file, especially when using cloud storage such as aws s3, google cloud storage or azure storage.

Here is what that might look like:

```yaml
# config/storage.yml
amazon:
  service: S3
  access_key_id: <%= ENV['AWS_ACCESS_KEY_ID'] %>
  secret_access_key: <%= ENV['AWS_SECRET_ACCESS_KEY'] %>
  region: eu-west-1
  bucket: your-s3-bucket
  encryption:
     key: <%= Rails.application.credentials.dig(:storage_encryption_key) %>
```

The critical part here is the `encryption` block, where you specify the encryption key. You’ll want to generate a strong random key and store this securely. In the example, I am using Rails’ encrypted credentials, which is usually considered a secure method. When using this, the key will be read from credentials, which you would have created using `rails credentials:edit`. This encrypted credential file will be decrypted with a master key stored in `config/master.key`.

*Note:* Do not store secrets directly in `storage.yml`. Always use environmental variables or, preferably, Rails’ encrypted credentials system.

It’s also important to remember that for existing files, enabling encryption will *not* automatically encrypt them. You'll need to either re-upload them or implement a migration to re-process them through the attachment process. This can be a time consuming operation if you have many files. There are various ways to do this, however, for efficiency, consider running this on a background job.

**Encrypting Model Attributes**

Moving on to encrypting specific model attributes, this is where Rails' `encrypts` method shines. This gem uses active support’s message verifier to provide encrypting attributes to your models. This is where you choose exactly what data gets encrypted.

Let’s look at an example on a fictional `User` model:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  encrypts :phone_number, deterministic: true
  encrypts :ssn
end
```

In this example, I've decided to encrypt the user's phone number and social security number. Note the `deterministic: true` option for `phone_number`. This indicates that the same input will always produce the same output. This enables searching or indexing the encrypted value, but it sacrifices some security compared to non-deterministic encryption, and I would advise against using it unless it is absolutely necessary. A social security number should, generally, not be searchable by its encrypted value.

Now, how is the encryption key configured? Similar to the Active Storage example, you would typically pull the key from your credentials. The encryption key, unless specified in `encrypts` method with `:key` option, which is not recommended, will be set from `Rails.application.credentials.encryption_key`. You will also need a salt value, stored by default under `Rails.application.credentials.encryption_salt`.

A few things to keep in mind here:

1.  **Key Management:** It is crucial that the encryption keys are stored securely. The Rails encrypted credentials mechanism is a good choice.
2.  **Deterministic Encryption:** As mentioned, use this option cautiously as it can compromise security if not applied thoughtfully. For sensitive data such as SSN, it’s recommended to avoid it.
3.  **Performance:** Encrypting and decrypting data adds overhead. Consider your usage patterns and whether the performance hit is acceptable.
4. **Initialization Vectors:** Rails’ encrypts method, by default, uses a random initialization vector (iv) which, in combination with the key, ensures a different ciphertext every time.

**Secrets Management**

The last, but certainly not least piece of the puzzle, is how you actually manage the encryption keys. Never store your encryption keys directly in your code or in version control. Instead use secure configuration mechanisms. I've seen way too many projects leak sensitive data by neglecting this step.

Rails offers a robust solution: `Rails.application.credentials`, which we've briefly touched upon already. This system stores encrypted secrets in `config/credentials.yml.enc` and uses a master key `config/master.key` to decrypt the secrets during application runtime. The master key should be stored somewhere very secure. In production environment, you will need to create a `master.key` file, and use an environment variable, or your secrets management system to load that master key into the container on startup.

Here’s a quick illustration of how you might set up your credentials:

```ruby
# config/credentials.yml.enc
# You will not see this in plain text.
# This is just to show an example of how credentials are stored
encryption_key: your_super_secret_encryption_key_here
encryption_salt: your_super_salt_value_here
storage_encryption_key: your_super_secret_storage_encryption_key_here
```

To generate a new encrypted credential file, and to edit this file, you use the following commands:

```bash
rails credentials:edit
```

This will open the file in your defined editor and after saving the changes, the `config/credentials.yml.enc` file will be updated.

```bash
rails credentials:show
```

This command will output all the decrypted secrets to the standard output. It’s useful to test if your configuration is correct.

It’s essential to understand that this file is *not* in clear text. It's encrypted using the master key, which is why you need that key stored securely, especially in production environments. Also be mindful of your secrets management approach, and consider things like secrets rotation. There are more advanced options, such as Vault or other secrets management services, if required.

**Further Reading**

For a deeper understanding, I recommend these resources:

*   **"Cryptography Engineering" by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno:** This book provides an excellent foundation in cryptographic principles. It's not specific to Rails, but it's invaluable for understanding the underlying mechanisms.
*   **The official Rails documentation on Active Storage:** For practical implementation details, go straight to the source. It’s continually updated. The rails guides are the canonical source of documentation for your project, and it can answer most of your questions.
*   **Rails’ API documentation:** To thoroughly understand how `encrypts` and other methods work, reviewing the API documentation is vital.

Implementing encryption properly is not an optional step; it’s a fundamental requirement for any application that deals with sensitive data. I’ve seen projects get into serious trouble by taking shortcuts on this, and it’s something I've always stressed to junior team members. Don’t leave security as an afterthought. Start by getting the fundamentals right, and the rest will follow more naturally.
