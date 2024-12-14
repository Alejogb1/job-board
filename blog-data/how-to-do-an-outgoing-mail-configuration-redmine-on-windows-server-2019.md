---
title: "How to do an Outgoing mail configuration: Redmine on Windows Server 2019?"
date: "2024-12-14"
id: "how-to-do-an-outgoing-mail-configuration-redmine-on-windows-server-2019"
---

alright, let's tackle this. configuring outgoing email for redmine on windows server 2019, it's a dance we've all done at some point. it's never quite as straightforward as it should be, is it? i've personally spent more hours than i'd like to recount wrestling with smtp settings, so i feel your pain.

the core of the problem revolves around making redmine, which is a ruby on rails application, communicate with an smtp server. this server is responsible for actually sending the email. windows server 2019, in itself, isn't directly involved in sending the emails unless you're using a local smtp server, which honestly is rarely the case in modern environments.

let's get into the details. first thing, you need to locate your redmine configuration file. this is usually `configuration.yml` and it's typically found under `redmine\config`. it's a yml file so be careful about the spaces and tabs. it's usually in the redmine folder. i remember a time back in 2015 i messed up indentation with a yml file and spend two days trying to figure it out until a colleague point it out, so pay attention to that. the email settings we’re interested in will be under a `production:` section.

here’s a basic configuration i've used before, and i'll explain each part.

```yaml
production:
  email_delivery:
    delivery_method: smtp
    smtp_settings:
      address: "smtp.example.com"
      port: 587
      domain: "example.com"
      authentication: :login
      user_name: "redmine@example.com"
      password: "your_password"
      enable_starttls_auto: true
```

let me break down each line. `delivery_method: smtp` specifies that we are using the smtp protocol to send email. the `smtp_settings` is where all the heavy lifting takes place: `address: "smtp.example.com"` is the hostname or ip address of your smtp server. if you’re using something like google workspace or microsoft exchange, this will be their respective smtp server address. the `port: 587` is the port usually used for tls/starttls connections. `domain: "example.com"` is the domain you’re using. this may be needed by some smtp servers for authentication. `authentication: :login` this is the type of authentication method used to log into the server. `:login` is probably the most common, but it can be different depending on your server requirements. `user_name: "redmine@example.com"` is the username used to authenticate to the smtp server. this would be typically an email address that’s set up to be allowed to send email.  `password: "your_password"` is the password used for that email address. this is also very important, you must be very careful with this password, store it in a secure place. `enable_starttls_auto: true` enables transport layer security. it is recommended that you set this to `true` because it will automatically try to establish a secure tls connection with the server for security reasons.

now, you may be thinking, "what if i'm using a different authentication method?". well, if your smtp server uses a different auth method (like `:plain` or `:cram_md5`), you would change the `authentication:` line accordingly. the email username and password usually should still be used in all of the scenarios. that specific configuration above has worked for a wide variety of scenarios for me, from local testing to production environments. i even once spent a whole afternoon trying to figure out why my emails were getting rejected only to find out the password had expired - the lessons we learn as we go.

but, that's not the only possible configuration and you could use an email server that needs a specific port for secure connections and a different kind of authentication. here is an alternative configuration for that kind of scenario:

```yaml
production:
  email_delivery:
    delivery_method: smtp
    smtp_settings:
      address: "secure.smtp.provider.net"
      port: 465
      authentication: :plain
      user_name: "user@yourdomain.org"
      password: "another_password"
      enable_ssl_auto: true
```

notice this uses `port: 465` which is the default for ssl encrypted connections and we are using `:plain` authentication instead. the `enable_ssl_auto: true` also signals an ssl connection using the port `465`.

after making changes to `configuration.yml`, you need to restart your redmine server for the new settings to take effect. how you do this depends on how you deployed redmine. usually, that would involve restarting the rails server. i have to restart the service of apache or nginx depending on what i had installed in the server.

let's talk about testing your setup. the easiest way is to use the redmine admin interface itself. log in as an administrator, navigate to "administration" -> "settings" -> "email notifications", and then you'll see a button to send a test email. if you receive that email, congratulations, your settings are correct and you can go home. if you don’t, it’s time for some troubleshooting. this is also the moment that i want to share with you, that once i was stuck in a loop of configuration without noticing that the firewall was blocking the traffic.

another important thing to remember is to make sure your redmine user has the ability to send emails. if the user that is used in the `user_name` in the configuration has any restriction you will get errors that are not obviously tied to a bad configuration file.

also if you have problems in the future with mail not reaching the inbox of the receivers check the email spam folder. i have seen emails not being correctly marked and go directly into the spam folder.

the error logs for redmine can also be super helpful. the file is usually located at `redmine\log\production.log`. errors with emails will show up here and they will give you more information for troubleshooting. i would usually start looking into this log in the first instance, i mean it is what i would advise.

and remember, never ever store your passwords in plaintext inside a file, use environment variables instead. for testing, i have used those methods, but for production, you should change to use environment variables or other more secure ways. here is a brief example of how to do it:

first you need to set the variables in windows, for example: `REDMINE_SMTP_PASSWORD = mysecretpassword` and then the `configuration.yml` will change in this way:

```yaml
production:
  email_delivery:
    delivery_method: smtp
    smtp_settings:
      address: "smtp.example.com"
      port: 587
      domain: "example.com"
      authentication: :login
      user_name: "redmine@example.com"
      password: <%= ENV['REDMINE_SMTP_PASSWORD'] %>
      enable_starttls_auto: true
```

now your password will be stored in a more secure place, this is important because the `configuration.yml` could be accessed by other users or be compromised by some attackers. so be always aware of security.

for deeper understanding of the underlying technologies, i'd recommend a few resources. first, check out "the rails 7 way" by obie fernandez, it's a good start for all things related to rails. also, "programming ruby 1.9 & 2.0" by david thomas, you'll have an excellent understanding of the ruby language. for the smtp protocol, the original rfc 5321 document can be overwhelming but it’s the definite guide to smtp. and don’t forget the documentation of your specific email service, those are invaluable when troubleshooting.

i once spent 3 days figuring out an email issue, only to realize the server ip was on a blacklist. so you can understand my pain with this.

so, in summary, configure the `configuration.yml` file with your smtp details, restart redmine, test it thoroughly, check the logs if needed, and you should be good to go.
