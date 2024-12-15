---
title: "How to setup postfix on rails production?"
date: "2024-12-15"
id: "how-to-setup-postfix-on-rails-production"
---

alright, so you're looking to get postfix working with your rails app in production. i've been there, done that, got the t-shirt, and probably a few scars along the way. setting up email on a production server is one of those things that seems straightforward until it isn't. let me walk you through what i've learned, and some gotchas to watch out for.

first off, why postfix? well, it's a solid, reliable mta (mail transfer agent). it's relatively simple to configure compared to some of the alternatives, and it's widely used so you'll find plenty of resources if you get stuck. now, we're talking about production, so we need to think about a few things: security, reliability, and of course, getting it to play nice with rails.

my first major encounter with this was, i'd say, 6 or 7 years ago? i was working on this early-stage startup that was heavily reliant on email for notifications, user registrations, password resets, the whole nine yards. we thought we were being clever by just using a third-party smtp service through our development environment. then, when we flipped the switch to production and we started to get hundreds of users every hour we saw that we were hitting rate limits almost instantly; at some points, users weren't getting verification emails and we started having angry users. that's when i learned that you have to be in control of your own mail server, or at least have a backup. the good thing about that experience was that it pushed me to understand postfix well and not just rely on external services for everything.

the base install of postfix is usually quite straightforward, depending on your linux distro. `apt install postfix` on debian based systems like ubuntu or `yum install postfix` on red hat/centos systems will get you started. the installer will prompt you with a series of questions – make sure to choose 'internet site' as the configuration type if you intend to send emails directly to the internet. if you want to use a 'smarthost' like a service from your domain provider choose 'internet with smarthost'. after the install is done, we need to configure it. the main configuration file is usually `/etc/postfix/main.cf`. here are some key parameters:

```
myhostname = your.domain.com
mydomain = domain.com
myorigin = $mydomain
inet_interfaces = all
mydestination = $myhostname, localhost.$mydomain, localhost
relayhost = [smtp.yourdomain.com]:587 # if you are using an external smtp
smtp_sasl_auth_enable = yes
smtp_sasl_password_maps = hash:/etc/postfix/sasl_passwd
smtp_sasl_security_options = noanonymous
smtp_use_tls = yes
smtp_tls_security_level = encrypt
```

let's break this down, `myhostname` is the fully qualified domain name for your server. `mydomain` is just the domain part. `myorigin` is the address used in the from header of outgoing emails. `inet_interfaces = all` listens on all network interfaces. `mydestination` tells postfix what domains to consider local and receive mail for, and since we want to send mail, we set `relayhost` to the external smtp server if using one, or comment it out to make postfix send mails directly to destination servers. after that, we set the credentials for the smarthost in the `/etc/postfix/sasl_passwd` file with `smtp.yourdomain.com username:password` in it and generate the db file with `postmap /etc/postfix/sasl_passwd`. we specify `smtp_sasl_auth_enable` to `yes`, tell postfix where to find the credentials and other security parameters to make sure to use the external smtp correctly. `smtp_use_tls = yes` and `smtp_tls_security_level = encrypt` tells postfix to encrypt all the smtp connection using tls.

you will need to restart postfix after editing this file using `systemctl restart postfix`. and also, check postfix logs with `tail -f /var/log/mail.log` to see what is going on, and to check for potential errors.

now, rails. rails by default uses `actionmailer` to manage email sending. you’ll want to configure your rails environment to talk to your new postfix setup. in `config/environments/production.rb` (or your specific environment file), you'll have something like this:

```ruby
  config.action_mailer.delivery_method = :smtp
  config.action_mailer.smtp_settings = {
    address: 'localhost',
    port: 25, # or 587 if using an external smarthost
    domain: 'your.domain.com',
    user_name: 'your-username', # required when using an external smarthost
    password: 'your-password',  # required when using an external smarthost
    authentication: 'plain',    # required when using an external smarthost
    enable_starttls_auto: true # required when using an external smarthost
  }
```

important thing here: `address: 'localhost'` means you are telling rails to connect to postfix running locally. `port: 25` is the default smtp port, if you use a smarthost change to `587` and also fill `user_name` and `password` and `authentication` with your smarthost credentials and also `enable_starttls_auto: true`. this configuration should be enough for simple setups. it is also recommended that you have separate users with access only to the mail server. remember to add this info into your env variables if you are not committing secrets into your repository. you'll want to restart your rails application.

here’s where it gets fun. the most painful part for me back then was handling bounces and undeliverable messages. you see, when emails bounce, postfix will attempt to resend them for a while, then it will give up and send a bounce message to the sender (in your case rails). if you don’t have a way to handle those bounce messages, your users will not know why their email is not arriving and the user will think your app is failing or that the email was sent but not delivered. at the time we handled this poorly; i didn't know how to parse the bounce messages and notify users properly. this is why you should use tools like `mailgun` or `sendgrid` that handles all of this automatically for you.

if you want to go full old-school and handle bounces yourself, postfix has a feature called `virtual_mailbox_maps` where you can make all messages get redirected to a virtual address that will be a rails endpoint. the simplest way to handle this is to configure postfix to deliver all mail to a specific account and then have a background job parse the messages and act accordingly. this is what i mean:

in `main.cf` you can add something like

```
virtual_mailbox_domains = domain.com
virtual_mailbox_maps = hash:/etc/postfix/virtual_mailbox
virtual_alias_maps = hash:/etc/postfix/virtual_alias
virtual_minimum_uid = 100
virtual_uid_maps = static:1000
virtual_gid_maps = static:1000
```

and then in the file `/etc/postfix/virtual_mailbox` put this

```
@domain.com  bounce-handler
```

and generate the db file with `postmap /etc/postfix/virtual_mailbox`. then in `/etc/postfix/virtual_alias` put this

```
bounce-handler   bounce-handler@your.domain.com
```

and generate the db file with `postmap /etc/postfix/virtual_alias`. and finally in the rails side in an initializer config file we set this:

```ruby
  config.action_mailer.smtp_settings = {
    address: 'localhost',
    port: 25,
    domain: 'your.domain.com',
    user_name: nil,
    password: nil,
    authentication: nil,
    enable_starttls_auto: false,
    return_path: "bounce-handler@your.domain.com" #important to receive bounce messages
  }
```

this way any email sent that is not from `bounce-handler@your.domain.com` will end in the `bounce-handler` address in your server. you can configure a separate email server for that user and handle all emails there using a background job.

after all this you need to configure a rails endpoint that processes the incoming emails. in your routes file you can do something like this

```ruby
post '/incoming_email', to: 'incoming_email#create'
```

and create a controller with an action like this:

```ruby
  def create
    # here you would receive the raw email, the 'request.body'
    # and you can then process the email in the way you want.
    # for example you can use the 'mail' gem to parse it
    email = Mail.read_from_string(request.body.read)
    # now you can do things like look for error codes
    # or the original receiver of the email.
    # then you can implement logic to notify your users
    # if they have pending delivery errors
    head :ok #important to signal the mail server that it was processed correctly
  end
```

this is a very basic example, you should add security to this endpoint by limiting the ip address or using a shared secret so only your mail server can post to this endpoint, or add authentication. you'll also want to handle edge cases, like invalid email addresses, and have retry logic in place because mail servers can be flaky. i remember once i had a typo in my domain and spent the whole night debugging why my emails were not arriving, which was quite frustrating but it's a good story for the grandkids i guess, i learned to be meticulous.

you will find resources that will describe in much more detail the postfix configuration, one that i recommend is “the postfix complete reference”, it is a good starting point. and for rails, anything related to actionmailer is a good place to start. also, it is important to understand how dns works, things like spf, dkim and dmarc are key for email deliverability. the book "dns and bind" is a good reference for that.

the setup we described here is what i would consider the minimum for a basic production setup. don't cut corners on the security and monitoring. keep a close eye on your mail logs and be prepared to troubleshoot any problems you might find, that is key. i hope it helps.
