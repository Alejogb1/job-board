---
title: "How do I configure SilverStripe's email sending using Symfony Mailer?"
date: "2024-12-23"
id: "how-do-i-configure-silverstripes-email-sending-using-symfony-mailer"
---

,  I recall a particularly sticky situation back in '19, when we were migrating a sizable SilverStripe platform to a containerized environment. The default email configuration was… let's just say it wasn't scaling with the new infrastructure. We ended up ditching the built-in `Email` class entirely and going full Symfony Mailer. It was a bit involved, but absolutely worth it. Let's walk through how you can configure SilverStripe to leverage Symfony Mailer.

The key here is understanding that SilverStripe's `Email` class is essentially an abstraction. To replace it, we need to essentially bypass that abstraction and inject Symfony Mailer services directly. It’s a shift from relying on SilverStripe’s internal mechanisms to a more direct integration with a more robust email library. This gives us much more control and flexibility, particularly around features like transport strategies, templating, and more complex email composition.

First, we need the `symfony/mailer` package installed. You’ll do this via composer:

```bash
composer require symfony/mailer
```

Now, let’s dive into the code changes. We'll start with creating a service configuration to define our mailer instance. In SilverStripe, services are typically managed via YAML files located under the `app/_config` directory. Create a file, for instance `app/_config/mailer.yml`, and put the following into it:

```yaml
---
Name: mailer_config
After:
  - '#framework'
---
SilverStripe\Core\Injector\Injector:
  Symfony\Component\Mailer\Transport:
    class: Symfony\Component\Mailer\Transport\Smtp\SmtpTransport
    constructor:
      - '%env(MAILER_DSN)%'
  Symfony\Component\Mailer\Mailer:
    constructor:
      - '%Symfony\Component\Mailer\Transport'
  App\Services\MailerService:
    constructor:
      - '%Symfony\Component\Mailer\Mailer'
```

This configuration defines three services. `Symfony\Component\Mailer\Transport` configures the transport strategy, which in this case is SMTP. Note the `%env(MAILER_DSN)%`. This is a powerful feature of Symfony's configuration system. You should define a variable in your `.env` or `.env.local` file as `MAILER_DSN`. This would resemble something like `smtp://user:password@host:port?encryption=tls&auth_mode=login` This approach allows for configuration to be managed external to the codebase, and allows for more secure storage of credentials.

The second service `Symfony\Component\Mailer\Mailer` sets up the mailer instance using the previously defined transport. Finally, the third service `App\Services\MailerService` creates an abstraction of the Symfony mailer to be used within our SilverStripe application.

Now, let’s create our `MailerService` class:

```php
<?php

namespace App\Services;

use Symfony\Component\Mailer\MailerInterface;
use Symfony\Component\Mime\Email;
use SilverStripe\Core\Config\Config;

class MailerService
{
    private MailerInterface $mailer;

    public function __construct(MailerInterface $mailer)
    {
        $this->mailer = $mailer;
    }

    public function sendEmail(string $to, string $subject, string $body, string $from = null): void
    {
        $email = (new Email())
            ->to($to)
            ->subject($subject)
            ->html($body)
            ->from($from ?: Config::inst()->get('SilverStripe\SiteConfig\SiteConfig', 'email'));

        $this->mailer->send($email);
    }
}

```

This `MailerService` encapsulates the logic for sending emails. It uses the injected `MailerInterface` instance and constructs a Symfony `Email` object with the specified parameters, along with a default `from` address from SilverStripe’s site configuration. You'll note that it uses the `html()` method here, which is important because most modern emails use html for formatting. If you wanted to send plain-text emails you would want to use `text()`.

Now, let's see how you would use this in a SilverStripe DataObject. I'm going to use the concept of an contact form submission as an example, but you can adapt this to whatever data object or controller you require.

```php
<?php
namespace App\Models;
use SilverStripe\ORM\DataObject;
use SilverStripe\Forms\FieldList;
use SilverStripe\Forms\TextField;
use SilverStripe\Forms\TextareaField;
use App\Services\MailerService;
use SilverStripe\Core\Injector\Injector;

class ContactSubmission extends DataObject {
    private static $table_name = 'ContactSubmission';

    private static $db = [
        'Name' => 'Varchar(255)',
        'Email' => 'Varchar(255)',
        'Message' => 'Text'
    ];

    public function getCMSFields()
    {
        $fields = FieldList::create(
            TextField::create('Name', 'Your Name'),
            TextField::create('Email', 'Your Email'),
            TextareaField::create('Message', 'Your Message')
        );
         return $fields;
    }

    public function onAfterWrite()
    {
      parent::onAfterWrite();
      $mailer = Injector::inst()->get(MailerService::class);
      $body = 'A new message has been received from '. $this->Name . ', email address '. $this->Email . '. Here is the message: '. $this->Message;
      $mailer->sendEmail(
          Config::inst()->get('SilverStripe\SiteConfig\SiteConfig', 'email'),
          'New Contact Form Submission',
          $body
      );
    }

}
```

In this DataObject, when a record is created we fetch the `MailerService` from the injector, then we construct the email message, and send using our custom service.

This approach has several benefits. Firstly, Symfony Mailer offers far more flexibility and customization options, like handling attachments, embedding images, and using different transport protocols (e.g., SendGrid, Mailgun). The configuration is centralized, avoiding scattering email settings throughout your codebase. Also, you gain finer-grained control of the transport layer, giving you significantly better error handling and debugging capabilities.

For deeper understanding, I'd highly recommend a couple of resources. Start with the official Symfony documentation on Mailer. It's exceptionally detailed and covers a wide range of topics, from basic usage to advanced techniques. Specifically, focus on the "Transport" and "Email" components. Additionally, the book "Symfony: The Fast Track" by Fabien Potencier (the creator of Symfony) is an excellent resource to get a grasp on the architectural principles behind Symfony.

This integration provides a much more maintainable and reliable email system than the default implementation. It does involve a deeper dive into the frameworks and dependency injection but brings significant benefits. Let me know if you have follow up questions.
