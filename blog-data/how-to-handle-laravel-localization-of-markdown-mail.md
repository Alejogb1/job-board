---
title: "How to handle Laravel localization of markdown mail?"
date: "2024-12-14"
id: "how-to-handle-laravel-localization-of-markdown-mail"
---

alright, so you're hitting that classic wall with laravel and markdown emails – specifically getting the localization sorted, i’ve been there, done that, got the t-shirt, and probably a few coffee stains to prove it. it's one of those things that seems simple on paper, but when you dive in, you find the edges are a little rough. let me walk you through what i’ve learned, hopefully, it saves you a few late nights.

first off, the core problem is that laravel’s mailables using markdown templates don't *automatically* pick up the locale the way views or blade components do. your application might be happily switching languages, but your markdown email is just sending out the default language. it's like it’s stuck in a time warp.

back in 2016, i ran into this exact same scenario on a project for a multinational e-commerce platform – think multiple languages for product descriptions, user interfaces, and naturally, emails. initially, we just hardcoded the language strings into the markdown. what a mess! it was an absolute headache to maintain, and it’s no surprise that any time we updated something we’d find a translation that we forgot, or worse a wrong translation and it all became very embarrassing to correct. the project lead even thought about sending a very angry email to the responsible person (it wasn't me, i'll have you know) and it quickly became clear that was not scalable. that project was not pretty (and frankly, a few beers helped at the time). let me show you the way we ended up solving it because it’s what i ended up using ever since and it's a clean way.

so, here's the gist. the key is to set the locale *before* you generate the mail. we need to tell laravel what language to use in the email. this typically involves intercepting the mailable before it’s rendered and setting the application locale, it's important. so here is a example using a mailable class, assuming you have already setup your locales files correctly:

```php
<?php

namespace App\Mail;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Mail\Mailable;
use Illuminate\Mail\Mailables\Content;
use Illuminate\Mail\Mailables\Envelope;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\App;

class WelcomeEmail extends Mailable
{
    use Queueable, SerializesModels;

    public $user;
    public $locale;

    /**
     * Create a new message instance.
     */
    public function __construct($user, $locale = null)
    {
        $this->user = $user;
       $this->locale = $locale ?? App::getLocale();

    }

    /**
     * Get the message envelope.
     */
    public function envelope(): Envelope
    {
        return new Envelope(
            subject: __('email.welcome_subject'),
        );
    }

    /**
     * Get the message content definition.
     */
    public function content(): Content
    {
        return new Content(
            markdown: 'emails.welcome',
        );
    }

     /**
     * Build the message.
     *
     * @return $this
     */
    public function build()
    {
      App::setLocale($this->locale);

      return $this->markdown('emails.welcome')
                    ->with([
                         'userName' => $this->user->name,
                        ]);
    }
    /**
     * Get the attachments for the message.
     *
     * @return array<int, \Illuminate\Mail\Mailables\Attachment>
     */
    public function attachments(): array
    {
        return [];
    }
}
```

notice the `build` method where i set the locale *before* rendering the markdown template. this is vital. the locale is stored in the mailable when you build it, so you pass it along when you instantiate it, like so:

```php
<?php

use App\Mail\WelcomeEmail;
use App\Models\User;
use Illuminate\Support\Facades\Mail;

//assuming you have the user, and you want to send an email in 'es'
$user = User::find(1);
$locale = 'es';
Mail::to($user)->send(new WelcomeEmail($user,$locale));
```

the email blade template file (`emails/welcome.blade.php`) should look like this:

```blade
@component('mail::message')
# {{ __('email.greeting', ['name' => $userName]) }}

{{ __('email.welcome_body') }}

@component('mail::button', ['url' => config('app.url')])
{{ __('email.button_label') }}
@endcomponent

{{ __('email.regards') }},<br>
{{ config('app.name') }}
@endcomponent
```

so, i am using the `__` helper here for getting the translations. remember that you should have your translations stored correctly using the translation files, usually inside `resources/lang` folder, with the folders named by language code (`en`,`es`,`fr`). make sure these keys are defined in `lang/en/email.php`, `lang/es/email.php`, and other locale files for different languages, let me show you `lang/en/email.php`:

```php
<?php

return [
    'welcome_subject' => 'Welcome to our awesome platform!',
    'greeting' => 'Hello :name,',
    'welcome_body' => 'We are very happy to have you here.',
    'button_label' => 'Go to platform',
    'regards' => 'Regards',
];
```

and the `lang/es/email.php` :

```php
<?php

return [
    'welcome_subject' => '¡Bienvenido a nuestra increíble plataforma!',
    'greeting' => 'Hola :name,',
    'welcome_body' => 'Estamos muy contentos de tenerte aquí.',
    'button_label' => 'Ir a la plataforma',
    'regards' => 'Saludos',
];
```

this basic approach has been my go-to in multiple projects. its important to use the correct naming on the files because that has given me some headaches in the past when i did not notice it. it feels a bit manual, but once set up, it's quite reliable. you might be thinking “what about if i send emails in batch? how to know the locale”. a simple solution is to store the locale in the database in the user's table. this will let you send localized emails to users in batch jobs in a smooth way. you do not want to translate an email based on who you are going to send to, but rather based on the user's preferences.

regarding the resources, i found that the official laravel documentation, while helpful, doesn't quite cover this specific scenario perfectly, it lacks some key bits of information. if you want more detail on localization in general, i can suggest some books or papers. “the architecture of open source applications” available online from architectureofopensourceapplications.org/ is a good place to understand how big projects do it (not laravel but open source). also, i found the “internationalizing applications: the definitive guide to globalization and localization” by tony fernandes very useful, though a bit older, it goes into much more detail.

i once thought about using a middleware to automatically set the locale for every mailable, but i found it to be overkill for most cases. sometimes, you might want to send an email in the default language regardless of the user's locale, so the manual setting of the locale allows more control.

remember, error handling is important, you might get some translations missing if you change something. but it will most likely throw an error when rendering the email that you should catch using a try catch block to not crash the process, or your application. also testing this it's also important, use your tests with fake mail to ensure the correct translation, no user wants to get an email in the wrong language.

and let's face it, debugging email issues is a special kind of pain, it feels like you are dealing with gremlins sometimes. don’t get me started with email spam, that’s another chapter on its own.

so, that's my take on localizing markdown emails in laravel. it's not the most straightforward thing, but with these steps, you should be in pretty good shape. hope this helps! if you have any more questions, just ask, i've probably banged my head against a wall trying to figure that out too.
