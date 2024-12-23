---
title: "How do I receive Mailgun responses using Laravel Mail?"
date: "2024-12-23"
id: "how-do-i-receive-mailgun-responses-using-laravel-mail"
---

, let's talk about handling Mailgun responses within a Laravel application. I've spent quite a bit of time navigating the intricacies of email delivery, including the various webhooks and tracking mechanisms that services like Mailgun provide. Getting it configured properly in Laravel isn't always straightforward, but once you understand the underlying principles, it becomes quite manageable.

From my own experiences, I recall building a rather complex notification system where we needed precise feedback on email delivery, opens, and clicks. We weren’t just shooting emails into the void; we had to understand *exactly* how each email performed. That’s where efficiently managing Mailgun's responses within the Laravel ecosystem really came to the forefront. The core of this is leveraging Mailgun’s webhooks and processing the incoming data effectively.

The first thing to recognize is that Mailgun doesn’t just "respond" to a Laravel email send in a synchronous way. The typical `Mail::to($recipient)->send($mailable)` function call doesn't directly receive delivery details back; it primarily validates if Mailgun *accepted* the email for sending. The magic of delivery tracking, opens, clicks, etc., happens asynchronously via webhooks. Mailgun essentially sends an http post request to a specified url in your application whenever an event occurs for an email you sent.

To handle these responses, you must expose an endpoint in your Laravel application and configure Mailgun to send webhooks to that specific url. Let me clarify how I typically approach this, using some illustrative code.

**1. Setting Up the Laravel Route and Controller:**

First, I create a dedicated route in `routes/web.php` (or, if your application is more involved you may want to group under a specific prefix, particularly if security is a concern as you’ll see in a moment) to receive webhook events:

```php
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\MailgunWebhookController;

Route::post('/mailgun/webhook', [MailgunWebhookController::class, 'handleWebhook']);
```

Here, we're mapping a `POST` request to the `/mailgun/webhook` url to the `handleWebhook` method in a new controller I'll create next called `MailgunWebhookController`.

Now, let's build the controller itself in `app/Http/Controllers/MailgunWebhookController.php`:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class MailgunWebhookController extends Controller
{
    public function handleWebhook(Request $request)
    {
      Log::info("Mailgun webhook received", $request->all());

      //Here's where the detailed event processing goes.

      return response('Webhook received', 200);
    }
}
```

This controller's `handleWebhook` method is where you'll receive and process all Mailgun webhook data. I'm logging all the incoming request information for now, so you can inspect it (check your Laravel logs!). You'll need to actually process the data, but for the purposes of showing how it works and how to get the information, I will leave that out for now. Crucially, you must return a 200 status code, as mailgun uses this to ensure that the endpoint is indeed active and listening, anything else and it may re-send the request.

**2. Securing the Endpoint (Very Important):**

Mailgun provides two mechanisms for verifying webhook requests, these are, **timestamps** and **signing keys**. A timestamp method relies on the time sent by Mailgun falling within a certain tolerance (5 minutes by default) to ensure the request is genuine. This is vulnerable to replay attacks, however, where the request is intercepted and re-sent at another time. As a result, you should always verify webhook requests using a **signing key**.

Mailgun provides a signature hash for each webhook request, which you must use to verify the validity of that request. Here's how to integrate that into our controller method:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Config;
use Symfony\Component\HttpKernel\Exception\AccessDeniedHttpException;

class MailgunWebhookController extends Controller
{
    public function handleWebhook(Request $request)
    {
        $signature = $request->input('signature');
        $timestamp = $request->input('timestamp');
        $token     = $request->input('token');

        $signingKey = Config::get('services.mailgun.webhook_signing_key');

        if (!$signature || !$timestamp || !$token || !$signingKey) {
            throw new AccessDeniedHttpException('Missing parameters to verify Mailgun webhook request.');
        }

        $computedSignature = hash_hmac('sha256', $timestamp . $token, $signingKey);

       if (!hash_equals($signature, $computedSignature)) {
            Log::error('Mailgun webhook signature verification failed', ['expected' => $computedSignature, 'received' => $signature, 'request' => $request->all()]);
            throw new AccessDeniedHttpException('Invalid Mailgun webhook signature.');
        }


        Log::info("Mailgun webhook received", $request->all());

        //Here's where the detailed event processing goes.

        return response('Webhook received', 200);
    }
}
```

In this version, I've added signature verification logic using the `hash_hmac` and `hash_equals` functions. This computes a signature based on the request timestamp and token along with a configured `webhook_signing_key` taken from the `services.mailgun.webhook_signing_key` config. You'll need to add this value in your `config/services.php` file, taking it from your Mailgun dashboard. If the signature doesn’t match, it throws an AccessDeniedHttpException, preventing a malicious request from being processed. This provides robust security for your webhook endpoint.

**3. Processing Specific Webhook Events:**

Now that you're successfully receiving webhooks and validating them, let’s talk about what to do with them. The payload sent by Mailgun will vary depending on the event. It could include events like:

*   `delivered`: Email successfully delivered.
*   `opened`: Email opened by the recipient.
*   `clicked`: Link in the email was clicked.
*   `failed`: Delivery failure.

Your controller needs to inspect the `event` value within the data and handle it accordingly. In my past systems, I’d typically set up specific handlers, like so:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Config;
use Symfony\Component\HttpKernel\Exception\AccessDeniedHttpException;
use App\Jobs\ProcessMailgunDeliveryEvent;
use App\Jobs\ProcessMailgunOpenEvent;
use App\Jobs\ProcessMailgunClickEvent;
use App\Jobs\ProcessMailgunFailureEvent;


class MailgunWebhookController extends Controller
{
    public function handleWebhook(Request $request)
    {
        $signature = $request->input('signature');
        $timestamp = $request->input('timestamp');
        $token     = $request->input('token');

        $signingKey = Config::get('services.mailgun.webhook_signing_key');

        if (!$signature || !$timestamp || !$token || !$signingKey) {
            throw new AccessDeniedHttpException('Missing parameters to verify Mailgun webhook request.');
        }

        $computedSignature = hash_hmac('sha256', $timestamp . $token, $signingKey);

        if (!hash_equals($signature, $computedSignature)) {
            Log::error('Mailgun webhook signature verification failed', ['expected' => $computedSignature, 'received' => $signature, 'request' => $request->all()]);
            throw new AccessDeniedHttpException('Invalid Mailgun webhook signature.');
        }

        Log::info("Mailgun webhook received", $request->all());


        $event = $request->input('event');

        switch ($event) {
            case 'delivered':
                ProcessMailgunDeliveryEvent::dispatch($request->all());
                break;
            case 'opened':
               ProcessMailgunOpenEvent::dispatch($request->all());
                break;
            case 'clicked':
                ProcessMailgunClickEvent::dispatch($request->all());
                break;
            case 'failed':
                ProcessMailgunFailureEvent::dispatch($request->all());
                break;
            default:
                Log::warning("Unknown mailgun event type", ['event'=>$event, 'request'=>$request->all()]);

        }

        return response('Webhook received', 200);
    }
}
```

Here, I'm switching based on the event received and dispatching a Laravel job with all the request data for specific handling. This is far more reliable, particularly if your application does any serious processing on those events, or communicates them to third parties. We're keeping the webhook handling lightweight, and offloading any work to queues.

For more information and a deeper understanding of webhooks and email protocols, I'd highly recommend checking out **"High Performance Web Sites" by Steve Souders**, though it doesn’t focus directly on Mailgun it does cover key web performance considerations which are relevant to effective webhook handling. Additionally, for a comprehensive overview of email infrastructure and protocols, I'd recommend *“Email: From Production to Protocol"*. That’s not a single book, but many papers and standards collected under this general topic. It's far more detailed and will provide a deep and thorough understanding of email fundamentals that helps contextualize the workings of services like Mailgun. I'd also recommend investigating the official Mailgun documentation which contains detailed information, particularly surrounding best practices with webhooks. Finally, it's good practice to frequently check Mailgun's changelog as there may be important information there that could impact your implementation.

Remember, handling webhooks effectively is crucial for maintaining robust applications that react intelligently to email delivery events. I've found this method of segregating the event handling into dedicated jobs to be far more resilient, maintainable and easier to manage, and it also ensures that your web server remains responsive by not blocking on potentially long-running processes. I hope these practical examples and explanations help you effectively handle Mailgun responses in your Laravel projects.
