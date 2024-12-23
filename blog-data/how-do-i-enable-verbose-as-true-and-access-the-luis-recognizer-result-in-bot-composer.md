---
title: "How do I enable verbose as `true` and access the LUIS recognizer result in Bot Composer?"
date: "2024-12-23"
id: "how-do-i-enable-verbose-as-true-and-access-the-luis-recognizer-result-in-bot-composer"
---

Alright, let's delve into this. I recall wrestling (okay, *engaging*) with this exact scenario back in 2019 while working on a multi-lingual customer service bot for a client. They needed very detailed intent recognition information, well beyond what the standard Composer setup provided. It quickly became apparent that verbose mode was the key, and accessing the LUIS data required a little more plumbing.

The standard Bot Composer interface, by default, abstracts away some of the lower-level details. While beneficial for rapid prototyping, it does mean you need to dive slightly deeper to enable verbose responses and extract the raw LUIS output. The challenge isn’t really about ‘enabling’ a global verbose flag as much as it’s about instructing specific LUIS recognizer instances within your bot to provide that expanded data and then accessing it programmatically.

The core of the matter revolves around how LUIS is configured as a recognizer within Composer and how this configuration is reflected in the resulting bot's code. Essentially, what you are trying to achieve is not a universal "verbose" setting but rather a per-recognizer property that needs to be handled on a case-by-case basis.

The most direct route is to programmatically access the `recognizerResult` within your dialogs after recognition takes place. This is typically available in an `OnIntent` trigger. I'll show you how to access it and use it through three examples.

**Example 1: Accessing raw LUIS result within the dialog context**

Imagine you have a LUIS recognizer named `myLUISRecognizer` in your dialog. Here's how I accessed the raw result:

```csharp
// C# example within a Bot Composer Custom Action
// Assuming the recognizer is already configured and used in the dialog

using Microsoft.Bot.Builder;
using Microsoft.Bot.Schema;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder.Dialogs;
using Newtonsoft.Json;
using Microsoft.Recognizers.Text;


public class LogLuisResult : Dialog
{
    public override async Task<DialogTurnResult> BeginDialogAsync(DialogContext dc, object options = null, CancellationToken cancellationToken = default)
    {
        var recognizerResult = (dc.State.Get<RecognizerResult>("recognizerResult") as RecognizerResult);


        if (recognizerResult != null && recognizerResult.Properties.ContainsKey("luisResult"))
        {
            var luisResult = recognizerResult.Properties["luisResult"] as Newtonsoft.Json.Linq.JObject;

            if (luisResult != null)
            {
                // Serialize to string for display or logging
                var formattedResult = JsonConvert.SerializeObject(luisResult, Formatting.Indented);
                await dc.Context.SendActivityAsync(MessageFactory.Text($"Verbose LUIS Result:\n {formattedResult}"), cancellationToken);

                // Alternatively, process the data as needed
                // e.g., access specific intent scores or entities:
                // var topIntent = luisResult["topScoringIntent"]["intent"];
                // await dc.Context.SendActivityAsync(MessageFactory.Text($"Top Intent: {topIntent}"), cancellationToken);

            } else
            {
              await dc.Context.SendActivityAsync(MessageFactory.Text($"Error: No LUIS result data available."), cancellationToken);
            }
        } else
        {
            await dc.Context.SendActivityAsync(MessageFactory.Text($"Error: No recognizer result or LUIS result found."), cancellationToken);
        }

        return await dc.EndDialogAsync(cancellationToken: cancellationToken);

    }
}

```

*Explanation:*

In this C# custom action, the key line is how we are retrieving the RecognizerResult from the dialog context `dc.State.Get<RecognizerResult>("recognizerResult")` and checking to see if it contains the `luisResult` property which is where the LUIS data lives when `verbose` is enabled. Once we get the data, we can serialize the json result and display in the bot.

**Example 2: Using an Adaptive Expression to access the top intent score**

Another approach, often more integrated into the Bot Composer visual canvas, involves using adaptive expressions to extract data. Here, we would set a property within a dialog using expressions referencing the result.

Inside a composer dialog, we could achieve this by:

1.  Adding an `OnIntent` trigger that matches an intent from `myLUISRecognizer`.
2.  Adding a `Set a property` action within that trigger.
3. In the `Property` input, you could use a name like `dialog.topIntentScore` and for the `Value` input, use this expression:
   `=if(exists(turn.recognized.luisResult.topScoringIntent.score), turn.recognized.luisResult.topScoringIntent.score, 0)`

*Explanation:*

This expression is compact and effective. It checks if the `turn.recognized.luisResult.topScoringIntent.score` exists, and if so, returns it. If not, it returns `0`. This illustrates how to access nested properties within the `luisResult` object using Adaptive Expressions, the core expression language of Bot Framework. The output can be used in further logic, displayed, or even logged.

**Example 3: Logging raw result to Application Insights**

In production, logging is crucial for monitoring and analysis. You can integrate with Application Insights to store the verbose LUIS results for analysis. This requires a custom action in a C# or Python. Here’s a modified version of Example 1 to achieve this:

```csharp
// C# Custom Action to Log LUIS Result to Application Insights
using Microsoft.Bot.Builder;
using Microsoft.Bot.Schema;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder.Dialogs;
using Newtonsoft.Json;
using Microsoft.ApplicationInsights;
using Microsoft.ApplicationInsights.DataContracts;

public class LogLuisResultToAppInsights : Dialog
{
  private readonly TelemetryClient _telemetryClient;

    public LogLuisResultToAppInsights(TelemetryClient telemetryClient)
    {
        _telemetryClient = telemetryClient;
    }

    public override async Task<DialogTurnResult> BeginDialogAsync(DialogContext dc, object options = null, CancellationToken cancellationToken = default)
    {
         var recognizerResult = (dc.State.Get<RecognizerResult>("recognizerResult") as RecognizerResult);
         if (recognizerResult != null && recognizerResult.Properties.ContainsKey("luisResult"))
        {

            var luisResult = recognizerResult.Properties["luisResult"] as Newtonsoft.Json.Linq.JObject;

             if (luisResult != null)
             {
                var formattedResult = JsonConvert.SerializeObject(luisResult, Formatting.None);
                 _telemetryClient.TrackTrace(new TraceTelemetry {
                                                 Message = $"Verbose LUIS Result: {formattedResult}",
                                                 SeverityLevel = SeverityLevel.Information
                });
             } else {
                _telemetryClient.TrackTrace(new TraceTelemetry {
                                                 Message = $"Error: No LUIS result data available.",
                                                 SeverityLevel = SeverityLevel.Warning
                });

             }

        } else {

             _telemetryClient.TrackTrace(new TraceTelemetry {
                                                 Message = $"Error: No recognizer result or LUIS result found.",
                                                 SeverityLevel = SeverityLevel.Warning
                });
        }


        return await dc.EndDialogAsync(cancellationToken: cancellationToken);
    }
}
```

*Explanation:*

This code does similar as example one, instead of displaying the `luisResult` we log it to application insights for analysis and monitoring.

**Important Notes and Best Practices:**

*   **Verbose is Per Recognizer:** Remember, `verbose` is set on a per-recognizer basis within LUIS itself (in the LUIS portal, the application settings, specifically in the publish settings). Ensure it's enabled *there* first.
*   **RecognizerResult Object:** The `recognizerResult` object is your entry point. The `Properties` dictionary usually holds key information that is accessible to the bot during a turn.
*   **Adaptive Expressions:** Get very familiar with Adaptive Expressions. They are the glue that binds Bot Composer together.
*   **Error Handling:** Implement robust error handling. Checks for null values are critical, since `luisResult` may not always exist, and neither does every property within the JSON object.
*   **Performance:** Be mindful that excessive logging may impact performance. Only log what is needed.
*   **Privacy:** Be aware that verbose responses may contain potentially sensitive user data. Be careful how you store, access, and process this information, and ensure it complies with any applicable privacy regulations.

**Recommended Resources:**

*   **"Programming Microsoft Bot Framework v4" by Joe Mayo:** While this book focuses on v4 of the SDK, it provides solid foundation in bot architecture which will benefit you in dealing with Bot Composer.
*   **Microsoft's Official Bot Framework Documentation:** Specifically, sections covering Dialogs, Recognizers, Adaptive Expressions, and Telemetry.
*   **The source code of Microsoft.Bot.Builder libraries**: A deep understanding of the underlying class structures will make it very clear how the code works.

In conclusion, accessing the verbose LUIS result isn't about flipping a single switch. It's about configuring your recognizer in LUIS, ensuring `verbose` is set during publishing, and then leveraging the `recognizerResult` in your dialogs through code or adaptive expressions. The three examples given provide a solid starting point for integrating this into your Bot Composer bot effectively. Take some time to learn about these components and you will quickly master this.
