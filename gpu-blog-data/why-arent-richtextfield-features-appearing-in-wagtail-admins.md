---
title: "Why aren't RichTextField features appearing in Wagtail admin's Watail edit form?"
date: "2025-01-30"
id: "why-arent-richtextfield-features-appearing-in-wagtail-admins"
---
The absence of RichTextField features within the Wagtail admin's edit form typically stems from a mismatch between the RichText field's configuration in your Wagtail models and the features enabled within your frontend JavaScript setup, specifically the configuration of the rich text editor used (e.g., Draftail).  I've encountered this issue numerous times during my work on large-scale Wagtail projects, often tracing the root cause to overlooked settings or incorrect feature flag deployment.

**1. Explanation:**

Wagtail leverages a decoupled approach for its rich text editing.  The backend (Django and your models) defines the *structure* of the RichTextField, specifying its allowed content types.  However, the *rendering* and *editing* experience are predominantly managed by a JavaScript-based rich text editor, usually Draftail, integrated into the Wagtail admin. This separation necessitates careful consideration of both sides to ensure consistency.

Problems arise when there's a discrepancy. For example, your model might allow embedding videos, but if the Draftail configuration doesn't include the necessary plugin or feature flag, the respective editing toolbar button won't appear, preventing users from adding videos.  Similarly, if you've customized the Draftail configuration to only permit certain features (e.g., bold, italic, headings), any features not explicitly permitted won't be accessible in the admin interface.  Furthermore, issues can also stem from incorrect or outdated JavaScript files loaded, particularly when implementing custom features or integrating third-party libraries.  Lastly, a cache problem within your browser or Wagtail's caching mechanism can sometimes mask these underlying issues, creating the appearance of missing functionality where none truly exists.

**2. Code Examples and Commentary:**

**Example 1: Missing Feature due to Incorrect Model Definition:**

```python
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.fields import RichTextField
from wagtail.core.models import Page

class MyPage(Page):
    body = RichTextField(blank=True, features=['bold', 'italic'])  # Missing 'h1', 'h2' etc.

    content_panels = Page.content_panels + [
        FieldPanel('body'),
    ]
```

In this case, if you expect headings (h1, h2, etc.) to be available, they're explicitly omitted from the `features` argument within the `RichTextField`.  Consequently, these formatting options won't appear in the editor, even if the Draftail configuration permits them. The solution involves adding the desired features to the `features` list, for example: `features=['bold', 'italic', 'h1', 'h2', 'link']`. Remember that this list is influenced by the available features in your `DRAFTAIL_FEATURES` settings.

**Example 2: Feature Flag Mismatch in Draftail Configuration:**

If you're extending Draftail’s functionality via custom blocks or features, ensure their corresponding settings are correctly configured within your `settings.py`. This often involves defining a `DRAFTAIL_FEATURES` dictionary:

```python
# settings.py
DRAFTAIL_FEATURES = {
    'h1': True,
    'h2': True,
    'image': True,
    'video': True,
    'embed': True,  #Enables general embedding
    'my_custom_block': True  #For a custom block
}
```

Failure to enable a feature, such as `video` here, will prevent the video embedding functionality from working in the editor, even if your Wagtail model allows it.  Note that the keys here must directly match the keys used to register the blocks in your Wagtail models.  Any mismatch will prevent the block's associated features from working as expected in the rich text editor.

**Example 3: JavaScript Loading Issues (Customizations):**

Assuming you’ve added a custom JavaScript block or extended Draftail with a new feature, it's crucial to confirm that your custom JavaScript files are correctly included and loaded.  A common mistake is misconfiguring the loading of these files within your Wagtail project's base templates.

```javascript
// custom_draftail_extension.js
// ... (your custom Draftail code) ...
```

This JavaScript file must be correctly added to your Wagtail templates, ideally within the `<head>` section. Inconsistent or incorrect loading paths can prevent the new functionality from functioning within the rich text editor. This problem requires careful examination of your Wagtail template inheritance and JavaScript inclusion strategy, ensuring your custom file is correctly added and prioritized relative to Wagtail's core JavaScript assets.  If you have multiple JS bundles, confirming the correct bundle is being served is essential; a minor misconfiguration can cause functionality to disappear.

**3. Resource Recommendations:**

I recommend reviewing the official Wagtail documentation pertaining to RichTextFields and Draftail.  Pay close attention to the sections detailing the `features` argument of `RichTextField`, the `DRAFTAIL_FEATURES` setting, and the process for extending Draftail's functionality.   Examining the source code of Draftail itself, and related Wagtail components can also be invaluable for more complex troubleshooting.  Finally, thoroughly debug your JavaScript and frontend interactions to pinpoint specific errors if the backend configuration seems correct.  Using your browser's developer tools will be indispensable in isolating JavaScript issues.  Checking your server logs for any exceptions related to JavaScript resource loading or execution will prove beneficial as well. Remember to clear your browser's cache and your Wagtail's cache after making any changes to resolve persisting issues.
