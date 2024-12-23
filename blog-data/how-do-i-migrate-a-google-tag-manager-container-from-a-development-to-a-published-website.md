---
title: "How do I migrate a Google Tag Manager container from a development to a published website?"
date: "2024-12-23"
id: "how-do-i-migrate-a-google-tag-manager-container-from-a-development-to-a-published-website"
---

Alright, let's tackle this migration of a Google Tag Manager (GTM) container. I've been through this process more times than I care to remember, and while it might seem straightforward, there are a few critical points that often get overlooked, leading to headaches down the line. The core challenge isn't just moving the container; it's ensuring a seamless transition without breaking existing tracking or data collection.

First, let's clarify the context. We're moving a GTM container from a development environment – let's assume this is a staging site – to a production website. The primary goal is to replicate the *exact* setup – the tags, triggers, and variables – without introducing any unexpected behavior in production. I’ve seen scenarios where forgetting a single trigger configuration or a variable change has had significant impacts on critical metrics, so precision is paramount.

The key approach isn't just copying and pasting configurations; instead, we will leverage GTM's built-in import/export functionality. It’s significantly more reliable and less prone to errors than trying to manually reconstruct everything. This function exports a json file representation of your container, which can then be imported. Before we dive in though, it's essential to establish a checklist. My typical process, refined over multiple projects, usually looks something like this:

1.  **Snapshot of Production:** Before you even touch the development container, take a snapshot of the *production* container. This involves exporting the container configuration. This serves as your baseline, a point of return if things go sideways. You can do this through the 'Admin' tab in GTM, then 'Export Container.' Name this export descriptively; something like "Production_PreMigration_[Date]" is usually good.

2.  **Comprehensive Review:** Examine your development container with painstaking detail. Make sure all the tags you intend to move are in the correct *state*, e.g. are enabled and using appropriate variables. Critically look at any dependencies such as custom templates and ensure these are compatible and ready. Ensure that all variables, especially those that rely on custom scripts, are validated and work as expected in your development environment. This also includes confirming naming conventions are consistent; I've had problems in the past with case sensitivity or even minor misspellings impacting functionality.

3.  **Export the Dev Container:** Once you're happy with the review, export the development container, again, using a descriptive name such as "Development_PreMigration_[Date]".

4.  **Import and Verify:** In the production GTM environment, create a new version or, if the container is totally new, an empty container. If you’re using a pre-existing container, it’s a *best practice* to create a new version *before* importing. This acts as a safety net, allowing you to rollback to the previous version quickly if there are issues. Import the exported development container. Immediately after importing *do not* publish. Instead, meticulously compare the imported settings to the desired configuration by manually checking a few key tags and triggers. Look for any discrepancies.

5.  **Preview and Debug:** Thoroughly test on staging *and* production, use the preview mode in GTM to see what data is being collected. Specifically, examine tags that fire on key pages or user interactions. Use your browser’s developer tools (network tab and console) to observe actual data being transmitted. Address any problems that surface.

6.  **Publish with Caution:** Once you've verified and are confident with the imported settings, publish the new version. Monitor all of your dashboards and logging. Remember, even with perfect execution, unexpected edge cases can appear on production environments due to user behaviour patterns not seen during development.

Let me show you some code, not to execute *in* GTM, but to illustrate the types of variable configurations that are vital to verify. These examples, while written in JavaScript for clarity, represent configurations I often find myself examining.

```javascript
// Example 1: Data Layer Variable Configuration

// In GTM: a variable named 'productName' using the 'Data Layer Variable' type,
// referencing the 'product.name' key
// Example usage (in HTML or a JavaScript function) pushing data into the dataLayer:
dataLayer.push({
  event: 'productView',
  product: {
    name: 'Awesome Widget',
    price: 29.99
  }
});

// Validating this variable configuration would ensure that the variable correctly fetches 'Awesome Widget'
// Checking in GTM: This would appear when previewing a page with this datalayer, confirm its value on your staging website.
```

```javascript
// Example 2: Custom JavaScript Variable Configuration

// In GTM: A variable named 'userAgentPlatform' using the 'Custom JavaScript' type:
function() {
  var userAgent = navigator.userAgent;
  if (/Android/.test(userAgent)) {
    return 'Android';
  } else if (/iPhone|iPad|iPod/.test(userAgent)) {
    return 'iOS';
  } else {
    return 'Desktop';
  }
}

// Validating this ensures this variable returns the expected values ('Android', 'iOS', 'Desktop') across various devices.
//Checking in GTM: During preview you will see the value calculated. Test across a variety of devices.
```

```javascript
// Example 3: Lookup Table Variable Configuration
// In GTM: a variable named 'conversionAction' using the 'Lookup Table' type:

// Input Variable: 'event'
// Mapping:
//   'purchase' --> 'Purchase Complete'
//   'addToCart' --> 'Added to Cart'
//   'viewProduct' --> 'Product Viewed'

//Checking in GTM: Preview mode would confirm the values being returned. You could also check the data layer to make sure the correct value is available for the mapping.

//The corresponding dataLayer event would have a value like 'purchase', 'addToCart', or 'viewProduct'
dataLayer.push({
 event: 'purchase',
 ... // other event specific data.
});

// Validation ensures the correct mapping based on event triggers.
```

These examples, while simple, highlight the common places that require scrutiny during a migration. Variables like these are the lifeblood of your tracking setup. For more complex scenarios, particularly when using custom templates, it's good practice to thoroughly test the underlying template logic and variable functionality on staging before production deployment.

A critical resource for deepening your knowledge on container management would be the "Google Tag Manager Fundamentals" course on Google Skillshop. While not strictly a book, it offers in-depth practical insights. Also, a deeper dive into the nuances of JavaScript and Data Layer implementation, I recommend "Eloquent JavaScript" by Marijn Haverbeke. It’s not specific to GTM, but a foundational text for this kind of work. Additionally, the official Google Tag Manager documentation is crucial, it's continuously updated, and provides the most current information regarding changes and best practices within the platform.

The approach I've outlined ensures minimal disruption, minimizes the risk of errors, and facilitates a far more controlled transition. Remember, moving a GTM container shouldn't be a nerve-wracking event, but rather a managed process. By understanding the details, employing robust testing, and always having a fallback, you'll ensure a smooth migration.
