---
title: "How can Google Tag Manager be used for cross-domain tracking across three sites?"
date: "2024-12-23"
id: "how-can-google-tag-manager-be-used-for-cross-domain-tracking-across-three-sites"
---

Alright, let's tackle cross-domain tracking with Google Tag Manager (GTM) across three sites. This is a topic I've had to implement numerous times, and each scenario presents its own nuanced challenges. It's definitely more involved than single-domain tracking, but with a solid understanding of how GTM manages cross-domain interactions, it's absolutely achievable and manageable.

My past experiences, particularly when I was part of an e-commerce platform migration a few years back, involved exactly this – managing user journeys across three distinct sites: a main marketing site, the core e-commerce platform, and finally, a separate order management portal. It was crucial to maintain a unified user view for accurate analytics and attribution. This is not merely about having numbers; it's about understanding the user's journey end-to-end.

The core problem with cross-domain tracking is the inherent limitation of first-party cookies. These cookies, by default, are only accessible within the specific domain they are set on. Therefore, when a user navigates from `siteA.com` to `siteB.com`, they are treated as a completely new visitor by your analytics system if you’re not managing the handoff. We solve this using a process that appends unique parameters to URLs that facilitate cookie sharing. GTM handles the technical side of this parameter appending and parsing through its Linker feature.

Essentially, the Linker functionality in GTM works by embedding a client identifier (`_ga`, `_gid`) or a specific string within the URL as query parameters. For Google Analytics, GTM specifically uses `_gl` parameter for cross domain linking. This parameter, when the visitor is on a different domain, triggers Google Analytics to recognize the user as the same. However, that's the high-level view; getting it to work correctly requires some configurations.

Let’s consider `site1.com`, `site2.com`, and `site3.com` as our three sites for demonstration.

**Step 1: GTM Container Configuration**

First, I'd set up a single GTM container. For optimal maintainability, I almost always opt for a single container managing all three sites, despite a temptation of three separate ones. The key is meticulous naming conventions and robust version control. This means defining triggers and variables clearly by scope. For example, I'd utilize naming conventions like `site1_event_trigger` or `site2_page_load`.

**Step 2: Setting up the Google Analytics Tags**

Within our single GTM container, we configure our Google Analytics (GA4 or Universal Analytics) tags. Make sure to utilize a Google Analytics Settings variable that holds your tracking ID. Importantly, in the Google Analytics Settings variable configuration, in the ‘Fields to set’ section, we'll be specifying the `allowLinker` setting and setting it to `true`. This is pivotal. Without this flag, cross-domain tracking won’t function. Furthermore, we'll set the `cookieDomain` value. When the default setting is `auto`, Google Analytics will typically find the correct cookie domain to use. But we must double check and make sure that all three sites are not working on subdomains of the same parent domain.

```javascript
// Example Settings Variable Configuration (JSON representation)
{
    "trackingId": "UA-XXXXX-Y", // Or G-XXXXXXXX
    "fieldsToSet": [
        {
            "fieldName": "allowLinker",
            "value": true
        },
    	{
            "fieldName": "cookieDomain",
            "value": "auto" // For handling sub-domains
        }

    ]
}
```

**Step 3: Linker Settings Configuration**

This is where the magic happens. Navigate to the container settings in GTM, and under the "Configure container settings," find "Domains." Here, we list all domains participating in the cross-domain setup: `site1.com`, `site2.com`, and `site3.com`. By listing these here, GTM knows which domains to listen to for incoming linker parameters and where to append linker parameters to outgoing links. GTM’s linker will automatically detect when users navigate to any of these other domains, appends the linker parameter to the URL of links on your website, and upon arrival on the destination domain, will parse the link and maintain the same session.

**Step 4:  Cross-Domain Form Submission**

If you use forms to move from one domain to another, it’s also important to configure them to use the linker. We’d typically achieve this by creating a custom html tag and adding javascript code to our forms. Below is the code for the custom tag:

```html
//  Example custom HTML tag
<script>
  var form = document.querySelectorAll('form'); // Replace with your appropriate selectors

  form.forEach(function(form) {
     form.addEventListener('submit', function(event) {

          var linkerParam =  ''; // Store the linker parameters
          var url = this.action;


        	  gtag('get', 'linker', function(linkerParam) {

            // Handle the situation where there's a fragment identifier
           if(url.includes('#')){
                var urlSplit = url.split('#');
                var urlToAppend =  urlSplit[0];
                var urlFragment = '#' + urlSplit[1];
                url = urlToAppend + '?' + linkerParam + urlFragment ;
             } else {
                 url = url + '?' + linkerParam;
             }


          });

        this.action = url
      });
   });
</script>

```

In this example we grab all the forms on a page, listen to submit events and modify form action to add the appropriate linker parameter before the form is submitted.

**Step 5: Testing and Validation**

Testing is critical. I use GTM's preview mode extensively. I’d navigate through all three sites, observing the network requests via the browser’s developer tools. Here, I verify that the `_gl` parameter is present and consistent across the domains. Furthermore, I'd examine the reports in Google Analytics, especially the user flow and path analysis to ensure that sessions are indeed being stitched together correctly.

**Technical Considerations:**

* **Canonical Tags:** Ensure canonical tags are correctly implemented on each page. Incorrect canonical tags can mislead the crawler regarding your true page url and invalidate cross domain tracking.
* **Subdomains:** If any of the three sites are subdomains of a single root domain, you might not need cross-domain tracking; however, it's always worth double-checking to be sure.
* **Referral Exclusion List:** In GA4 and Universal Analytics, add all three domains to the Referral Exclusion List. This stops traffic from each domain being classified as a referral from each other. This keeps attribution clear and accurate.
* **User Identification:** Remember that simply tracking across domains doesn't necessarily identify users in a personalized manner. You’d need to implement additional strategies (e.g., user authentication) to link users across sessions.

**Recommended Reading:**

For a deeper understanding of this topic, I'd suggest delving into the following:

1.  **Google Analytics Documentation:** The official documentation on cross-domain tracking is comprehensive and frequently updated. It's a must-read for this task. I’d focus specifically on sections pertaining to `_gl`, the `allowLinker` parameter, and the Referral Exclusion List.
2.  **"Google Tag Manager for Developers" by Christopher S. Penn:** This book covers GTM from a more technical perspective. It explores custom implementations beyond the typical use cases, such as our use case of form tracking.
3.  **The Google Analytics Developer Guides:** The official Google Analytics API guides are very useful for troubleshooting. Specifically, familiarize yourself with the data collection methods and the structure of tracking parameters.
4. **"Advanced Web Metrics with Google Analytics" by Brian Clifton** This book offers insights into more sophisticated Google Analytics techniques, including the nuanced data collection methods essential to correctly set up Google Tag Manager tracking.

Implementing cross-domain tracking, particularly across three different sites, requires careful planning, a meticulous approach to configuration, and diligent testing. It's essential to focus on the underlying principles of how analytics systems handle sessions and cookies. By employing GTM’s Linker functionality with the correct settings, and paying close attention to the recommended reading materials, you can achieve a robust and accurate view of your user journeys across multiple domains. Remember, it’s not just about the numbers but the insights they provide that inform better decision-making.
