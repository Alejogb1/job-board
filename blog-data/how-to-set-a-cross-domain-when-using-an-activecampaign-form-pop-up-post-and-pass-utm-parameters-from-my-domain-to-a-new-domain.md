---
title: "How to set a cross domain when using an ActiveCampaign form pop-up (POST) and pass UTM parameters from my domain to a new domain?"
date: "2024-12-14"
id: "how-to-set-a-cross-domain-when-using-an-activecampaign-form-pop-up-post-and-pass-utm-parameters-from-my-domain-to-a-new-domain"
---

alright, let's unpack this activecampaign cross-domain form submission with utm parameters issue. it's a classic problem, i've bumped into this kind of headache more times than i care to remember. it's not pretty, but we can definitely get it sorted. basically, you have a form on your site (domain a) that pops up (probably using their javascript snippet) and it submits to activecampaign's servers (domain b). and, because marketing is a thing, you need to carry over those precious utm parameters.

the core problem is the browser's same-origin policy. it's a security feature designed to prevent malicious scripts from messing with data on other websites. it's a good thing in general, but here, it makes our lives a little bit harder. because activecampaign’s form submits via a `post` method, we are going to deal with data submission that is not as trivial as a simple get method with parameters appended to the url.

i’ve had my fair share of encounters with this cross-domain stuff. back in 2015, i was working on a project where we were consolidating leads from multiple microsites into a central crm. each microsite had a slightly different set of utm parameters and form fields, and passing all of them correctly through the ajax submit function to the crm endpoint using a reverse proxy server became a real mess. i ended up spending a whole week basically debugging the reverse proxy configuration, that was not pretty at all. i felt like a detective in a bad 80s tv show at that time. anyway, live and learn i guess. so i can tell you that dealing with cross domain data can be a pain if you don't approach the right way and understand the fundamentals.

first, understand that we have two main areas to tackle: capturing the utm parameters on domain a and then passing them along to activecampaign on submission.

**capturing utm parameters:**

we need to grab the utm parameters from the url when the user lands on your site (domain a). we can do this using javascript. here is an example of how to extract those url params, and i mean the utm ones, specifically:

```javascript
function getutmparameters() {
    const params = new urlsearchparams(window.location.search);
    const utmparams = {};
    for (const [key, value] of params.entries()) {
        if (key.startswith('utm_')) {
            utmparams[key] = value;
        }
    }
    return utmparams;
}
```

this function uses the `urlsearchparams` object to parse the url’s query string, then it iterates through them, looking for any key that begins with `utm_`, and then those values will be added to the `utmparams` object.

we’ll then need to store them somewhere so they’re available when the activecampaign form is submitted. session storage or local storage can be used for this but for simplicity we can add them as hidden inputs within the ac form.

**injecting utm parameters into the form:**

now comes the tricky part. activecampaign gives you a javascript snippet to embed your form, usually as a popup. we need to modify that to include our utm parameters as hidden input fields before the submission happens.

first we need to get the generated active campaign form id. the script injects the generated form inside a div element with that id, and we can then grab it with a selector and append our hidden elements with the utm params:

```javascript
function injectutmparams(formid) {
    const utmparams = getutmparameters();
    const form = document.getelementbyid(formid);

    if (form) {
        for (const key in utmparams) {
            if (utmparams.hasownproperty(key)) {
                const hiddeninput = document.createelement('input');
                hiddeninput.type = 'hidden';
                hiddeninput.name = key;
                hiddeninput.value = utmparams[key];
                form.appendchild(hiddeninput);
            }
        }
    }
}

// you will call this function with your formid (e.g. ac-embedded-form-123) after the ac form script is loaded
injectutmparams('ac-embedded-form-123');
```

this javascript function will call the `getutmparameters` function and then append the hidden parameters to your active campaign's form.

**activecampaign configuration:**

now, here is where things can get hairy with their internal form processing, and there is no clean way to solve the data mismatch if the field names are not exactly equal, but if you set the input names in the same way as they are named in the form (as the `name` attributes in the hidden inputs) you should see them in activecampaign. i remember, one time i spent hours trying to figure out that only those fields with the corresponding name in activecampaign’s config would be saved. it was a classic brain fart.

so go into your activecampaign account, open the form you are using and add the custom fields like `utm_source`, `utm_medium`, `utm_campaign`, etc. it's vital that these field names exactly match the keys we're using in our javascript – otherwise, the data won't get passed correctly. double check the field names to ensure it works.

**a final caveat and suggestions:**

testing is key. inspect the network tab in your browser dev tools when you submit the form. you should see the utm parameters in the post request payload. also, do a dry run of an activecampaign form submission from your domain a, using several url parameters to make sure they are working as expected before going live.

this approach keeps things relatively clean, avoids using `get` request parameters that would be easily modified and keeps the whole logic using only javascript, with no server-side language or framework required.

**resources**

if you are interested in delving deeper into this, i recommend you to read these books:

*   "professional javascript for web developers" by nicholas c. zakas: this book contains a comprehensive overview of javascript features and techniques, including form handling and dom manipulation, essential for understanding the underlying mechanisms involved in the process we just covered.
*   "http: the definitive guide" by david gourley and brian totty: this book is a deep dive into the http protocol, that will help you to understand the request/response cycle and the crucial details about the interaction between the client side and the server.
*   “understanding the dom” by joshua d. davis: this book can provide the required knowledge about working with the dom and javascript in a browser, a perfect reading for a deeper undestanding of the js snippets i just wrote.

remember: always sanitize your input. this example just covers the basic implementation, always sanitize the url parameters if you are going to save it in your database.

i hope this helps you out. cross-domain form submissions can be a pain, but with a solid understanding of how things work, it is doable, just remember to test, test and test it before going into production. good luck.
