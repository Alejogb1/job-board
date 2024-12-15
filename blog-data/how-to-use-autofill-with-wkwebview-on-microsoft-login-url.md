---
title: "How to Use AutoFill with WKWebView on Microsoft login URL?"
date: "2024-12-15"
id: "how-to-use-autofill-with-wkwebview-on-microsoft-login-url"
---

so, you’re running into the autofill problem with wkwebview on a microsoft login page, eh? i've been there, spent more than a few late nights staring at my screen trying to figure out why the darn thing won't populate those input fields. it’s a frustrating situation, and i get why you're asking. let's break down what's probably going on and how to fix it, based on my own painful experiences.

first off, the problem isn’t inherently with wkwebview itself, it’s more about how apple’s autofill system interacts with complex login forms, especially the ones you find on microsoft’s auth services. these forms are usually not straightforward html, often javascript heavy, use dynamic ids, and have multiple steps. autofill looks for common patterns and identifiers to know where to inject the saved credentials. if those identifiers are not what autofill expects or are missing completely, it will throw up its hands and do nothing.

my initial stumble with this problem happened way back when i was building a mobile app for internal use at a small startup. we were integrating with several microsoft apis, needing users to authenticate through their microsoft accounts. like you, i used a wkwebview, and it all worked perfectly *except* the auto-fill. passwords were never offered, usernames never pre-filled. it was a nightmare for our testers, not to mention the users. i spent several days, at least, looking over the javascript code the login page was using, i even ended up inspecting network requests from the login flow trying to see where the input fields were and how they worked.

it turned out that microsoft, in some of its login implementations, uses javascript to dynamically generate input elements. apple's autofill system does not always play well with this. it's built to look for static html with consistent attributes, and when the structure is changing on the fly, it gets confused. it's like trying to identify a moving target.

here's a simple example of what the html structure should look like for autofill to work smoothly with a standard username field. this is not what microsoft does:

```html
<input type="text" name="username" id="username" autocomplete="username" />
```

this is a simple input field with an "autocomplete" attribute set to "username". when you implement this in your code, autofill will look for such structure. simple enough. but, microsoft login form can generate dynamically many html structures. i have seen some where id's change on every page load.

the key here is the "autocomplete" attribute. it tells the os what kind of information to offer to autofill. there are a bunch of values like "password", "email", "name", etc.

now, onto the wkwebview aspect, there are things you can do to nudge autofill in the right direction. the most important bit is setting the correct configuration for your wkwebview. specifically you need to enable the "userinteractionenabled" attribute:

```swift
import webkit

class loginviewcontroller: uiviewcontroller {

  var webview: wkwebview!

  override func viewdidload() {
    super.viewdidload()

    let webviewconfig = wkwebviewconfiguration()
    let webview = wkwebview(frame: .zero, configuration: webviewconfig)
    webview.translatesautoresizingmaskintoconstraints = false
    webview.uiinteractionenabled = true // this is important!
    view.addsubview(webview)

    nslayoutconstraint.activate([
      webview.topanchor.constraint(equalto: view.topanchor),
      webview.bottomanchor.constraint(equalto: view.bottomanchor),
      webview.leadinganchor.constraint(equalto: view.leadinganchor),
      webview.trailinganchor.constraint(equalto: view.trailinganchor),
    ])

    self.webview = webview
    let microsoftloginurl = url(string: "https://login.microsoftonline.com/")!
    webview.load(urlrequest(url: microsoftloginurl))
  }
}

```

setting `uiinteractionenabled` will make sure the view responds to user input, triggering autofill suggestions. if this was not enabled autofill would not work at all.

but, even with this correct configuration, the dynamically generated elements on the microsoft page might still throw things off. i have found that sometimes what you need to do is inject some javascript to help autofill recognize the elements.

for example, you could inject a script that adds the 'autocomplete' attribute to input fields after the page has loaded and the elements are present:

```swift
func webview(_ webview: wkwebview, didfinishnavigation navigation: wknavigation!) {
    // injecting a javascript that sets the autocomplete on the first field it can find
    // this should be adjusted depending on the website login flow.
   webview.evaluatejavascript("""
        var inputs = document.getelementsbytagname('input');
        if (inputs.length > 0) {
             for (var i = 0; i < inputs.length; i++) {
                var input = inputs[i];
                if(input.type == 'email' || input.type == 'text') {
                   input.setattribute('autocomplete','username');
                  break;
                }
               if(input.type == 'password') {
                 input.setattribute('autocomplete','current-password');
               }

            }
        }
   """)
}
```

note that i'm using javascript here to select input elements and add the `autocomplete` attribute. this should be done on `didfinishnavigation` so that the elements are already loaded in the page.

this approach should be used as a workaround. you should inspect the page to correctly target the input elements you need to autofill. sometimes you might need to modify the script to handle different types of fields or to use more specific selectors to locate those fields.

there are no magic bullet solutions. it's a game of cat and mouse where you adapt to the specific page's structure. sometimes, it involves reading through apple's documentation on how autofill works and then examining the page source to understand the fields you want to target. it really is all about understanding what the autofill system expects, and ensuring the login page structure provides those.

a good resource to understand better what is going on with autofill in ios, is apple's own documentation on how autofill works. it provides guidelines to how the elements should look like in order to be properly autofilled by the system. in the past there were also interesting blogs about how people use webview for authentication but they are harder to find nowadays.

sometimes you may encounter other issues. for example, if the microsoft page uses multiple redirects, autofill can get confused when the dom has changed a lot and the form has been reloaded with new ids. a workaround for that can be injecting the javascript on the `didcommitnavigation` instead of `didfinishnavigation`. i had a weird situation where my webview was loading the page twice and autofill was not working at all. the problem was that i was loading the webpage from a wrong url with `https` and the original page was actually `http`. once i fixed that the autofill started working as expected. i learned a lot on those days, those were challenging. as someone very smart one day told me, debugging is like being a detective in a crime scene, and i can't agree more. it requires lots of focus and patience.

another issue i've encountered is that sometimes autocomplete will only offer suggestions once you've manually typed something into the input field. it’s not a bug, it’s a weird security precaution by the os. it's like, "oh, you can actually type here? ok, i trust you now, here are your saved passwords".

it's a pain, i am not going to lie, but if you persevere, use the `autocomplete` attributes correctly, enable `userinteractionenabled` in your webview, and use the javascript to nudge the autofill system in the right direction. also, it helps a lot to inspect the page carefully and see how the input fields are really named.

good luck, i'm confident you'll nail it. just remember to examine the specific login page html structure, and use the tools available in wkwebview to help you. and don't hesitate to use javascript to dynamically set those `autocomplete` attributes when needed, just make sure you don't over-write them, and that you know what the script is actually doing.
