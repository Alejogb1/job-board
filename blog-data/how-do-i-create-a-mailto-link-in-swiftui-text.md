---
title: "How do I create a mailto link in SwiftUI Text?"
date: "2024-12-23"
id: "how-do-i-create-a-mailto-link-in-swiftui-text"
---

Alright, let's delve into this. I've tackled `mailto` links in SwiftUI Text more than a few times, particularly during a project where we were building a contact page in a mobile app. It wasn't as straightforward as web development, which might be your initial thought, but it's entirely manageable. The key is understanding how SwiftUI handles attributed strings and how we can leverage that to incorporate tap actions with specific protocols.

The challenge arises because `Text` view in SwiftUI, in its basic form, doesn’t directly support tappable links like HTML's `<a>` tag. We can’t simply embed a `mailto:` protocol string. Instead, we need to make use of `NSAttributedString` or its SwiftUI-compatible wrapper, `AttributedString`. We need to create an attributed string where a specific part of the string—the email address, in our case—has a link attribute associated with it. This attribute is what SwiftUI then recognizes and acts upon when tapped.

Let’s start with the fundamental approach, the foundation we'll build upon for more complex scenarios. This involves programmatically constructing an `AttributedString`:

```swift
import SwiftUI

struct MailtoLinkBasic: View {
    let emailAddress = "support@example.com"

    var body: some View {
        Text(createAttributedMailtoLink())
    }

    func createAttributedMailtoLink() -> AttributedString {
       var attributedString = AttributedString("Contact us at \(emailAddress)")
        if let range = attributedString.range(of: emailAddress) {
            attributedString[range].link = URL(string: "mailto:\(emailAddress)")
        }
        return attributedString
    }
}


struct MailtoLinkBasic_Previews: PreviewProvider {
    static var previews: some View {
        MailtoLinkBasic()
    }
}
```

In this first example, `createAttributedMailtoLink()` dynamically constructs an `AttributedString` using a base string with the email address. We locate the range of the email address within that string and then apply the `link` attribute to that specific part, using the `mailto:` url scheme. This is the minimum necessary to make the email address tappable and trigger the email app on a device. Notice, this simple implementation handles *only* the email address. What if you want to link a more verbose phrase?

Now, let's say we want to incorporate a custom display phrase for our link instead of the email itself. That is, a link saying "Email Support" or "Contact Us" that actually opens the email client. Here is how that looks:

```swift
import SwiftUI

struct MailtoLinkCustomText: View {
    let emailAddress = "support@example.com"
    let linkText = "Email Support"

    var body: some View {
        Text(createAttributedLinkText())
    }


    func createAttributedLinkText() -> AttributedString {
        var attributedString = AttributedString("To get help, \(linkText)")
        if let range = attributedString.range(of: linkText) {
            attributedString[range].link = URL(string: "mailto:\(emailAddress)")
        }
         return attributedString
    }
}


struct MailtoLinkCustomText_Previews: PreviewProvider {
    static var previews: some View {
        MailtoLinkCustomText()
    }
}
```

This adaptation modifies our approach to display *linkText*, an arbitrary string, which when tapped opens the mail client using the *emailAddress* value. This is a more user-friendly approach, as it avoids exposing email addresses directly in your text, which is often preferable. We still leverage the `range(of:)` method to target our phrase for attribute application.

The method I've outlined works well for singular email links. However, what if we need a dynamic approach? What if the text could contain several phrases or links? The approach would need to programmatically construct several ranges of linked text. Here is an example showing multiple, independent links in the same view:

```swift
import SwiftUI

struct MailtoMultipleLinks: View {
    let emailAddress1 = "support@example.com"
    let emailAddress2 = "inquiries@example.com"
    let linkText1 = "Customer Support"
    let linkText2 = "General Inquiries"
    var body: some View {
        Text(createMultipleAttributedLinks())
    }
    func createMultipleAttributedLinks() -> AttributedString {
        var attributedString = AttributedString("Contact \(linkText1) or \(linkText2).")
        
        if let range1 = attributedString.range(of: linkText1) {
           attributedString[range1].link = URL(string: "mailto:\(emailAddress1)")
        }
        if let range2 = attributedString.range(of: linkText2) {
            attributedString[range2].link = URL(string: "mailto:\(emailAddress2)")
        }
        return attributedString
    }
}


struct MailtoMultipleLinks_Previews: PreviewProvider {
    static var previews: some View {
        MailtoMultipleLinks()
    }
}
```
In this final snippet, you can see that our attributed string now includes multiple email addresses with corresponding custom link texts in a single Text element. This is accomplished by iterating through each linkable portion of the full string, and assigning a custom `mailto:` URL scheme for each segment. This is a practical solution for real-world use cases where you might need several distinct email contacts accessible through the same interface, or, indeed, a combination of hyperlinks to different resources, not just email.

For further reading on `AttributedString` and its capabilities, I'd recommend exploring Apple's official documentation on `Foundation.AttributedString`. Additionally, the book "SwiftUI by Tutorials" (multiple editions) offers practical insights into how `AttributedString` is used in SwiftUI, specifically chapter devoted to advanced text handling. "Programming iOS 17" by Matt Neuburg also provides a deep dive into the underlying text management in Apple's platforms.

In conclusion, using `AttributedString` provides a robust way to create `mailto` links within SwiftUI's `Text` view. It's crucial to programmatically construct the attributed string with correct `mailto` URL schemes and to correctly target the text you want to be tappable. While there's no direct equivalent of the html's <a> tag in SwiftUI, this approach offers flexibility in how you present and integrate tappable email links in your applications. It’s worked well in numerous projects for me, and with a bit of practice, you’ll find it’s a straightforward approach.
