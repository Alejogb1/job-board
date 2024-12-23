---
title: "How can I create a clickable phone number link in a Wicked PDF document?"
date: "2024-12-23"
id: "how-can-i-create-a-clickable-phone-number-link-in-a-wicked-pdf-document"
---

Let's tackle this. I remember a project back in '17, we were generating some rather complex legal documents via Wicked PDF and ran into the very same problem: making phone numbers clickable. Standard html-based links weren’t directly translating through Wicked PDF’s rendering process. It took some fiddling, but we found a reliable workaround, which I'll share here.

The core issue stems from how Wicked PDF (which, under the hood, is typically using a webkit-based rendering engine) interprets and translates html into a PDF. Simple `<a>` tags with `href="tel:..."` often don't render the expected clickable behavior, or worse, result in the number being displayed as text and not a link at all. This happens because the PDF rendering process doesn’t always faithfully translate the interactive aspects of HTML, like the `tel:` URI scheme, directly. We have to be a little more explicit.

The key is understanding that while the browser might know how to treat `tel:`, a pdf document ultimately needs a specific, PDF-defined way to signify a link action. The solution involves leveraging the underlying html and css capabilities that wicked pdf can understand, and then generating a pdf that utilizes pdf annotation objects correctly.

First, we ensure the phone number is formatted as a standard anchor tag with a `tel:` href attribute. This is the browser-view functionality, and it does help if the generated HTML is also rendered in a web browser before the pdf generation. This confirms the link will work for screen preview, which is also a great debugging step. This is what the base would look like in your HTML source:

```html
<p>
  For assistance, please call us at: <a href="tel:+15551234567">+1 (555) 123-4567</a>
</p>
```

While this may *display* a link in the browser (or when using the rendered html), it’s unlikely to result in a functional link in the PDF by itself. Now, for the actual pdf-specific part, this is where we add some css styling. In this case, we will need to explicitly control the color of the text, to make it clear that it’s a clickable item. We also need to avoid styling that might interfere with the PDF rendering engine. For this, we’ll embed some basic styling into the page's `<head>` section. Typically, Wicked PDF picks this up. This next snippet would need to exist in your header or be added through a style tag embedded in the document:

```html
<style>
a[href^="tel:"] {
  color: blue; /* Make the link blue, a commonly understood visual cue */
  text-decoration: underline; /* Underline for more clarity */
}
</style>
```

The above css will make sure that the phone numbers appear as links in the preview mode and in the rendered pdf. However, the `tel:` URI will still need to be properly interpreted by the pdf reader to trigger the phone call initiation. The browser does this implicitly, but the PDF readers do not. And this behavior is where Wicked PDF’s magic (or sometimes, lack thereof) comes in. While Wicked PDF's output is great, it's not going to directly create a link that is recognized by all pdf readers as a click-to-call. It just renders the html into a pdf representation.

Thus, to be absolutely certain, we need to introduce a JavaScript workaround that will use the `window.location.href` javascript attribute on a click to mimic the functionality of the `href` tag. The trick here is to attach an inline javascript function to the `onclick` attribute, like so:

```html
<p>
   For assistance, please call us at:
  <a
   href="tel:+15551234567"
   style="color: blue; text-decoration: underline;"
   onclick="window.location.href='tel:+15551234567'; return false;">+1 (555) 123-4567
   </a>
</p>
```

This is slightly less than ideal as it requires an inline JavaScript handler, but it's necessary in some scenarios where the PDF render engine doesn't handle tel: urls correctly. In many cases, this final snippet will generate a clickable phone number on the PDF. The inline style will make sure the text appears as a link, and the inline javascript will redirect the page (which is a PDF) to trigger the call dialog, which works on most pdf renderers, including iOS and Android devices.

It's crucial to recognize that this approach is a pragmatic solution to a limitation. Wicked PDF isn’t designed to be a full-blown interactive document creator, so leveraging CSS and JavaScript, rather than attempting to inject PDF-specific instructions directly, is the path of least resistance. The approach is effective because we're relying on browser-like behavior that the PDF rendering process generally respects.

Remember, thorough testing across different PDF readers is essential because the behavior might vary slightly. It’s advisable to test using Adobe Acrobat, some common mobile PDF readers (like those on iOS and Android), and maybe a few browser-based viewers to cover the vast majority of users. If you find discrepancies, you might need to adjust the CSS or JavaScript slightly, but the core technique should still hold.

Regarding further resources, I'd highly recommend checking out the PDF specification itself, ISO 32000. It's the bible of all things PDF. While a full read is hefty, you can skim specific sections on "annotation objects" to better understand how pdfs represent interactive elements. Another great resource would be “Pdf Explained” by John Whitington. It’s incredibly practical for understanding the inner workings of PDF files and it’s easier to digest than the full spec. And, of course, studying the specifics of your webkit implementation could also be beneficial, you can search for that on the webkit's official website, or through articles related to it, but generally, the approach mentioned above should work fine with most common implementations. Understanding those underlying concepts really helped us when we were facing similar issues, and it continues to do so. While it took a bit of trial and error initially, these strategies have served us well ever since. Good luck with your project, and happy coding!
