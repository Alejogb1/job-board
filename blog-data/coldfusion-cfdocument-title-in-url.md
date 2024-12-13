---
title: "coldfusion cfdocument title in url?"
date: "2024-12-13"
id: "coldfusion-cfdocument-title-in-url"
---

Okay so you're asking about getting the cfdocument title in the URL huh Been there done that got the t-shirt Let me tell you this is a classic problem that pops up more often than you'd think especially with legacy ColdFusion apps that haven't really been touched in ages. It's not super straightforward and CF's kind of quirky behavior with cfdocument doesn't always help.

I've been dealing with ColdFusion since CF 5 you remember the days of that old server config UI. I remember clearly when I had this problem in an old e-commerce system I was working on back in 2010. A project for an online book store they wanted to generate dynamic PDFs of order confirmations each one with the order ID in the filename and the PDF title. It was a real struggle then because we had to figure out how to get that title dynamically without resorting to some crazy file system hack.

So the core issue here is that cfdocument’s title attribute doesn't directly impact the URL or filename when the pdf is created. The title is just part of the PDF metadata. It shows up when you open the PDF in a reader program but doesn't change where the file is actually located in the web context. You can't just magically get the title into the URL without doing a bit of extra work.

Now here’s the way I see this issue it's about understanding the server side process a bit better. When you generate a PDF using cfdocument ColdFusion is creating a temporary PDF file on the server and then either streaming it to the client or saving it to the file system and providing a link. The server-side code is what dictates the ultimate URL and filename.

To get the title into the URL you'll need to do a few things you can't just rely on cfdocument’s internal handling of metadata. You need to extract the title you set and then use that to construct the URL which includes the filename part.

Here’s a basic example showing you the concept without file saves:

```cfml
<cfset myTitle = "Order Confirmation - #orderID#">
<cfdocument format="pdf" filename="order_#orderID#.pdf" title="#myTitle#">

    <h1>Order Confirmation</h1>
    <p>Order ID: #orderID#</p>
    <p>Your order has been placed successfully!</p>

</cfdocument>
```

This example is simple. It generates a PDF streams it to the client and its using the `filename` attribute of `cfdocument` to set the file name in the browser download dialog this is very important and it's probably where you are messing it up. This is not the title I know you are trying to set but the code below shows how to do it. This example sets the `filename` in the dialog when the client downloads the file as the client does not see the file title as you are thinking.

Now what about the dynamic part? You need to use the title you set in the cfdocument tag. Here is how you do it by using the `cfcontent` tag.

```cfml
<cfset myTitle = "Order Confirmation - 12345">
<cfdocument format="pdf" title="#myTitle#" name="myPDF">

    <h1>Order Confirmation</h1>
    <p>Order ID: 12345</p>
    <p>Your order has been placed successfully!</p>

</cfdocument>

<cfset pdfBytes = myPDF>
<cfset fileName = "order_" & Replace(myTitle, " ", "_", "all") & ".pdf">
<cfheader name="Content-Disposition" value="attachment; filename=""#fileName#""">
<cfcontent type="application/pdf" variable="#pdfBytes#" />
```

In this example I added an variable called `myPDF` which stores the binary PDF generated from `cfdocument`. This is needed to dynamically build the header for the name using `cfcontent` and `cfheader`. The code first creates the PDF using `cfdocument` with your title and then reads its output to the `myPDF` variable. Then the `myTitle` is used again to create the filename after removing spaces replacing them with "_". The `cfheader` and `cfcontent` output the created file to the client but before it outputs the client side is given the desired filename with `Content-Disposition` header.

This approach uses `cfcontent` and custom headers which gives you total control over how the file is served to the browser. This method is what you are looking for.

Now that I covered the basic approach lets go to something a little more advanced using cfdocument’s save function.

```cfml
<cfset myTitle = "Invoice - #invoiceID#">
<cfset fileName = "invoice_" & Replace(myTitle, " ", "_", "all") & ".pdf">
<cfdocument format="pdf" title="#myTitle#" filename="#fileName#" savepath="#ExpandPath('./temp')#" overwrite="true" name="myPDF">

    <h1>Invoice</h1>
    <p>Invoice ID: #invoiceID#</p>
    <p>Thank you for your business!</p>

</cfdocument>

<cfset pdfPath = ExpandPath('./temp/#fileName#')>
<cfoutput>
    <a href="temp/#fileName#">Download Invoice</a>
</cfoutput>
```

This is using the `savepath` and generating a url that serves the downloaded file this is a very common use case and it's why I always prefer to use `cfdocument` with `savepath` instead of directly streaming. Also it has the same functionality where the spaces are being converted to "_".

Regarding `cfdocument` and this whole subject I recommend you read Adobe's official documentation on ColdFusion. This is your primary source for everything. It’s not the most exciting read but it’s thorough and covers the nitty-gritty details. Another fantastic resource is the CFML in Action book that goes way beyond the basics. It gives real-world examples and is a great resource to understand what's really going on behind the scenes. Also always check out the ColdFusion documentation on the Adobe website for any new updates.

One tip from a CF old-timer always name your variables consistently like myTitle or myPDF and not something like a or tempDoc it will make your code so much easier to debug when you are having a late night issue.

One last note. Don't overthink this one the title in cfdocument is meant for PDF metadata and not for URLs. The URL and filename are constructed in the server side and not by the PDF internal properties.

Also when working with PDF make sure the client supports the PDF format it happens quite often that the PDF is broken or can't be read by the client PDF reader. In a recent project we spent hours debugging that and all it was a client configuration issue that was not even related to the code. You should always try to download it in many different browsers or devices to see if the PDF is rendering correctly.

And one last thing I’ve been telling this to new ColdFusion developers all the time. I was having a bad day a few years ago after working 20 straight hours without sleeping when a junior dev asked me a question about why his code was not working at 3 AM. I simply told him in frustration "you gotta love bugs otherwise you should not be in this field" then he just replied "well bugs are not that bad once you get used to them" and he was completely correct and that changed my perspective.

This is it hope it helps let me know if you got other questions.
