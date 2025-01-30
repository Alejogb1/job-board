---
title: "How does a print view controller work?"
date: "2025-01-30"
id: "how-does-a-print-view-controller-work"
---
The core function of a `UIPrintInteractionController`, as I've observed across numerous iOS projects, isn't to directly render content for printing. Instead, it acts as a high-level coordinator, orchestrating the interaction between your application's data and the iOS printing subsystem. It’s the intermediary facilitating communication with available printers and presenting a standardized user interface for print job configuration. My experience stems from developing both simple document printing functionalities for a small business app and complex, custom-layout reports for a large enterprise system, giving me exposure to a wide range of scenarios.

The `UIPrintInteractionController` relies on several key objects to accomplish its task. The most important is the `UIPrintInfo` object, which specifies general characteristics about the print job, including the print job's name, the type of media (e.g., photo, general), the print range, and duplexing options. This object allows you to preconfigure the print job based on your application's needs, reducing the number of options that the user needs to specify. Following that, a `UIPrintFormatter` acts as the bridge between your content and the print system. It is responsible for laying out the content onto pages based on the selected paper size and margins. `UIPrintFormatter` itself is an abstract class, you will typically interact with it using concrete subclasses such as `UIViewPrintFormatter`, `UISimpleTextPrintFormatter` or, for custom printing, through a custom subclass. The `UIPrintInteractionController` then manages the presentation of a system-provided print dialog that shows available printers, paper size options, and lets the user configure the print job. Lastly, once the user initiates the print process, the controller hands off the formatted data to the system for background processing and printing.

Internally, the controller utilizes the print services framework, a collection of APIs that handle communication with printer drivers and management of print queues. When your application initiates a print job, it is added to a system-wide print queue. The iOS system then handles all the details of sending the output to the printer. It's important to note that this process is entirely asynchronous, therefore your application should not block waiting for the print job to complete. The `UIPrintInteractionController`’s delegate methods allow you to monitor the print process and react when specific events occur, for instance when the print dialogue is presented, the printing process starts or when the job completes.

Here's a breakdown with code examples:

**Example 1: Basic Printing of a UIView**

This scenario demonstrates the simplest form of printing: taking a view and sending it to the printer. This scenario is common when wanting to print simple charts or user interface elements.

```swift
import UIKit

func printUIView(viewToPrint: UIView) {
    let printController = UIPrintInteractionController.shared
    let printInfo = UIPrintInfo(dictionary: nil)
    printInfo.outputType = .general
    printInfo.jobName = "UIView Print"
    printController.printInfo = printInfo

    let formatter = UIViewPrintFormatter(view: viewToPrint)
    printController.printFormatter = formatter

    printController.present(animated: true, completionHandler: nil)
}

// Usage example (assuming you have a view named 'myView'):
// printUIView(viewToPrint: myView)
```

In this example, we obtain the shared instance of `UIPrintInteractionController`. We configure a basic `UIPrintInfo` setting its output type and job name, which will be visible to the user. We then instantiate a `UIViewPrintFormatter`, associating it with the view to print, then assigning it to the `printFormatter` property of the controller.  Finally, we present the print interface. The system handles all printing processes after the print job is started from the print UI. The print dialog automatically appears when you call the `present(animated:completionHandler:)` method.

**Example 2: Printing Text Content Using UISimpleTextPrintFormatter**

This code demonstrates printing text, where we can set font, color and other simple text formatting options. It handles text wrapping, page breaks and renders a document with multiple pages if necessary. I’ve often used this for printing receipts or simple textual reports.

```swift
import UIKit

func printText(textToPrint: String) {
    let printController = UIPrintInteractionController.shared
    let printInfo = UIPrintInfo(dictionary: nil)
    printInfo.outputType = .general
    printInfo.jobName = "Text Print"
    printController.printInfo = printInfo

    let textFormatter = UISimpleTextPrintFormatter(text: textToPrint)
    textFormatter.font = UIFont.systemFont(ofSize: 14)
    textFormatter.color = .black
    printController.printFormatter = textFormatter

    printController.present(animated: true, completionHandler: nil)
}

// Usage example:
// printText(textToPrint: "This is the text to be printed. It might be long and wrap to multiple lines. This can demonstrate multi-page printing.")
```

Here, the setup is similar to the previous example, but instead of a view, we instantiate a `UISimpleTextPrintFormatter`, passing the text we want to print. We also set properties such as `font` and `color` to further style the output.  The text is automatically laid out across multiple pages by the system. The `UISimpleTextPrintFormatter` handles word wrapping and pagination automatically, abstracting away the complexities of multiple page layouts.

**Example 3: Custom Print Formatter using UIPrintPageRenderer**

For more complex printing scenarios where custom page layout is needed, `UIPrintPageRenderer` becomes useful. This example shows a highly customized printing output with custom drawing and page layouts. In my experience, I’ve used this when needing to render complex forms with dynamic data.

```swift
import UIKit

class CustomPrintRenderer: UIPrintPageRenderer {
    let content: String
    let title: String

    init(content: String, title: String) {
        self.content = content
        self.title = title
        super.init()
    }


    override func drawPage(at pageIndex: Int, in printableRect: CGRect, for printFormatter: UIPrintFormatter) {
        super.drawPage(at: pageIndex, in: printableRect, for: printFormatter)

        let titleFont = UIFont.boldSystemFont(ofSize: 20)
        let contentFont = UIFont.systemFont(ofSize: 12)
        let titleAttributes: [NSAttributedString.Key: Any] = [.font: titleFont, .foregroundColor: UIColor.black]
        let contentAttributes: [NSAttributedString.Key: Any] = [.font: contentFont, .foregroundColor: UIColor.darkGray]
        let attributedTitle = NSAttributedString(string: title, attributes: titleAttributes)
        let attributedContent = NSAttributedString(string: content, attributes: contentAttributes)

         let titleSize = attributedTitle.size()
         let titleRect = CGRect(x: printableRect.origin.x, y: printableRect.origin.y + 20, width: printableRect.width, height: titleSize.height)
        attributedTitle.draw(in: titleRect)

         let contentRect = CGRect(x: printableRect.origin.x, y: titleRect.maxY + 10, width: printableRect.width, height: printableRect.height - titleRect.maxY - 10)
         attributedContent.draw(in: contentRect)
    }

    override func numberOfPages() -> Int {
      return 1
    }
}

func printCustomContent(content: String, title: String) {
  let printController = UIPrintInteractionController.shared
  let printInfo = UIPrintInfo(dictionary: nil)
  printInfo.outputType = .general
  printInfo.jobName = "Custom Print"
  printController.printInfo = printInfo

  let renderer = CustomPrintRenderer(content: content, title: title)
    let pageInfo = UIPrintPageRenderer()
  renderer.printFormatter.perPageContentInsets = UIEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)

  printController.printPageRenderer = renderer

  printController.present(animated: true, completionHandler: nil)
}

// Usage example:
// printCustomContent(content: "This is my custom printed content. I have fine-grained control over page layout.", title: "Custom Title")

```

In this more advanced example, we create a custom `UIPrintPageRenderer` subclass. Within this subclass, we override `drawPage(at:in:for:)` to control precisely how content is drawn onto the page, taking into account the page index and printable area. The number of pages is determined in `numberOfPages()`.  The controller is then set up as before, but instead of setting `printFormatter`, we set `printPageRenderer` to our custom renderer. This affords the most flexibility for complex layout requirements, although it requires a deeper understanding of Core Graphics drawing.  The `perPageContentInsets` in this example ensure that text is drawn inside page margins.

For further learning, Apple's documentation for `UIPrintInteractionController`, `UIPrintInfo`, and `UIPrintFormatter` is a core resource. A thorough understanding of Core Graphics drawing is essential when creating custom `UIPrintPageRenderer` subclasses, and the relevant Core Graphics documentation should be referred to for more advanced custom layouts. Understanding the Print Services Framework can be a helpful resource to grasp how printing is handled system-wide. These resources are the primary learning materials I have leaned on during my career in this area.
