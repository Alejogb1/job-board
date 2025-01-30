---
title: "How can I use AirPrint with multiple MVCs in Swift?"
date: "2025-01-30"
id: "how-can-i-use-airprint-with-multiple-mvcs"
---
AirPrint functionality, while seemingly straightforward from a user perspective, requires careful architectural consideration when integrating it across multiple Model-View-Controller (MVC) instances in a Swift application. The core challenge lies in managing the shared print interaction and preventing redundant `UIPrintInteractionController` instantiations, which can lead to unexpected behavior or resource leaks. I've encountered this firsthand when building a multi-view report generation feature for a medical records application, where each view (patient summary, lab results, imaging reports) needed to support printing.

The primary issue with naive implementation is creating a new `UIPrintInteractionController` within each MVC that requires printing. This approach results in each controller managing its own print flow, leading to potential conflicts if multiple printing actions are initiated simultaneously or if a single, consistent print interface is desired. Furthermore, the responsibility for configuring the print information (e.g., print formatters, page settings) becomes duplicated across the controllers.

To address this, a centralized approach is paramount. We need to delegate the responsibility for creating and managing `UIPrintInteractionController` to a dedicated component, typically a singleton or a class accessible from across the application. This component can be responsible for initiating the print interaction, configuring the print job, and handling the completion and error states. This approach promotes code reusability, ensures consistency in the print experience, and simplifies the management of the underlying print machinery.

One effective method is to utilize a dedicated manager class that encapsulates the `UIPrintInteractionController`. Each MVC requiring printing would interact with this manager, sending data relevant to its specific content. This manager would then handle the printing process, thereby avoiding direct control of the printing system in individual view controllers.

Here’s how such a system might be implemented, building off experience of migrating print functionalities across the different sections of the medical application:

**Code Example 1: PrintManager Class**

```swift
import UIKit

class PrintManager {
    static let shared = PrintManager() // Singleton for global access
    private var printController: UIPrintInteractionController?
    private init() {} // Prevent direct instantiation

    func printContent(printFormatter: UIPrintFormatter) {
        if let controller = UIPrintInteractionController.shared {
            controller.printFormatter = printFormatter
            controller.present(animated: true, completionHandler: nil)
        } else {
          // this is here to handle the unexpected
          // I have seen instances where for an unkonwn reasons UIPrintInteractionController.shared is nil
          // in those cases we create it
          printController = UIPrintInteractionController.shared
            printController?.printFormatter = printFormatter
             printController?.present(animated: true, completionHandler: nil)
        }
    }
    
    func reset() {
        printController = nil
    }
}
```

*   **Singleton Pattern:** This ensures a single instance of the `PrintManager` exists throughout the application, facilitating centralized print management. The `shared` property provides easy access.
*   **Encapsulation:** The `printController` is kept private, preventing external access and ensuring that the print process is managed only through the `PrintManager` class. This design decision also helps contain the memory used.
*   **`printContent` Method:** This is the central point of interaction for MVCs requiring print functionality. It accepts a `UIPrintFormatter` as an argument, making it agnostic to the specific content being printed. The `UIPrintInteractionController.shared` gives a better way to manage the controller instance and its life cycle.
*   **`reset()` method:** This method is useful in cases where we want to release the instance of the `printController`, it is not always necessary but in long running apps, memory allocation can be problematic, releasing the references improves the apps stability

**Code Example 2: MVC Usage**

```swift
import UIKit

class PatientSummaryViewController: UIViewController {
    
    //...other view controller logic...

    func printSummary() {
        let summaryString = "Patient Name: John Doe\nAge: 45\nMedical History: ..." // Actual data
        let printFormatter = UISimpleTextPrintFormatter(text: summaryString)
        printFormatter.pageWidth = 500
        printFormatter.pageHeight = 700

        PrintManager.shared.printContent(printFormatter: printFormatter)
    }
    
    deinit {
            PrintManager.shared.reset()
        }
}

class LabResultsViewController: UIViewController {
   
    //...other view controller logic...

    func printResults() {
        let resultsString = "Lab Test: Blood Count\nResult: Normal\nDate: ..." // Actual data
        let printFormatter = UISimpleTextPrintFormatter(text: resultsString)
       printFormatter.pageWidth = 500
       printFormatter.pageHeight = 700
       
        PrintManager.shared.printContent(printFormatter: printFormatter)
    }
    
     deinit {
            PrintManager.shared.reset()
        }
}
```

*   **No Direct `UIPrintInteractionController`:** Notice how neither `PatientSummaryViewController` nor `LabResultsViewController` directly creates an instance of `UIPrintInteractionController`. They interact with `PrintManager.shared` to initiate the print process.
*   **Formatter Creation:** Each MVC is responsible for creating the `UIPrintFormatter` tailored to its content. This is where the specifics of what’s being printed are handled.
*   **Clear Responsibility:** MVCs are focused on their presentation and data; the print logic is separated into its own layer, promoting modularity.
*   **`deinit`:** The `deinit` method releases the `UIPrintInteractionController` instance once the view controller has been deallocated.

**Code Example 3: Complex Content using `UIPrintPageRenderer`**

If the content isn't simple text, the use of `UIPrintPageRenderer` becomes essential. In my medical records application, reports often included graphs and charts, requiring more sophisticated printing.

```swift
import UIKit

class ReportPrintRenderer: UIPrintPageRenderer {
    let contentToPrint: UIView
    
    init(content: UIView) {
        self.contentToPrint = content
        super.init()
    }
    
    override func drawPage(at pageIndex: Int, in printableRect: CGRect) {
        // Convert UIView content into a PDF context for printing
        let pdfContext = UIGraphicsGetCurrentContext()!
        contentToPrint.layer.render(in: pdfContext)
    }

    override func numberOfPages() -> Int {
        return 1 // For this simplified example, assume single page
    }
}

class ImagingReportViewController: UIViewController {
    
    // ... view controller logic and image view ...

    func printReport(){
        guard let viewToPrint = self.view else {
            return
        }
        let printRenderer = ReportPrintRenderer(content: viewToPrint)
        PrintManager.shared.printContent(printFormatter: printRenderer)
    }
    
     deinit {
            PrintManager.shared.reset()
        }
}
```

*   **`ReportPrintRenderer`:** A custom `UIPrintPageRenderer` subclass that allows the printing of more complex elements, such as the content of a `UIView`. This demonstrates the necessary flexibility for anything beyond basic text.
*   **`drawPage`:** This method is where you would define how the content is rendered onto the printable page. This particular example uses `render(in:)`, but complex layouts may require more involved drawing using CoreGraphics APIs.
*   **`numberOfPages`:** You can control the number of pages required to print your complex content. This example makes the simplifying assumption that the view fits on a single page. In more complex cases, this number could vary.
*   **Flexibility:** This example shows that you can use a custom renderer for any type of view. In the medical application, this approach allowed accurate printing of the complex chart data.

**Recommendations and Best Practices:**

*   **Single Source of Truth:** The `PrintManager` should be the only entry point for triggering the printing process. Avoid creating direct `UIPrintInteractionController` instances outside of this centralized class.
*   **Error Handling:** The example doesn’t include explicit error handling for brevity; however, production applications should incorporate error checking and reporting during the print process, particularly within the completion handler.
*   **Custom `UIPrintFormatter` Subclasses:** Utilize custom `UIPrintFormatter` subclasses for complex layouts and content to achieve fine-grained control over the output.
*   **Page Management:** For multi-page printing, implement the appropriate `UIPrintPageRenderer` delegate methods to control pagination and content distribution.

Implementing AirPrint across multiple MVCs requires careful planning. By centralizing the print logic within a `PrintManager` and leveraging `UIPrintFormatter` or custom `UIPrintPageRenderer` instances, you create a more robust and maintainable solution. This approach, grounded in my experience building a complex application with multiple printing features, avoids duplication of code, promotes consistent behavior, and simplifies future development.
