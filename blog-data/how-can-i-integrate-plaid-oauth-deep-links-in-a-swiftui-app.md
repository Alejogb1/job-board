---
title: "How can I integrate Plaid OAuth deep links in a SwiftUI app?"
date: "2024-12-23"
id: "how-can-i-integrate-plaid-oauth-deep-links-in-a-swiftui-app"
---

, let's tackle this Plaid OAuth deep linking challenge in SwiftUI. It's something I've navigated a few times, and while the initial setup can feel a bit tangled, it becomes pretty manageable once you understand the underlying flow and how SwiftUI handles deep links. I recall one project, specifically, where we were building a financial management app, and getting this precisely implemented was crucial for a smooth user experience. We had a significant amount of back-and-forth trying to get the redirect handling ironed out across various devices. Let me share some insights from that experience, and provide a structured approach to integrating Plaid's OAuth deep links.

The core of the issue lies in seamlessly transitioning the user from Plaid's web-based Link flow back into your application after they've completed the authentication process. Plaid initiates the authentication in a web browser or a web view, and, upon completion, redirects the user to a predefined URL scheme specified during your Plaid application configuration. This redirection is handled by your application's deep linking mechanism.

In SwiftUI, deep link handling primarily involves the `.onOpenURL` view modifier and the associated `URL` object. First things first, you need to properly configure your Xcode project to recognize the custom URL scheme you'll be using for Plaid's redirects. This usually involves going into your project's target settings, under the "Info" tab, and adding a new "URL Types" entry. You'd specify your URL scheme in the "URL Schemes" field (e.g., `yourapp-plaid`). This part's critical, as without it the system won't know where to route the redirect.

Now, diving into the SwiftUI code, let's consider a simplified example where we initiate the Plaid Link flow and then handle the redirect. We'll presume you’re using a modal view to display the Plaid Link SDK or a webview as required, after triggering a button click.

```swift
import SwiftUI

struct ContentView: View {
    @State private var showingPlaidLink = false
    @State private var plaidPublicToken: String? = nil

    var body: some View {
        VStack {
            Text("Plaid Authentication")
                .font(.title)
            Button("Open Plaid Link") {
                self.showingPlaidLink = true
                // Simulate starting the Plaid Link flow, typically by fetching an access token.
                // Instead of opening a webview, this example simulates the flow.
                // In a real app, you'd replace this with logic to fetch a link token and launch Plaid Link
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                   handleMockPlaidRedirect()
                }
            }
            .padding()

            if let token = plaidPublicToken {
                Text("Public Token: \(token)")
                    .padding()
            }

            Text("Status: \(plaidPublicToken == nil ? "Pending" : "Connected")")
                .padding()
            
            .onOpenURL { url in
                handlePlaidRedirect(url: url)
            }
        }
        .sheet(isPresented: $showingPlaidLink) {
             // Replace with your Plaid Link initialization code
             // A webview could be used here
            Text("Plaid Link Modal Placeholder")
        }
    }


    private func handleMockPlaidRedirect() {
         //  Mock the Plaid redirect URL.
           let redirectUrl = URL(string: "yourapp-plaid://?oauth_state=state_value&public_token=mock_public_token")!
         handlePlaidRedirect(url: redirectUrl)
    }


    private func handlePlaidRedirect(url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: true),
              let queryItems = components.queryItems else {
            print("Invalid Plaid redirect URL")
            return
        }

        var parameters = [String: String]()
        for item in queryItems {
            parameters[item.name] = item.value
        }

        if let publicToken = parameters["public_token"] {
             self.plaidPublicToken = publicToken
            print("Public Token received: \(publicToken)")
             showingPlaidLink = false
        } else {
            print("Error: Could not find public_token in the redirect url")
        }

        // In a real implementation you'd likely have error handling and also validate the `oauth_state`
        // parameter against the one you initially sent
        // before exchanging the public_token for an access token
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

```
In this first snippet, we’ve defined a `ContentView` with a button to 'Open Plaid Link'. The `onOpenURL` modifier will capture the Plaid redirect URL. I've also added a mock version of the handlePlaidRedirect to simulate what happens after a successful Plaid link connection. This version of the code also stores the public_token. The actual Plaid Link flow would need to be implemented by you, typically involving a view containing a webview or using Plaid's SDK. The `handlePlaidRedirect` method parses the incoming url, extracting the public token and error handling. In practice, you would also use the oauth_state, for verification. The public token would then be exchanged for an access token which allows the application to make authorized requests to the Plaid API.

Now, let’s address how you would typically present the Plaid Link screen, for example using a `SFSafariViewController`. Here's how you could modify the previous example to use a Safari View Controller and handle the callback within the view:

```swift
import SwiftUI
import SafariServices

struct ContentView: View {
    @State private var presentingSafariView = false
    @State private var plaidPublicToken: String? = nil
     //  For simplicity, this is a hard coded URL for the example.
     //  You would dynamically create this with the `link_token` from Plaid.
    private let plaidLinkURL = URL(string: "https://link.plaid.com/i/e099d90a-1b61-4138-9d42-ef0528a64e8b?oauth_state=state_value&client_name=Test+App&is_web=true&is_mobile=true")!

    var body: some View {
        VStack {
            Text("Plaid Authentication")
                .font(.title)
            Button("Open Plaid Link") {
                self.presentingSafariView = true
            }
            .padding()

            if let token = plaidPublicToken {
                Text("Public Token: \(token)")
                    .padding()
            }

            Text("Status: \(plaidPublicToken == nil ? "Pending" : "Connected")")
                .padding()
             .onOpenURL { url in
                 handlePlaidRedirect(url: url)
             }
        }
        .sheet(isPresented: $presentingSafariView) {
           SafariView(url: plaidLinkURL) { result in
             if case .completed = result {
                 print("Safari View Controller dismissed")
                  // The redirect is handled via the .onOpenURL modifier, not here.
             }
           }
        }
    }

    private func handlePlaidRedirect(url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: true),
              let queryItems = components.queryItems else {
            print("Invalid Plaid redirect URL")
            return
        }

        var parameters = [String: String]()
        for item in queryItems {
            parameters[item.name] = item.value
        }

        if let publicToken = parameters["public_token"] {
            self.plaidPublicToken = publicToken
            print("Public Token received: \(publicToken)")
            self.presentingSafariView = false //Dismiss the safari view after a successful redirect.
        } else {
            print("Error: Could not find public_token in the redirect url")
        }

       // In a real implementation you'd likely have error handling and also validate the `oauth_state`
       // parameter against the one you initially sent
       // before exchanging the public_token for an access token.

    }
}

struct SafariView: UIViewControllerRepresentable {
    let url: URL
    let onDismiss: (SFSafariViewController.DismissalReason) -> Void

    func makeUIViewController(context: UIViewControllerRepresentableContext<SafariView>) -> SFSafariViewController {
        let config = SFSafariViewController.Configuration()
        config.entersReaderIfAvailable = true
        let sfViewController = SFSafariViewController(url: url, configuration: config)
        return sfViewController
    }

    func updateUIViewController(_ uiViewController: SFSafariViewController, context: UIViewControllerRepresentableContext<SafariView>) {
    }

    func makeCoordinator() -> Coordinator {
            Coordinator(self)
    }

    class Coordinator: NSObject, SFSafariViewControllerDelegate {
        let parent: SafariView

        init(_ parent: SafariView) {
                self.parent = parent
        }

        func safariViewControllerDidFinish(_ controller: SFSafariViewController) {
          parent.onDismiss(.completed)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

```

In this version, we present Plaid using `SFSafariViewController`. The `SafariView` struct makes it compatible with SwiftUI, and our `onOpenURL` modifier within `ContentView` still catches the redirect, parsing the returned public token.

Finally, for better state management, especially if you need to make API calls to exchange the public token, consider using a dedicated class or struct conforming to `ObservableObject`. This would keep your view layer clean, while managing the data and API interactions. Here is a basic example of how you would create such a manager, along with a minor tweak to `ContentView`:

```swift
import SwiftUI
import Combine

class PlaidManager: ObservableObject {
    @Published var publicToken: String?
    @Published var connectionStatus: String = "Pending"
    @Published var hasError: Bool = false
    @Published var errorMessage: String = ""
    
    func handlePlaidRedirect(url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: true),
              let queryItems = components.queryItems else {
            self.hasError = true
            self.errorMessage = "Invalid Plaid redirect URL"
            print("Invalid Plaid redirect URL")
            return
        }

         var parameters = [String: String]()
        for item in queryItems {
            parameters[item.name] = item.value
        }

        if let publicToken = parameters["public_token"] {
            self.publicToken = publicToken
             self.connectionStatus = "Connected"
            print("Public Token received: \(publicToken)")
        } else {
             self.hasError = true
             self.errorMessage = "Error: Could not find public_token in the redirect url"
             print("Error: Could not find public_token in the redirect url")
        }
         // Perform the exchange of the public_token for the access_token.
        //  This example doesn't perform any requests.

    }
}

struct ContentView: View {
    @State private var presentingSafariView = false
    @ObservedObject var plaidManager = PlaidManager()
      //  For simplicity, this is a hard coded URL for the example.
      //  You would dynamically create this with the `link_token` from Plaid.
    private let plaidLinkURL = URL(string: "https://link.plaid.com/i/e099d90a-1b61-4138-9d42-ef0528a64e8b?oauth_state=state_value&client_name=Test+App&is_web=true&is_mobile=true")!

    var body: some View {
        VStack {
            Text("Plaid Authentication")
                .font(.title)
            Button("Open Plaid Link") {
                self.presentingSafariView = true
            }
            .padding()
             if let token = plaidManager.publicToken {
                  Text("Public Token: \(token)")
                     .padding()
            }
           Text("Status: \(plaidManager.connectionStatus)")
               .padding()

           if plaidManager.hasError {
               Text("Error: \(plaidManager.errorMessage)")
                  .padding()
                  .foregroundColor(.red)
           }
            .onOpenURL { url in
                  plaidManager.handlePlaidRedirect(url: url)
             }
        }
        .sheet(isPresented: $presentingSafariView) {
             SafariView(url: plaidLinkURL) { result in
              if case .completed = result {
                 print("Safari View Controller dismissed")
                 // The redirect is handled via the .onOpenURL modifier, not here.
                }
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
```

This third example demonstrates the use of an `ObservableObject` called `PlaidManager` to handle state updates after a successful Plaid Link process. You should also implement error handling, including checking for invalid `oauth_state` parameters before exchanging the public token for an access token.

For deeper understanding, I'd highly recommend looking at Apple's official documentation for `URL` handling and the `onOpenURL` modifier in SwiftUI. Specifically, search for the `UIApplicationDelegate` protocol and how it relates to deep linking, even though SwiftUI handles it a bit differently. Additionally, Plaid's own developer documentation on deep linking and OAuth is essential for a complete understanding of the Plaid side of things. Finally, the book "Combine: Asynchronous Programming with Swift" by Donny Wals is excellent for improving your understanding of reactive programming, and how that can improve the management of asynchronous tasks.

These three code samples, along with the outlined recommendations, will give you a solid foundation for integrating Plaid OAuth deep links into your SwiftUI application. The key is understanding the flow and ensuring your URL scheme is properly configured. The rest comes down to clean implementation and managing the asynchronous nature of the authentication process with a solid approach to state management.
