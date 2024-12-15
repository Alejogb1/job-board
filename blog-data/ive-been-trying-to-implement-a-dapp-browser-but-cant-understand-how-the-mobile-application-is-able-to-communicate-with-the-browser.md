---
title: "I've been trying to implement a DApp browser, but can't understand how the Mobile application is able to communicate with the browser?"
date: "2024-12-15"
id: "ive-been-trying-to-implement-a-dapp-browser-but-cant-understand-how-the-mobile-application-is-able-to-communicate-with-the-browser"
---

it's a common head-scratcher, i've definitely been there. building a dApp browser can seem like a complex dance between a native app and a web environment, but the underlying mechanisms are quite straightforward once you break it down. basically, we're talking about inter-process communication (ipc) but in a context where one process is a web view loaded in your mobile app and another process is the native mobile application.

let's first talk about the fundamental concept: web views. in your mobile app, you're essentially embedding a stripped-down browser within your application, which is also known as a web view. this web view renders the dApp's front end, which is built with web technologies like javascript, html and css, the same way a regular browser does.

the challenge, of course, is that this web view is mostly isolated from the native part of the application. they are different worlds, different execution environments. think of it like two separate rooms in a house; they need a way to talk to each other.

now, how do we get them chatting? there are mainly two paths you can take: **message passing** via javascript bridges and **custom url schemes** handling. i'll cover both since depending on your architecture you might need both.

**message passing via javascript bridges**

this is the most common approach. the core idea revolves around creating a javascript interface that acts as a proxy between the web view and the native app. the webview exposes a specific object to javascript in your website that contains functions. these functions will trigger code written in the native environment using the bridge. the web view calls these functions, passing data, and on the native side, the app listens for these calls and executes specific code. once the native code finishes, it can send data back to the web view using similar mechanisms.

in android this is accomplished with `webview.addjavascriptinterface()` which exposes a java object with publicly available methods to the javascript environment inside the webpage. in ios this is done with `wkwebview.uiwebviewdelegate` allowing message handling from the webpage to native code.

here's a really simplified example, of what it can look like in javascript running in the webpage, let's say you have created an interface called 'mybridge':

```javascript
//javascript code in your dApp (webpage)

function sendTransaction(toAddress, amount) {
    if (window.mybridge && typeof window.mybridge.sendtransaction === 'function') {
        window.mybridge.sendtransaction(toAddress, amount);
    } else {
         console.error('native bridge is not available or sendTransaction function is undefined');
    }
}

function getAccounts() {
    if(window.mybridge && typeof window.mybridge.getaccounts === 'function') {
        window.mybridge.getaccounts();
    } else {
        console.error('native bridge is not available or getAccounts function is undefined');
    }
}

// example of sending a transaction call:
sendTransaction("0x1234....", 10);
getAccounts();

```

then in the native side you would have some code like this, again very simplified, we would need to implement some interfaces/protocols, but for the sake of simplicity:

```java

//android side java code
import android.webkit.JavascriptInterface;
import android.webkit.WebView;

class MyJavascriptInterface {

    private final WebView webview;

    MyJavascriptInterface(WebView webview) {
       this.webview = webview;
    }

    @JavascriptInterface
    public void sendTransaction(String toAddress, int amount) {
    // native code logic to handle the transaction
    System.out.println("transaction requested to: " + toAddress + " with amount: " + amount);

    // after transaction process return some data to javascript

     String transactionResult = "success";
     String jsCode = "javascript:window.dispatchEvent(new CustomEvent('transactionResponse', { detail: '" + transactionResult + "' }));";
    this.webview.post(() -> this.webview.evaluateJavascript(jsCode, null));

    }

    @JavascriptInterface
    public void getAccounts() {
    // native code to get users accounts from local storage or keychain

      String accounts = "['0x123', '0x456']";
      String jsCode = "javascript:window.dispatchEvent(new CustomEvent('accountsResponse', { detail: '" + accounts + "' }));";
      this.webview.post(() -> this.webview.evaluateJavascript(jsCode, null));

    }
}


// somewhere in your activity class
webview = findViewById(R.id.yourwebviewid);
webview.getSettings().setJavaScriptEnabled(true);
MyJavascriptInterface myInterface = new MyJavascriptInterface(webview);
webview.addJavascriptInterface(myInterface, "mybridge");

```

```swift
//ios side swift code
import WebKit

class MyJavascriptInterface: NSObject, WKScriptMessageHandler {
    weak var webview: WKWebView?

    init(webview: WKWebView){
      self.webview = webview;
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
      if message.name == "sendTransaction" {
        if let params = message.body as? Dictionary<String, Any> {
          if let toAddress = params["toAddress"] as? String, let amount = params["amount"] as? Int {
                // native code logic to handle the transaction
                  print("transaction requested to: \(toAddress) with amount: \(amount)")
              // after transaction process return some data to javascript
             let transactionResult = "success";
            self.webview?.evaluateJavaScript("window.dispatchEvent(new CustomEvent('transactionResponse', { detail: '\(transactionResult)' }));")
            }
          }
      } else if message.name == "getAccounts" {
          // native code to get users accounts from local storage or keychain
        let accounts = "['0x123', '0x456']"
        self.webview?.evaluateJavaScript("window.dispatchEvent(new CustomEvent('accountsResponse', { detail: '\(accounts)' }));")

        }
    }

}

// somewhere in your view controller
let webConfiguration = WKWebViewConfiguration()
let myInterface = MyJavascriptInterface(webview: webview)
webConfiguration.userContentController.add(myInterface, name: "sendTransaction")
webConfiguration.userContentController.add(myInterface, name: "getAccounts")
webview = WKWebView(frame: .zero, configuration: webConfiguration)
```
*note* the code above is very much simplified and some security considerations and error handling mechanisms should be implemented. specially when receiving and sending data from the webpage.

in the code snippets, i've used the `window.mybridge` name, and that is the name we have given to the interface we created in the native side when we called `addjavascriptinterface` or `userContentController.add`, it is how the webpage can access it's methods. the events `transactionResponse` and `accountsResponse` are events that are triggered once the native side finished processing the data, this events can be listened on the webpage. in short you receive a message, process it in the native side and you send the data back to the webpage using javascript execution inside the webpage.

this approach lets you securely handle sensitive operations like signing transactions on the native side, keeping your keys safe from the web view’s potentially vulnerable environment. the javascript bridge concept is heavily used in applications like metamask and trust wallet, it's a foundational piece in their architectures.

**custom url schemes**

another method you might stumble upon is custom url schemes. imagine you click on a link like `myapp://do-something?param1=value1&param2=value2`. your operating system recognizes "myapp://" as a scheme registered with your app and launches it, passing the url string along.

you can use this mechanism to send commands to your native application from javascript inside the web view. you might use this approach, for example, if you want to initiate a login with a third party application. i would recommend not using this for secure information since it will be very easy for a malicious webpage to steal sensitive information using this approach. the information is sent as strings so some formatting will be needed on both sides of the transaction to decode the information.

here's a brief javascript example of how you could trigger a custom url scheme:

```javascript
//javascript code in your dApp
function initiateLogin() {
    window.location.href = 'myapp://login?service=google';
}
function sendSomeData(data) {
   const encodedData = encodeURIComponent(JSON.stringify(data));
    window.location.href = 'myapp://transfer?data=' + encodedData;
}
```

on the native side, your app would register this custom url scheme and implement a handler that's called when it receives it. this is how, for example, metamask opens when you click on a dapp website that tries to connect to the wallet.

```java
//android side code
import android.net.Uri;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

public class MyActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.your_activity_layout);

        Uri data = getIntent().getData();
        if (data != null && data.getScheme().equals("myapp")) {
            String action = data.getHost();
            if (action.equals("login")) {
               String service = data.getQueryParameter("service");
               handleLoginRequest(service);
            } else if (action.equals("transfer")) {
               String encodedData = data.getQueryParameter("data");
                handleTransferRequest(encodedData);
            }
        }
    }

   void handleLoginRequest(String service){
   // native code to handle the login request
        System.out.println("login request from service: " + service);
   }

    void handleTransferRequest(String encodedData){
       // native code to decode and handle data
       System.out.println("transfer data: " + encodedData);
    }
}
```

```swift
//ios side code
import UIKit
class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

        if let url = self.view.window?.windowScene?.activationState.urlContexts.first?.url {
           let scheme = url.scheme
           if scheme == "myapp" {
            if let host = url.host {
               if host == "login" {
                  if let service = url.queryParameters?["service"] {
                     handleLoginRequest(service)
                   }
                } else if host == "transfer" {
                    if let encodedData = url.queryParameters?["data"] {
                        handleTransferRequest(encodedData);
                    }
                 }
                }
            }
        }
    }

    func handleLoginRequest(_ service: String) {
       //native code to handle the login request
        print("login request from service: \(service)");
    }

    func handleTransferRequest(_ encodedData: String) {
       //native code to decode data
        print("transfer data: \(encodedData)")
    }
}

//this is an extension that is usually created to read query parameters from a url
extension URL {
    var queryParameters: [String: String]? {
        guard let components = URLComponents(url: self, resolvingAgainstBaseURL: true),
              let queryItems = components.queryItems else {
            return nil
        }
        return queryItems.reduce(into: [String: String]()) { (result, item) in
            result[item.name] = item.value
        }
    }
}

```
the code above, in android, gets the url data inside the `onCreate` method, the `intent` will contain the information sent from the webpage and we can extract the scheme, host and the parameters. in ios, we are reading the url when the `viewDidAppear` method is called using `windowScene?.activationState.urlContexts`. *note* this method is only called if the application is opened using a custom scheme, therefore the `activationState` will have at least one url in its context array. after that, we just read the scheme, host and query parameters as in the android version.

**a bit of my history**

back when i was messing around with my first dApp browser, i initially relied solely on url schemes for almost everything, which, in retrospect, was a really bad idea. i quickly ran into major limitations when i tried to pass more complex data, and more importantly, i faced major security concerns. the moment i moved to a bridge implementation i saw a major increase in usability and security. it was a painful lesson learned from experience. one time the test user did some weird combination of clicks and the application crashed i had to look into the logs and find out that the url parameters where way over the allowed url length, that made the application simply crash, haha good times.

**resources for more information**

if you want a deeper understanding, i recommend exploring the webkit documentation, particularly the sections on `webview` and its apis related to message passing and url scheme handling, this applies for both android and ios. as well look into webkit's security guidelines to be extra cautious. "building mobile apps with javascript" by andrew lunny and brian leroux offers detailed insights into native bridge patterns for mobile hybrid applications, although it does not focus on blockchain, the core mechanisms are all explained there.
