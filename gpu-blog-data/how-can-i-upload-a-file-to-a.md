---
title: "How can I upload a file to a Mailgun server using Swift?"
date: "2025-01-30"
id: "how-can-i-upload-a-file-to-a"
---
The Mailgun API requires a multipart form request when sending email with file attachments, deviating from simpler JSON-based API interactions. I've frequently encountered this while developing iOS applications requiring robust email functionality, specifically handling user-generated attachments. Successfully uploading a file to Mailgun involves crafting a correctly formatted HTTP request, including specifying the proper content type and boundary within the request body. Swift's `URLSession` class provides the necessary tools, although the process can be nuanced.

The fundamental challenge lies in constructing the multipart form data. It’s not merely appending data; each element, including text fields and file attachments, must be carefully encoded with specific headers and a designated boundary string separating distinct parts. We'll explore this process in detail and demonstrate how to implement it correctly. Incorrect formatting will result in failed requests and debugging headaches. I've spent considerable time troubleshooting these types of server requests, and proper understanding is crucial.

Here's a step-by-step breakdown of the process, followed by code examples:

**1. Setting Up the URL and Request:**

First, create a `URL` object pointing to the Mailgun message API endpoint, which typically looks something like `https://api.mailgun.net/v3/YOUR_DOMAIN/messages`. You'll need to replace `YOUR_DOMAIN` with your actual Mailgun domain. Construct a `URLRequest` object from this URL, setting the method to `POST` and including the necessary authentication. Mailgun uses basic authentication, requiring your `api` key to be encoded with your domain in the format “api:YOUR_API_KEY.” This string needs to be Base64 encoded and included in the `Authorization` header of the request.

**2. Creating the Multipart Form Data:**

This is the core step, and the most error-prone. Begin by creating a `NSMutableData` object to hold the request body. A crucial element is a randomly generated boundary string. This string should be complex and unique to prevent accidental conflicts within the form data. Generate the boundary, then append each form field (recipient, sender, subject, body) as individual parts to the `NSMutableData`. Each part should adhere to this format:

```
--YOUR_BOUNDARY_STRING
Content-Disposition: form-data; name="FIELD_NAME"

FIELD_VALUE
```

For file attachments, the format changes slightly. The format for a file attachment is:

```
--YOUR_BOUNDARY_STRING
Content-Disposition: form-data; name="attachment"; filename="FILE_NAME"
Content-Type: MIME_TYPE

FILE_DATA
```
Here, `FILE_NAME` is the name of your file, `MIME_TYPE` is the file's content type (e.g., `image/jpeg`), and `FILE_DATA` is the raw binary data of the file. Remember to append a carriage return and line feed (`\r\n`) after each line and before the end boundary to satisfy correct HTTP formatting. Finally, append the closing boundary string, prefixed with two dashes, to the end of your data.

**3. Configuring the `URLSession` and Uploading Data:**

Create a `URLSession` and a `URLSessionUploadTask`. The upload task takes the `URLRequest` and the multipart form data created in the previous step. Execute the task, and handle the response.  Responses from the Mailgun API are typically JSON, which you can decode to verify successful submission or identify errors.

**Code Examples:**

Here are three code snippets demonstrating the process. The first shows building the multipart request body, the second how to create and start the `URLSession` upload task and the final snippet provides an example of handling the server’s response.

**Example 1: Constructing Multipart Form Data**

```swift
func createMultipartFormData(to: String, from: String, subject: String, body: String, fileData: Data?, fileName: String?, mimeType: String?) -> Data? {

    let boundary = UUID().uuidString
    let body = NSMutableData()
    let parameters = ["to": to, "from": from, "subject": subject, "text": body]


    for (key, value) in parameters {
      body.appendString("--\(boundary)\r\n")
      body.appendString("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
      body.appendString("\(value)\r\n")
    }

    if let fileData = fileData, let fileName = fileName, let mimeType = mimeType {
      body.appendString("--\(boundary)\r\n")
      body.appendString("Content-Disposition: form-data; name=\"attachment\"; filename=\"\(fileName)\"\r\n")
      body.appendString("Content-Type: \(mimeType)\r\n\r\n")
      body.append(fileData)
      body.appendString("\r\n")
    }

    body.appendString("--\(boundary)--\r\n")

    return body as Data
}

extension NSMutableData {
    func appendString(_ string: String) {
        if let data = string.data(using: .utf8) {
            self.append(data)
        }
    }
}
```
**Commentary:** This function constructs the multipart form data. It takes the email recipient, sender, subject, and body as text parameters and optionally file data and file information. It iterates through the email parameters, appending them to the request body with the correct formatting and boundary separation. Finally it handles the file attachment if provided appending the file data and its metadata.

**Example 2: Creating and Executing the `URLSession` Upload Task**
```swift

func uploadFileToMailgun(to: String, from: String, subject: String, body: String, fileData: Data?, fileName: String?, mimeType: String?, apiKey: String, domain: String, completion: @escaping (Result<Data, Error>) -> Void) {

    let urlString = "https://api.mailgun.net/v3/\(domain)/messages"
    guard let url = URL(string: urlString) else {
        completion(.failure(NSError(domain: "Invalid URL", code: -1, userInfo: nil)))
        return
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"

    let authString = "api:\(apiKey)"
    let authData = authString.data(using: .utf8)!.base64EncodedString()
    request.setValue("Basic \(authData)", forHTTPHeaderField: "Authorization")
    let multipartData = createMultipartFormData(to: to, from: from, subject: subject, body: body, fileData: fileData, fileName: fileName, mimeType: mimeType)

    guard let data = multipartData else {
      completion(.failure(NSError(domain: "Multipart data error", code: -2, userInfo: nil)))
      return
    }
    request.setValue("multipart/form-data; boundary=\(UUID().uuidString)", forHTTPHeaderField: "Content-Type")

    let task = URLSession.shared.uploadTask(with: request, from: data) { data, response, error in
      if let error = error {
          completion(.failure(error))
          return
      }
      guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            completion(.failure(NSError(domain: "Invalid HTTP Status Code: \(statusCode)", code: -3, userInfo: nil)))
           return
        }

        if let data = data{
            completion(.success(data))
        }else {
            completion(.failure(NSError(domain: "No data in server response", code: -4, userInfo: nil)))
        }

    }
    task.resume()
}

```

**Commentary:** This function handles the core request logic. It takes the email parameters and file data, along with the API key and domain. It creates a URLRequest, sets the authentication header and the content type header including the boundary. The function then uses `URLSession` to upload the data, providing a completion handler that provides the result of the upload. Critically, the error handling is included here checking for a valid HTTP response and ensures a completion handler is called in all cases, either with data, or an error.

**Example 3: Handling Server Response:**

```swift
// Example usage:
func sendEmailWithAttachment() {
  let apiKey = "YOUR_API_KEY"
  let domain = "YOUR_DOMAIN"
  let recipient = "recipient@example.com"
  let sender = "sender@example.com"
  let subject = "Test Email with Attachment"
  let body = "This is a test email with an attached image."

  guard let image = UIImage(named: "testImage.png"),
        let imageData = image.pngData()
  else{
    print("Error loading image")
    return
  }

  uploadFileToMailgun(to: recipient, from: sender, subject: subject, body: body, fileData: imageData, fileName: "testImage.png", mimeType: "image/png", apiKey: apiKey, domain: domain) { result in
        switch result {
        case .success(let data):
             if let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
               if let message = json["message"] as? String{
                  print("Success: \(message)")
               }
             }else{
                print("Success: server responded with:\n \(String(data: data, encoding: .utf8) ?? "Unable to interpret data")")
             }
        case .failure(let error):
            print("Failure: \(error)")
        }
    }
}

```

**Commentary:** This example demonstrates how to call the previous `uploadFileToMailgun` function, incorporating a real image for the attachment. Upon a successful upload, it attempts to decode the response JSON, extracting and printing the message field of the response, providing a user-friendly output. If an error occurs the error is printed, allowing for debugging. Note that you'll need to replace `YOUR_API_KEY`, `YOUR_DOMAIN`,  `recipient@example.com`, `sender@example.com` and  `testImage.png` with your actual details to test it.

**Resource Recommendations:**

For deeper understanding, consult the official Mailgun API documentation. Pay close attention to the sections describing the message endpoint and the specifics of multipart form data formatting. Numerous online resources explain the fundamentals of HTTP requests and `URLSession`, which you should understand well to effectively work with network calls. Finally, researching common pitfalls when building multipart form requests can be invaluable for avoiding common errors I’ve encountered in previous projects. I found that an iterative testing approach is highly beneficial when debugging server request issues.
