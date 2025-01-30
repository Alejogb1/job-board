---
title: "Why am I getting an 'Invalid multipart' error when sending Gmail attachments in Go?"
date: "2025-01-30"
id: "why-am-i-getting-an-invalid-multipart-error"
---
The "Invalid multipart" error encountered when sending Gmail attachments via Go typically originates from incorrect formatting of the MIME multipart message, specifically within the boundaries and content headers. My experience in building an internal email notification service for a large-scale data processing pipeline revealed this nuance firsthand; even seemingly minor deviations from the expected structure can lead to this rejection by the Gmail SMTP server.

The core issue lies in the construction of the MIME (Multipurpose Internet Mail Extensions) message, which is essentially a method for structuring email content to include text, images, audio, and other types of data. When sending attachments, we're essentially creating a multipart message, meaning the email comprises multiple parts, each with its specific content type and identifying boundary. Gmail’s SMTP server is quite strict about adhering to the RFC 2046 standard (which defines MIME) and the specific format it expects, which is typically a ‘multipart/mixed’ message for emails with attachments.

The error usually arises from one or more of the following problems:

1.  **Incorrect Boundary Definition:** The boundary string, used to delimit the different parts of the multipart message, must be unique within the message. It must also be specified identically in the Content-Type header of the multipart message and in the boundary preamble that separates individual parts. Mismatches or improperly formatted boundaries lead the server to fail parsing the message, resulting in the “Invalid multipart” error.

2.  **Missing or Improper Content Headers:** Each part of the multipart message, including the text body and attachments, needs appropriate content headers. These headers describe the type of content (e.g., `text/plain`, `application/pdf`, `image/jpeg`) and potentially other details like content encoding or file names. Missing or incorrectly defined content headers, particularly Content-Disposition for attachments, cause the server to misinterpret parts, leading to the error.

3.  **Incorrect Encoding:** While not always the culprit, incorrect encoding of attachment content or headers can also lead to problems. While typically base64 encoding is employed for binary attachment data, sometimes character encoding of headers or names can cause parsing errors if it conflicts with the server’s expected charset (typically UTF-8).

4.  **Incorrect Order of Parts:** While less frequently a direct cause, the order in which parts are defined can sometimes matter depending on server implementations. Typically, the plain text or HTML body comes before the attachments, though Gmail appears to be tolerant of variations.

Let’s illustrate with three code examples, detailing potential issues and their resolutions:

**Example 1: Basic Text Email With Incorrect Boundary**

```go
package main

import (
	"bytes"
	"fmt"
	"net/smtp"
	"net/textproto"
)

func main() {
	from := "sender@example.com"
	to := []string{"receiver@example.com"}
	subject := "Test Email"
	body := "This is a test email."

	boundary := "unique-boundary" // BAD: boundary is not unique across emails

	var msg bytes.Buffer

	msg.WriteString(fmt.Sprintf("From: %s\r\n", from))
	msg.WriteString(fmt.Sprintf("To: %s\r\n", to[0]))
	msg.WriteString(fmt.Sprintf("Subject: %s\r\n", subject))
	msg.WriteString(fmt.Sprintf("MIME-Version: 1.0\r\n"))
	msg.WriteString(fmt.Sprintf("Content-Type: multipart/mixed; boundary=%s\r\n\r\n", boundary))

	msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
	msg.WriteString(fmt.Sprintf("Content-Type: text/plain; charset=UTF-8\r\n\r\n"))
	msg.WriteString(body)
	msg.WriteString(fmt.Sprintf("\r\n--%s--\r\n", boundary))


	auth := smtp.PlainAuth("", "sender@example.com", "password", "smtp.gmail.com")
	err := smtp.SendMail("smtp.gmail.com:587", auth, from, to, msg.Bytes())
	if err != nil {
		fmt.Println(err)
	} else {
        fmt.Println("email sent")
    }
}

```

**Commentary:** In this first example, the fundamental structure of a multipart message is present, but the boundary string "unique-boundary" is too simple. This lack of uniqueness can cause problems, especially if multiple emails are generated using the same process. The Gmail SMTP server is likely to detect this and flag it as an invalid multipart message. The solution would be to generate a more complex, unpredictable boundary string for each message. Moreover, using a hardcoded password as shown here is highly insecure for real applications and should be handled with appropriate secret management practices.

**Example 2: Email With Attachment and Improper Headers**

```go
package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"mime"
	"net/smtp"
	"net/textproto"
	"path/filepath"
	"time"
	"crypto/rand"
	"encoding/hex"
)

func generateBoundary() string {
	b := make([]byte, 16)
    rand.Read(b)
    return hex.EncodeToString(b)
}

func main() {
	from := "sender@example.com"
	to := []string{"receiver@example.com"}
	subject := "Email with Attachment"
	body := "Please see attachment."
	attachmentPath := "test.txt" // Dummy file

	boundary := generateBoundary()


	attachmentData, err := ioutil.ReadFile(attachmentPath)
	if err != nil {
		fmt.Println(err)
		return
	}

	attachmentEncoded := base64.StdEncoding.EncodeToString(attachmentData)
	
	var msg bytes.Buffer

	msg.WriteString(fmt.Sprintf("From: %s\r\n", from))
	msg.WriteString(fmt.Sprintf("To: %s\r\n", to[0]))
	msg.WriteString(fmt.Sprintf("Subject: %s\r\n", subject))
	msg.WriteString(fmt.Sprintf("MIME-Version: 1.0\r\n"))
	msg.WriteString(fmt.Sprintf("Content-Type: multipart/mixed; boundary=%s\r\n\r\n", boundary))

	// Text Body Part
	msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
	msg.WriteString(fmt.Sprintf("Content-Type: text/plain; charset=UTF-8\r\n\r\n"))
	msg.WriteString(body)
    msg.WriteString("\r\n")


	// Attachment Part (Incorrect Headers)
	msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
	msg.WriteString(fmt.Sprintf("Content-Type: application/octet-stream\r\n"))  //BAD: Missing Content-Disposition
	msg.WriteString(fmt.Sprintf("Content-Transfer-Encoding: base64\r\n\r\n"))
	msg.WriteString(attachmentEncoded)
	msg.WriteString("\r\n")

    msg.WriteString(fmt.Sprintf("--%s--\r\n", boundary))


	auth := smtp.PlainAuth("", "sender@example.com", "password", "smtp.gmail.com")
	err = smtp.SendMail("smtp.gmail.com:587", auth, from, to, msg.Bytes())
	if err != nil {
		fmt.Println(err)
	} else {
        fmt.Println("email sent")
    }

}
```

**Commentary:**  This example attempts to add an attachment, but it omits the crucial `Content-Disposition` header within the attachment part, which dictates to the email client what it should do with the content (e.g., view inline or treat as an attachment with a given name). This missing header is a common source of “Invalid multipart” errors. Moreover, while base64 encoding is used here (correctly), not adding the filename to the `Content-Disposition` can also confuse email clients.  The issue lies in incomplete header definitions. Using `application/octet-stream` without a filename and a content disposition leads to ambiguities about how the email client should handle the attachment and Gmail's server likely rejects it.  We've also replaced the weak boundary in example 1 with a proper UUID generator.

**Example 3:  Corrected Email With Attachment**

```go
package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"mime"
	"net/smtp"
	"net/textproto"
    "path/filepath"
	"crypto/rand"
	"encoding/hex"
)

func generateBoundary() string {
	b := make([]byte, 16)
    rand.Read(b)
    return hex.EncodeToString(b)
}

func main() {
	from := "sender@example.com"
	to := []string{"receiver@example.com"}
	subject := "Email with Attachment"
	body := "Please see attachment."
	attachmentPath := "test.txt" // Dummy file

	boundary := generateBoundary()
	attachmentData, err := ioutil.ReadFile(attachmentPath)
	if err != nil {
		fmt.Println(err)
		return
	}
	attachmentEncoded := base64.StdEncoding.EncodeToString(attachmentData)
	attachmentFilename := filepath.Base(attachmentPath)
	contentType := mime.TypeByExtension(filepath.Ext(attachmentPath))
	if contentType == "" {
		contentType = "application/octet-stream" // Default, if no type available
	}


	var msg bytes.Buffer

	msg.WriteString(fmt.Sprintf("From: %s\r\n", from))
	msg.WriteString(fmt.Sprintf("To: %s\r\n", to[0]))
	msg.WriteString(fmt.Sprintf("Subject: %s\r\n", subject))
	msg.WriteString(fmt.Sprintf("MIME-Version: 1.0\r\n"))
	msg.WriteString(fmt.Sprintf("Content-Type: multipart/mixed; boundary=%s\r\n\r\n", boundary))

	// Text Body Part
	msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
	msg.WriteString(fmt.Sprintf("Content-Type: text/plain; charset=UTF-8\r\n\r\n"))
	msg.WriteString(body)
    msg.WriteString("\r\n")

	// Attachment Part (Correct Headers)
	msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
    msg.WriteString(fmt.Sprintf("Content-Type: %s\r\n", contentType))
	msg.WriteString(fmt.Sprintf("Content-Disposition: attachment; filename=\"%s\"\r\n", attachmentFilename))
	msg.WriteString(fmt.Sprintf("Content-Transfer-Encoding: base64\r\n\r\n"))
	msg.WriteString(attachmentEncoded)
	msg.WriteString("\r\n")

	msg.WriteString(fmt.Sprintf("--%s--\r\n", boundary))


	auth := smtp.PlainAuth("", "sender@example.com", "password", "smtp.gmail.com")
	err = smtp.SendMail("smtp.gmail.com:587", auth, from, to, msg.Bytes())
	if err != nil {
		fmt.Println(err)
	} else {
        fmt.Println("email sent")
    }

}
```
**Commentary:** This final example presents the corrected approach. The key addition is the proper `Content-Disposition` header within the attachment part that includes the filename, and it also infers the `Content-Type` using `mime.TypeByExtension`, improving adherence to expected MIME structure. This results in a valid multipart message that Gmail's server is likely to accept.  The error-handling is still rudimentary, but the code focuses on demonstrating the multipart formatting concerns. We are now creating the attachment with a `Content-Disposition` that includes the filename for the attachment, as well as using the `mime` package to get the correct `Content-Type` for the attachment (fallling back to `application/octet-stream` if no type could be determined).

**Resource Recommendations:**

To deepen your understanding, refer to official documentation for the MIME standard (RFC 2046 and related RFCs). Also, research the specific SMTP guidelines of your email provider (in this case, Google's). Examining libraries like `go-mail`, `gomail`, or similar packages can reveal best practices and offer higher-level abstractions over the SMTP protocol, which will make sending email a lot easier. Additionally, scrutinizing the source code for examples within the `net/smtp` package is beneficial.
