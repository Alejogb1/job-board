---
title: "How can I send an iText PDF as an email attachment using Java?"
date: "2025-01-30"
id: "how-can-i-send-an-itext-pdf-as"
---
Generating and emailing PDF documents programmatically is a common requirement in many applications, and combining iText with Java's mail capabilities provides a robust solution. My experience across multiple projects dealing with document generation and distribution has highlighted the necessity of a clear, efficient approach to this process. The key is to understand how to create the PDF as a byte array in memory, rather than writing it to disk, allowing seamless attachment to an email.

First, you'll need to include the necessary libraries. iText, for PDF generation, and JavaMail, for email functionality, are critical. Typically, this will involve adding the iText 7 core library and the JavaMail API to your project's dependencies. Assuming your project uses Maven, you’d include entries similar to:

```xml
<dependency>
    <groupId>com.itextpdf</groupId>
    <artifactId>itext7-core</artifactId>
    <version>7.2.5</version> <!-- Adjust to your iText version -->
</dependency>
<dependency>
    <groupId>com.sun.mail</groupId>
    <artifactId>javax.mail</artifactId>
    <version>1.6.2</version> <!-- Adjust to your JavaMail version -->
</dependency>
```

With dependencies resolved, the core process involves these three stages: PDF generation with iText, email construction with JavaMail, and attachment of the generated PDF data.

**1. PDF Generation**

Instead of writing the PDF to a file, we will use a `ByteArrayOutputStream` which lets you write the PDF into memory. This stream will be the source of the PDF byte array.  This approach eliminates intermediate file creation and simplifies the process significantly. Here is a demonstration:

```java
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class PdfGenerator {

    public byte[] generatePdf(String content) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            PdfWriter writer = new PdfWriter(baos);
            PdfDocument pdf = new PdfDocument(writer);
            Document document = new Document(pdf);
            document.add(new Paragraph(content));
            document.close();
            return baos.toByteArray();
        }
    }
}
```
*Commentary:* This method, `generatePdf`, demonstrates a minimal iText implementation. A `ByteArrayOutputStream` is created, which captures the output of the PDF writer. A `PdfDocument` is then created using this writer, and finally a `Document` which allows adding content via `Paragraph` objects. When the document is closed, the PDF data is available in the `ByteArrayOutputStream`.  The `.toByteArray()` method obtains the raw bytes, which are then returned from the method.

**2. Email Construction**

JavaMail's structure is built around the `Session`, `Message`, and `Transport` classes. A `Session` encapsulates the email properties (e.g., server address, authentication credentials). The `Message` defines the email's content, recipients, and attachments. `Transport` is responsible for sending the message via SMTP. Here's an example email construction:

```java
import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import java.util.Properties;

public class EmailSender {

    public void sendEmailWithAttachment(String to, String subject, byte[] pdfData, String filename, String username, String password) throws MessagingException {
        Properties props = new Properties();
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.starttls.enable", "true");
        props.put("mail.smtp.host", "smtp.example.com");  // Replace with your SMTP server
        props.put("mail.smtp.port", "587");              // Replace with your SMTP port

        Session session = Session.getInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication(username, password);
            }
        });

        Message message = new MimeMessage(session);
        message.setFrom(new InternetAddress(username));
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setSubject(subject);

        MimeMultipart multipart = new MimeMultipart();

        MimeBodyPart textPart = new MimeBodyPart();
        textPart.setText("Please find the attached PDF document.", "UTF-8");
        multipart.addBodyPart(textPart);

        MimeBodyPart attachmentPart = new MimeBodyPart();
        attachmentPart.setContent(pdfData, "application/pdf");
        attachmentPart.setFileName(filename);
        multipart.addBodyPart(attachmentPart);

        message.setContent(multipart);
        Transport.send(message);
    }
}
```

*Commentary:* This method sets up the email properties and authentication. It defines the recipient, subject, and the email body. A `MimeMultipart` structure is crucial for including attachments within an email.  First a simple text part is added, and then the PDF data is added as an attachment via `MimeBodyPart`. The `setContent` method is important - its content type parameter specifies the data type as `application/pdf` for proper handling by email clients.  Finally, the `setFileName` sets the name of the attachment as it will appear to the user. The `Transport.send` method does the actual sending over SMTP.

**3. Integration**

Combining the PDF generation and email sending is straightforward, as shown in the following combined example:

```java
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;

import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Properties;

public class PdfEmailer {

    public static void main(String[] args) {
        String recipientEmail = "recipient@example.com";   // Replace with recipient's email
        String emailSubject = "Generated PDF Document";
        String pdfContent = "This is a dynamically generated PDF document.";
        String smtpUsername = "your_email@example.com";   // Replace with your email
        String smtpPassword = "your_password";           // Replace with your password

        try {
            byte[] pdfData = generatePdf(pdfContent);
            sendEmailWithAttachment(recipientEmail, emailSubject, pdfData, "generated_document.pdf", smtpUsername, smtpPassword);
            System.out.println("Email with PDF attachment sent successfully.");
        } catch (MessagingException | IOException e) {
           System.err.println("Error processing PDF and Emailing: " + e.getMessage());
        }
    }

    private static byte[] generatePdf(String content) throws IOException {
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            PdfWriter writer = new PdfWriter(baos);
            PdfDocument pdf = new PdfDocument(writer);
            Document document = new Document(pdf);
            document.add(new Paragraph(content));
            document.close();
            return baos.toByteArray();
        }
    }

    private static void sendEmailWithAttachment(String to, String subject, byte[] pdfData, String filename, String username, String password) throws MessagingException {
         Properties props = new Properties();
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.starttls.enable", "true");
        props.put("mail.smtp.host", "smtp.example.com");  // Replace with your SMTP server
        props.put("mail.smtp.port", "587");              // Replace with your SMTP port

        Session session = Session.getInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication(username, password);
            }
        });

        Message message = new MimeMessage(session);
        message.setFrom(new InternetAddress(username));
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setSubject(subject);

        MimeMultipart multipart = new MimeMultipart();

        MimeBodyPart textPart = new MimeBodyPart();
        textPart.setText("Please find the attached PDF document.", "UTF-8");
        multipart.addBodyPart(textPart);

        MimeBodyPart attachmentPart = new MimeBodyPart();
        attachmentPart.setContent(pdfData, "application/pdf");
        attachmentPart.setFileName(filename);
        multipart.addBodyPart(attachmentPart);

        message.setContent(multipart);
        Transport.send(message);
    }
}
```
*Commentary:*  The `main` method here orchestrates the process.  It initializes example data,  calls the `generatePdf` method to obtain the PDF byte array. This byte array is then passed along with other email parameters into `sendEmailWithAttachment`. This encapsulates all the necessary logic for sending a PDF via email. Error handling is included for catching potential `IOExceptions` from PDF generation and `MessagingExceptions` from email sending.

**Resource Recommendations:**

To deepen your knowledge of iText, I recommend exploring the official iText 7 documentation, which is meticulously maintained and provides detailed information on all aspects of the library, including more advanced layout and styling techniques. Additionally, the JavaMail API documentation is valuable for understanding the intricacies of email communication in Java, including authentication mechanisms, different mail protocols, and advanced messaging features. Consider exploring articles and tutorials from online learning platforms that address Java email and PDF manipulation techniques as they can provide hands-on, practical advice. These resources, coupled with experimentation, will provide the basis for successful PDF attachment email operations.
