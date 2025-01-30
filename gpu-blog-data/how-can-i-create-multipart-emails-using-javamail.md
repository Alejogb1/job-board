---
title: "How can I create multipart emails using JavaMail?"
date: "2025-01-30"
id: "how-can-i-create-multipart-emails-using-javamail"
---
Multipart emails, a requirement for rich content and attachments, often present a stumbling block for developers unfamiliar with the JavaMail API. My experience building a system for generating complex transactional emails highlighted the nuances involved in structuring these messages correctly. The core principle lies in understanding the `javax.mail.internet.MimeMultipart` class, which acts as a container for different parts of an email, each with its own content type and potentially encoded data. This container, unlike a simple text body, allows you to combine textual descriptions, embedded images, file attachments, and even alternative text formats for different viewing capabilities within the same email.

The key to constructing a multipart email involves creating a `MimeMultipart` object, adding different `MimeBodyPart` instances to it, and then setting this multipart object as the content of the `MimeMessage`. Each `MimeBodyPart` represents a single part of the email and is configured with the content type that describes the nature of the data it holds (e.g., text/plain, text/html, image/jpeg, application/pdf). Proper handling of content types and encoding is paramount, otherwise, email clients may display content incorrectly, or attachments may not be decoded as intended. This is especially crucial for encoding image data, which usually requires base64 encoding. The ordering of these parts can also matter, particularly with HTML emails that reference embedded resources. In those cases, the resource must appear prior to being referenced within the HTML markup.

Consider a simple scenario: an email composed of plain text and an attached PDF document. First, I create a `MimeMessage` instance tied to my `Session`, the configuration object for sending emails. Next, I will instantiate a `MimeMultipart` object to manage the different parts of the email. For the text component, a `MimeBodyPart` is created, its content is set as plain text, and it’s added to the `MimeMultipart`. Similarly, another `MimeBodyPart` is used for the PDF attachment. I will populate the PDF `MimeBodyPart` with a `DataSource`, which can come from a file path, or an array of bytes. The content type for the PDF is set to `application/pdf`, and its disposition is set to `attachment`, which dictates the behavior of the email client to show it as an downloadable attachment rather than in the body. Finally, the `MimeMultipart` is set as the content of the `MimeMessage`. Here is the java code that implements the aforementioned steps:

```java
import javax.mail.*;
import javax.mail.internet.*;
import javax.activation.*;
import java.io.File;
import java.util.Properties;

public class SimpleMultipartEmail {
    public static void sendMultipartEmail(String to, String from, String subject, File pdfAttachment) throws MessagingException {
        Properties props = System.getProperties();
        props.put("mail.smtp.host", "your_smtp_host"); // replace with real smtp host
        props.put("mail.smtp.port", "587"); // replace with real port
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.starttls.enable", "true");

        Session session = Session.getInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication("your_email", "your_password"); // replace credentials
            }
        });

        Message message = new MimeMessage(session);
        message.setFrom(new InternetAddress(from));
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setSubject(subject);


        Multipart multipart = new MimeMultipart();

        // Text part
        MimeBodyPart textPart = new MimeBodyPart();
        textPart.setText("This is the main body of the email. Please see the attachment for details.");
        multipart.addBodyPart(textPart);


        // PDF Attachment part
        MimeBodyPart pdfPart = new MimeBodyPart();
        DataSource pdfSource = new FileDataSource(pdfAttachment);
        pdfPart.setDataHandler(new DataHandler(pdfSource));
        pdfPart.setFileName(pdfAttachment.getName());
        pdfPart.setHeader("Content-Transfer-Encoding", "base64");
        pdfPart.setDisposition(Part.ATTACHMENT);
        multipart.addBodyPart(pdfPart);



        message.setContent(multipart);

        Transport.send(message);
        System.out.println("Multipart email sent successfully.");

    }
    public static void main(String[] args) {
        try{
           File file = new File("path/to/your/sample.pdf"); //replace with real path
           sendMultipartEmail("recipient@example.com", "sender@example.com", "Multipart Email With Attachment", file);

        }catch (MessagingException e){
            e.printStackTrace();
        }
    }
}

```

In this first example, a simple email with a plain text body and a single PDF attachment was constructed. The `Authenticator` class is used to handle authentication for the SMTP server. A `FileDataSource` is then used to read the PDF file and attach it to the message. Important to note that in this example, both the text and the PDF content parts are independent; one is not directly reliant on the other. The `Content-Transfer-Encoding` is set to `base64` for the PDF attachment to ensure the mail client decodes it correctly. The `setDisposition` to `Part.ATTACHMENT` forces the client to render the attachment as an attachment instead of inlined.

Now, let's consider a more intricate situation: an HTML email that embeds an image. Here the `MimeMultipart` container will include two distinct `MimeBodyPart` instances, one for the HTML content and another for the embedded image. The content type for the HTML part is `text/html`, and the `<img>` tag within the HTML references the image using a `cid:` URL scheme. The `cid:` scheme refers to a Content-ID, which is generated for the image part. The `Content-ID` is used as the value of the `src` attribute in the `<img>` tag. Additionally, the image part needs to have its `Content-ID` set so that the email client knows what content to render in the referenced URL. Here is the java code implementing this:

```java
import javax.mail.*;
import javax.mail.internet.*;
import javax.activation.*;
import java.io.File;
import java.util.Properties;
import java.util.UUID;


public class HtmlMultipartEmail {
    public static void sendHtmlMultipartEmail(String to, String from, String subject, File imageFile) throws MessagingException {
         Properties props = System.getProperties();
        props.put("mail.smtp.host", "your_smtp_host"); // replace with real smtp host
        props.put("mail.smtp.port", "587"); // replace with real port
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.starttls.enable", "true");

        Session session = Session.getInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication("your_email", "your_password"); // replace credentials
            }
        });
        Message message = new MimeMessage(session);
        message.setFrom(new InternetAddress(from));
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setSubject(subject);

        Multipart multipart = new MimeMultipart("related");

       // Generate a Content-ID for the image
        String cid = UUID.randomUUID().toString();

         //HTML Part
        MimeBodyPart htmlPart = new MimeBodyPart();
        String htmlContent = "<p>This is an HTML email with an embedded image.</p><img src='cid:" + cid +"'>";
        htmlPart.setContent(htmlContent, "text/html");
        multipart.addBodyPart(htmlPart);


        //Embedded Image part
        MimeBodyPart imagePart = new MimeBodyPart();
        DataSource imageSource = new FileDataSource(imageFile);
        imagePart.setDataHandler(new DataHandler(imageSource));
        imagePart.setContentID("<" + cid + ">");
        imagePart.setDisposition(Part.INLINE); // This makes the image inline
        multipart.addBodyPart(imagePart);

       message.setContent(multipart);

       Transport.send(message);
       System.out.println("Multipart HTML email sent successfully.");

    }

    public static void main(String[] args) {
        try{
           File imageFile = new File("path/to/your/sample.jpg"); //replace with real path
           sendHtmlMultipartEmail("recipient@example.com", "sender@example.com", "HTML Email With Embedded Image", imageFile);

        }catch(MessagingException e){
            e.printStackTrace();
        }
    }
}
```
In the above code, we set the content type to “related” which is required when you embed images in an html body. The `setContentID` method is used to set the `Content-ID` of the image. The `setDisposition` method, when set to `Part.INLINE`, forces the image to render inline instead of being displayed as a separate attachment. It’s crucial to note that the image part *must* be added to the multipart content *after* being referenced in the HTML markup by the `cid` scheme. The ordering is vital for email clients to correctly display the embedded content.

Lastly, it's important to handle alternative content, specifically offering a plain text alternative for HTML emails. This is crucial for accessibility and for email clients that do not render HTML correctly. This requires setting the multipart type to "alternative" and structuring the parts accordingly. The email will contain a `MimeMultipart` which contains two nested `MimeBodyPart` objects. The first `MimeBodyPart` will have the plain text version and the second `MimeBodyPart` will contain an inner `MimeMultipart` with the HTML version and any additional embedded resources.

```java
import javax.mail.*;
import javax.mail.internet.*;
import java.util.Properties;

public class AlternativeMultipartEmail {
    public static void sendAlternativeMultipartEmail(String to, String from, String subject, String htmlContent, String textContent) throws MessagingException {
        Properties props = System.getProperties();
        props.put("mail.smtp.host", "your_smtp_host"); // replace with real smtp host
        props.put("mail.smtp.port", "587"); // replace with real port
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.starttls.enable", "true");


        Session session = Session.getInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication("your_email", "your_password"); // replace credentials
            }
        });

        Message message = new MimeMessage(session);
        message.setFrom(new InternetAddress(from));
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setSubject(subject);

        Multipart multipartAlternative = new MimeMultipart("alternative");

        //Plain Text Version
        MimeBodyPart textPart = new MimeBodyPart();
        textPart.setContent(textContent, "text/plain");
        multipartAlternative.addBodyPart(textPart);


       // HTML version. It can contain embedded resources in real-world cases
        MimeBodyPart htmlPart = new MimeBodyPart();
        htmlPart.setContent(htmlContent, "text/html");
        multipartAlternative.addBodyPart(htmlPart);

        message.setContent(multipartAlternative);

        Transport.send(message);
        System.out.println("Multipart alternative email sent successfully.");


    }

    public static void main(String[] args) {
        try {
            String htmlContent = "<p>This is a sample <b>HTML</b> email with an alternative text version.</p>";
            String textContent = "This is a sample TEXT email with an alternative HTML version.";
            sendAlternativeMultipartEmail("recipient@example.com", "sender@example.com", "Email with Text and HTML alternatives", htmlContent, textContent);
        } catch(MessagingException e){
            e.printStackTrace();
        }
    }
}
```

Here the outermost `MimeMultipart` is set to `alternative`.  The plain text part,  added to the `alternative` multipart, is constructed first, followed by the HTML alternative.  Email clients that support HTML will render the HTML part, those that do not support or have HTML rendering disabled, will render the plaintext. This structure ensures the broadest possible compatibility.

To deepen your understanding, I recommend reviewing resources on MIME types, and SMTP protocols, along with documentation for the JavaMail API. In addition, researching common issues surrounding email rendering across different clients and debugging techniques for mail delivery problems will benefit your development efforts. Books on Internet mail standards and detailed guides on JavaMail API functionality will provide a robust foundation. Understanding the theory behind email protocols and standards can help one debug any issue and implement more advanced features of JavaMail.
