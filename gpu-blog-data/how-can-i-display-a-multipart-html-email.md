---
title: "How can I display a multipart HTML email in a JEditorPane?"
date: "2025-01-30"
id: "how-can-i-display-a-multipart-html-email"
---
Displaying a multipart HTML email, specifically one containing both HTML and plain text alternatives, within a `JEditorPane` requires careful handling of the MIME types and structure inherent in email messages. A `JEditorPane` by default only understands a singular HTML representation, thus necessitating a mechanism to extract the appropriate HTML content from a multipart email for correct rendering.

In my experience working on legacy email clients, I’ve frequently encountered this challenge.  Email messages structured according to MIME (Multipurpose Internet Mail Extensions) specifications, often use the `multipart/alternative` content type to offer both rich HTML renditions alongside a fallback plain text version. The `JEditorPane`, however, is not inherently designed to process the intricate structure of multipart MIME messages; instead, it expects a single source of HTML content. Therefore, the responsibility falls upon the developer to interpret and select the proper HTML part before presenting it to the component.

The standard Java Mail API, specifically javax.mail, offers tools to parse these multipart messages. The core challenge revolves around navigating the `Multipart` object and identifying the desired `BodyPart`, typically the one with the `text/html` MIME type. Once extracted, this body part's content can then be set into the `JEditorPane`. Critically, not all emails contain a `text/html` part. In those scenarios, it’s prudent to fall back to the `text/plain` alternative or display an appropriate message indicating HTML rendering is unavailable.

Let’s consider the scenario where the `javax.mail.Message` object `emailMessage` is already available, representing the parsed email. The initial step involves checking if the email content is indeed a multipart message.  The `getContent()` method of a `Message` returns an `Object` which, for multipart emails, will be an instance of `Multipart`.  We can then iterate through the individual `BodyPart` objects contained within.

**Example 1: Basic Extraction of HTML Part**

```java
import javax.mail.*;
import javax.mail.internet.*;
import java.io.IOException;
import javax.swing.JEditorPane;
import javax.swing.text.html.HTMLEditorKit;

public class EmailViewer {

    public static void displayEmail(Message emailMessage, JEditorPane editorPane) throws MessagingException, IOException {

         if (emailMessage.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) emailMessage.getContent();
            for (int i = 0; i < multipart.getCount(); i++) {
                BodyPart bodyPart = multipart.getBodyPart(i);
                if (bodyPart.isMimeType("text/html")) {
                    String htmlContent = (String) bodyPart.getContent();
                    editorPane.setText(htmlContent);
                    editorPane.setEditorKit(new HTMLEditorKit()); //Ensure HTML content is rendered
                    return;
                 }
             }
         }
        //If no HTML is found, try to display plain text
        if (emailMessage.isMimeType("text/plain")) {
          String textContent = (String) emailMessage.getContent();
           editorPane.setText(textContent);
          return;
        }

        // If neither HTML nor plaintext found display a message
        editorPane.setText("Unable to display email content.");
    }

    public static void main(String[] args) throws MessagingException {
         // Sample Email Message, Replace with actual parsed email message
         Message sampleMessage = createSampleEmail();

         JEditorPane editorPane = new JEditorPane();
         try {
            displayEmail(sampleMessage, editorPane);
         } catch (IOException e) {
           System.err.println("Error loading email content: " + e.getMessage());
         }

        javax.swing.JFrame frame = new javax.swing.JFrame("Email Viewer");
        frame.add(new javax.swing.JScrollPane(editorPane));
        frame.setSize(800,600);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
    }

     private static Message createSampleEmail() throws MessagingException{
          //Simulating a basic multipart email
           Session session = Session.getDefaultInstance(System.getProperties());
           Message message = new MimeMessage(session);

           Multipart multipart = new MimeMultipart("alternative");

           MimeBodyPart textPart = new MimeBodyPart();
           textPart.setContent("This is plain text alternative", "text/plain");

           MimeBodyPart htmlPart = new MimeBodyPart();
           htmlPart.setContent("<html><body><h1>Hello</h1><p>This is HTML content</p></body></html>", "text/html");

           multipart.addBodyPart(textPart);
           multipart.addBodyPart(htmlPart);
           message.setContent(multipart);

          return message;
     }
}
```

*Commentary:*  This code first checks if the email is multipart. If so, it iterates through each body part.  When a part with MIME type "text/html" is found, its content is extracted and set as the `JEditorPane`'s text. Setting an `HTMLEditorKit` is crucial for correct rendering. I've also added a basic fallback to `text/plain` if no HTML alternative was present. Finally, a default message is displayed if no suitable content is available.  The main method provides a rudimentary setup to display the email in a window with a sample message being created in the 'createSampleEmail' method for demonstration.

It is important to note that simply calling `setText()` with HTML content may not render it correctly unless the corresponding `EditorKit` is set. The `HTMLEditorKit` is specifically designed for this purpose. Additionally, error handling, specifically with `IOException` and `MessagingException`, should be present in real-world applications.

However, this example provides minimal handling for resources referenced in the HTML content, such as images embedded using CID (Content-ID) or referenced via URLs. For displaying images embedded using Content-ID a more complex handler is needed.

**Example 2: Handling Embedded Images (Simplified)**

```java
import javax.mail.*;
import javax.mail.internet.*;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.swing.JEditorPane;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.Image;
import javax.swing.ImageIcon;

public class EmailViewerWithImages {

  public static void displayEmail(Message emailMessage, JEditorPane editorPane) throws MessagingException, IOException {
        if (emailMessage.isMimeType("multipart/*")) {
             Multipart multipart = (Multipart) emailMessage.getContent();
             String htmlContent = extractHtmlWithImages(multipart, new HashMap<>());
             if (htmlContent != null) {
                  editorPane.setText(htmlContent);
                  editorPane.setEditorKit(new HTMLEditorKit());
                  return;
               }
          }
        if (emailMessage.isMimeType("text/plain")) {
          String textContent = (String) emailMessage.getContent();
          editorPane.setText(textContent);
          return;
        }
       editorPane.setText("Unable to display email content.");
   }

  private static String extractHtmlWithImages(Multipart multipart, Map<String, Image> imageCache) throws MessagingException, IOException {

        for (int i = 0; i < multipart.getCount(); i++) {
            BodyPart bodyPart = multipart.getBodyPart(i);

            if(bodyPart.isMimeType("text/html")) {
                 String htmlContent = (String) bodyPart.getContent();
                 return resolveImages(htmlContent, multipart, imageCache);
             }
        }
        return null;
   }


    private static String resolveImages(String htmlContent, Multipart multipart, Map<String, Image> imageCache) throws MessagingException, IOException {

            for (int i=0; i < multipart.getCount(); i++) {
               BodyPart bodyPart = multipart.getBodyPart(i);
                 String contentIdHeader = bodyPart.getHeader("Content-ID",null);
                if (contentIdHeader != null) {
                 String cid = contentIdHeader.replaceAll("[<>]",""); //remove < and >

                 if(bodyPart.isMimeType("image/*")){
                    if(!imageCache.containsKey(cid)){
                       Object content = bodyPart.getContent();
                        if (content instanceof javax.activation.DataHandler)
                        {
                          javax.activation.DataHandler dh = (javax.activation.DataHandler)content;
                           ImageIcon imageIcon = new ImageIcon(dh.getInputStream().readAllBytes());
                             imageCache.put(cid, imageIcon.getImage());
                        }
                       }
                    }
                }

            }
            String resolvedHtml = htmlContent;
             for (String cid : imageCache.keySet()){
               Image image = imageCache.get(cid);
               ImageIcon icon = new ImageIcon(image);
             if (image != null) {
               String imageTag = "<img src='data:image/png;base64," +java.util.Base64.getEncoder().encodeToString(convertImageToBytes(image)) + "'/>"; // Convert image to base64
              resolvedHtml = resolvedHtml.replaceAll("cid:" + cid, imageTag); //Replace CID with data URI
               }
            }
            return resolvedHtml;
    }


   private static byte[] convertImageToBytes(Image image) throws IOException{
      java.awt.image.BufferedImage bufferedImage = new java.awt.image.BufferedImage(image.getWidth(null), image.getHeight(null),java.awt.image.BufferedImage.TYPE_INT_RGB);
      java.awt.Graphics2D graphics = bufferedImage.createGraphics();
      graphics.drawImage(image,0,0,null);
      graphics.dispose();

      java.io.ByteArrayOutputStream outputStream = new java.io.ByteArrayOutputStream();
      javax.imageio.ImageIO.write(bufferedImage, "png", outputStream);
      return outputStream.toByteArray();
   }



  public static void main(String[] args) throws MessagingException {
     // Sample Email Message, Replace with actual parsed email message
      Message sampleMessage = createSampleEmail();

     JEditorPane editorPane = new JEditorPane();
     try {
          displayEmail(sampleMessage, editorPane);
      } catch (IOException e) {
         System.err.println("Error loading email content: " + e.getMessage());
      }

       javax.swing.JFrame frame = new javax.swing.JFrame("Email Viewer");
       frame.add(new javax.swing.JScrollPane(editorPane));
       frame.setSize(800,600);
       frame.setVisible(true);
       frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
     }


  private static Message createSampleEmail() throws MessagingException {
        Session session = Session.getDefaultInstance(System.getProperties());
        Message message = new MimeMessage(session);

       Multipart multipart = new MimeMultipart("related");

       MimeBodyPart textPart = new MimeBodyPart();
       textPart.setContent("<html><body><h1>Hello</h1><p>This is HTML content <img src=\"cid:image1\" /></p></body></html>", "text/html");

       MimeBodyPart imagePart = new MimeBodyPart();
       imagePart.attachFile(new java.io.File("src/image.png")); // Replace image.png with a valid png file in the directory
       imagePart.setHeader("Content-ID", "<image1>");

        multipart.addBodyPart(textPart);
        multipart.addBodyPart(imagePart);
        message.setContent(multipart);
        return message;
    }
}
```

*Commentary:* This expanded example now addresses embedded images.  The `extractHtmlWithImages` method iterates through the multipart structure, extracting HTML and caching images. The `resolveImages` method identifies images via their "Content-ID" headers and creates a base64 encoded image string. These data URIs are inserted directly into the HTML, replacing "cid:image_id".  The `convertImageToBytes` method converts an `Image` object to a byte array so that it can be encoded as base64. The main method has been modified to display this email with a sample "image.png" file located in a 'src' folder.  This approach provides a relatively straightforward method to handle embedded images, although in practice further refinements such as handling image scaling may be required.

**Example 3: Handling Multiple Parts & Content-Transfer-Encoding**

In rare scenarios, email parts might use Content-Transfer-Encoding schemes that require decoding before their content is usable. Here’s a revised snippet demonstrating decoding and handling multiple parts (including plain text and html).

```java
import javax.mail.*;
import javax.mail.internet.*;
import javax.mail.util.ByteArrayDataSource;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import javax.swing.JEditorPane;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.Image;
import javax.swing.ImageIcon;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

public class EmailViewerEncoding {

     public static void displayEmail(Message emailMessage, JEditorPane editorPane) throws MessagingException, IOException {
      if (emailMessage.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) emailMessage.getContent();
            String htmlContent = extractHtmlWithImages(multipart, new HashMap<>());
            if(htmlContent != null){
                editorPane.setText(htmlContent);
                editorPane.setEditorKit(new HTMLEditorKit());
                return;
            }
        }
        if(emailMessage.isMimeType("text/plain")){
              String textContent =  decodeContent(emailMessage.getContent(), emailMessage.getHeader("Content-Transfer-Encoding", null));
                editorPane.setText(textContent);
            return;
          }
       editorPane.setText("Unable to display email content.");
    }


    private static String extractHtmlWithImages(Multipart multipart, Map<String, Image> imageCache) throws MessagingException, IOException {
       for (int i = 0; i < multipart.getCount(); i++) {
            BodyPart bodyPart = multipart.getBodyPart(i);
            if(bodyPart.isMimeType("text/html")) {
                String htmlContent = decodeContent(bodyPart.getContent(), bodyPart.getHeader("Content-Transfer-Encoding", null));
                return resolveImages(htmlContent, multipart, imageCache);
           }
        }
      return null;
  }



  private static String resolveImages(String htmlContent, Multipart multipart, Map<String, Image> imageCache) throws MessagingException, IOException {
        for (int i=0; i < multipart.getCount(); i++) {
          BodyPart bodyPart = multipart.getBodyPart(i);
            String contentIdHeader = bodyPart.getHeader("Content-ID",null);
            if (contentIdHeader != null) {
              String cid = contentIdHeader.replaceAll("[<>]","");

             if(bodyPart.isMimeType("image/*")){
              if(!imageCache.containsKey(cid)){
                  Object content = bodyPart.getContent();
                   if (content instanceof javax.activation.DataHandler)
                      {
                          javax.activation.DataHandler dh = (javax.activation.DataHandler)content;
                            ImageIcon imageIcon = new ImageIcon(dh.getInputStream().readAllBytes());
                           imageCache.put(cid, imageIcon.getImage());
                      }
                }
            }
         }
      }
      String resolvedHtml = htmlContent;
       for (String cid : imageCache.keySet()){
           Image image = imageCache.get(cid);
           ImageIcon icon = new ImageIcon(image);

          if (image != null) {
           String imageTag = "<img src='data:image/png;base64," +java.util.Base64.getEncoder().encodeToString(convertImageToBytes(image)) + "'/>";
             resolvedHtml = resolvedHtml.replaceAll("cid:" + cid, imageTag);
          }
        }
        return resolvedHtml;
    }


    private static byte[] convertImageToBytes(Image image) throws IOException{
         java.awt.image.BufferedImage bufferedImage = new java.awt.image.BufferedImage(image.getWidth(null), image.getHeight(null),java.awt.image.BufferedImage.TYPE_INT_RGB);
         java.awt.Graphics2D graphics = bufferedImage.createGraphics();
         graphics.drawImage(image,0,0,null);
         graphics.dispose();

        java.io.ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
         javax.imageio.ImageIO.write(bufferedImage, "png", outputStream);
        return outputStream.toByteArray();
    }


    private static String decodeContent(Object content, String[] transferEncoding) throws MessagingException, IOException {
       if (content instanceof String) {
          return (String) content; //if string, no decoding needed
      }

        if(transferEncoding != null && transferEncoding.length > 0){
              String encoding = transferEncoding[0]; //consider only the first encoding type
              if(encoding != null){
                     encoding = encoding.toLowerCase();
                    if (encoding.equalsIgnoreCase("quoted-printable"))
                     {
                       javax.mail.internet.MimeUtility.decodeText(content.toString());

                         ByteArrayOutputStream baos = new ByteArrayOutputStream();
                        if (content instanceof InputStream) {
                           ((InputStream) content).transferTo(baos);
                        } else if (content instanceof byte[]) {
                           baos.write((byte[])content);
                        } else {
                           return null;
                        }
                       return javax.mail.internet.MimeUtility.decodeText(baos.toString(StandardCharsets.UTF_8));
                      }
                  if(encoding.equalsIgnoreCase("base64")){
                      ByteArrayOutputStream baos = new ByteArrayOutputStream();
                      if (content instanceof InputStream) {
                          ((InputStream) content).transferTo(baos);
                      } else if (content instanceof byte[]) {
                            baos.write((byte[])content);
                        } else {
                          return null;
                        }

                       return new String(java.util.Base64.getDecoder().decode(baos.toByteArray()), StandardCharsets.UTF_8);

                   }
                }
        }
        if (content instanceof InputStream){
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ((InputStream)content).transferTo(baos);
            return new String(baos.toByteArray(), StandardCharsets.UTF_8);
        }

       if (content instanceof byte[])
        {
              return new String((byte[]) content, StandardCharsets.UTF_8);
       }

      return String.valueOf(content);
    }


      public static void main(String[] args) throws MessagingException {
      // Sample Email Message, Replace with actual parsed email message
          Message sampleMessage = createSampleEmail();

         JEditorPane editorPane = new JEditorPane();
        try {
            displayEmail(sampleMessage, editorPane);
         } catch (IOException e) {
            System.err.println("Error loading email content: " + e.getMessage());
         }

         javax.swing.JFrame frame = new javax.swing.JFrame("Email Viewer");
         frame.add(new javax.swing.JScrollPane(editorPane));
          frame.setSize(800,600);
          frame.setVisible(true);
         frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
    }


    private static Message createSampleEmail() throws MessagingException {
      Session session = Session.getDefaultInstance(System.getProperties());
       Message message = new MimeMessage(session);

        Multipart multipart = new MimeMultipart("mixed");

      MimeBodyPart textPart = new MimeBodyPart();
        textPart.setContent("This is a text alternative.", "text/plain");


        MimeBodyPart htmlPart = new MimeBodyPart();
        htmlPart.setContent("<html><body><h1>Hello</h1><p>This is HTML content <img src=\"cid:image1\" /></p></body></html>", "text/html");
        htmlPart.setHeader("Content-Transfer-Encoding", "quoted-printable");

         MimeBodyPart imagePart = new MimeBodyPart();
         imagePart.attachFile(new java.io.File("src/image.png"));
         imagePart.setHeader("Content-ID", "<image1>");


       multipart.addBodyPart(textPart);
       multipart.addBodyPart(htmlPart);
       multipart.addBodyPart(imagePart);
       message.setContent(multipart);

      return message;
    }
}
```
*Commentary:* In this version, I've introduced the `decodeContent` method which handles the "quoted-printable" and "base64" content encoding. The previous `extractHtmlWithImages` is also updated to use this method to decode the HTML content. This shows how emails with different encoding types can be handled. The sample email has been updated to encode the HTML content with quoted-printable to demonstrate its use.

For comprehensive understanding and practical application of mail parsing, I suggest consulting the official JavaMail API documentation.  Furthermore, exploring resources like the "Java Mail Developer's Guide" by John Zukowski can be beneficial. Studying RFC 822 (Standard for ARPA Internet Text Messages) and related MIME specifications is also invaluable for understanding the underlying structure of email messages. Finally, examining open source email client implementations can provide additional practical insight.
