---
title: "Why is there a Wrong character in mail depending on the sending computer?"
date: "2024-12-15"
id: "why-is-there-a-wrong-character-in-mail-depending-on-the-sending-computer"
---

Alright, let's break this down. The issue of seeing garbled characters in emails, especially when it seems to depend on *which* computer sent it, is a classic and often frustrating problem. It usually boils down to character encoding, and i've had my fair share of late nights chasing these bugs. 

Here's the deal, under the hood, computers represent characters as numbers. simple enough. the letter 'a' is not actually an 'a' stored in a computer's memory but a numerical code like 97 in the ascii standard. The trouble starts when we need to represent characters beyond the simple English alphabet; symbols like é, or ö, or entire alphabets like Cyrillic or Japanese which need their own codes.

The *encoding* is the agreed upon system for mapping these characters to their numerical representation. The original and really simple character encoding is ASCII, which covers only basic english characters, digits, and some punctuation. That's it. But what happens if you need to email someone who speaks spanish and wants to send a message with an 'ñ'? this is where things start getting hairy because ascii doesn't support it.

Then came encodings like iso-8859-1 (latin-1) which expanded the character set, covering western european languages. that was good for those guys, but a mess for the rest. then we had all sorts of other iso-8859-* variants, all adding a particular set of characters, and, of course they are not compatible with each other. The problem was that one computer can write a message using iso-8859-1, and another computer expects utf-8 to decode it, and you will see gobbledygook, that is the issue.

The real solution came with Unicode and its most popular encoding, UTF-8. UTF-8 is designed to represent pretty much any character in any language, so it's very good for global communications like the internet. It is also backwards compatible with ascii, meaning a file that is encoded with ascii can be read with the utf-8 encoding and the ascii characters will be interpreted as usual. The tricky part is when systems disagree on which encoding was used to compose the message and there are a number of ways that can happen.

**Mismatched Encoding Declarations**

The first and most common problem is that when an email is created, it often includes metadata that tells the email client or server *what* encoding was used to write the content. When this declaration is wrong or not there or different on both side, that's where you get the weirdness. For instance, a system might use utf-8 to write the message, but it incorrectly declares iso-8859-1 or does not declare any encoding at all in the metadata. When a receiving email system tries to display it, the decoder tries to make sense of the characters as if they were iso-8859-1, hence rendering the whole thing as a mess of nonsensical characters, like the '¿' instead of the intended 'ü' or some hieroglyphic character from an unknown font. I went through hell when i was building a chat application for a multi-lingual university system and not all the emails were interpreted correctly until i realised it was a problem with encoding.

**Email Client Issues**

Sometimes, it’s not even the declared encoding that’s wrong. Some older email clients or badly configured ones might not handle different encodings properly. they might try to "guess" the encoding and, if they guess wrong, you get the junk characters. This was a particularly bad issue in the early 2000s. i had a desktop that was not capable of handling unicode and I would get those characters all the time.

**Encoding Conversion**

In the transit of the email, there are many computers in between from the sender's computer to the receiver. In this route, the encoding may change between various systems due to automatic conversion or other system configurations which can introduce errors. Some email servers try to 'fix' encoding issues, but if they get the 'fix' wrong, you end up with the same mess.

**Character Encoding in Code**

If you are building an application that deals with sending or receiving emails, you absolutely need to make sure you handle character encoding properly, and you can do it in different languages with different libraries. here is an example in Python.

```python
import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_email(sender_email, sender_password, receiver_email, subject, message, encoding='utf-8'):
  """Sends an email with the specified encoding."""

  msg = MIMEText(message, 'plain', encoding)
  msg['Subject'] = Header(subject, encoding)
  msg['From'] = sender_email
  msg['To'] = receiver_email

  try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server: #Use your smtp server, not necessarily gmail.
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print("email sent successfully")
  except Exception as e:
    print(f"An error occurred: {e}")

# example usage, send a simple email
sender_email = 'your_email@gmail.com'
sender_password = 'your_password'
receiver_email = 'receiver_email@gmail.com'
subject = 'Test with unicode chars'
message = 'This is a test message with some üöäáéíóúñ chars.'

send_email(sender_email, sender_password, receiver_email, subject, message)
```

This snippet demonstrates creating a basic email message with MIMEText, explicitly specifying `utf-8` as the encoding for both the content and the subject header. this avoids a whole array of issues.

Here is another example, this time in JavaScript, that takes a text message and converts it to base64, so you can pass it as an email attachment or similar.

```javascript
function encodeMessage(message, encoding = 'utf-8') {
  try {
    const encoder = new TextEncoder(encoding);
    const encodedMessage = encoder.encode(message);
    const base64Message = btoa(String.fromCharCode(...encodedMessage));
    return base64Message;
  } catch (error) {
    console.error('encoding error:', error);
    return null;
  }
}

function decodeMessage(base64Message, encoding = 'utf-8') {
  try {
    const decodedString = atob(base64Message);
    const byteArray = new Uint8Array(decodedString.length);
    for (let i = 0; i < decodedString.length; i++) {
        byteArray[i] = decodedString.charCodeAt(i);
    }
    const decoder = new TextDecoder(encoding);
    const decodedMessage = decoder.decode(byteArray);
    return decodedMessage;
  } catch (error) {
    console.error('Decoding error:', error);
    return null;
  }
}

const originalMessage = 'Hello, this is a message with special chars üöäáéíóúñ';
const encoded = encodeMessage(originalMessage);
console.log('Encoded Message:', encoded);

const decoded = decodeMessage(encoded);
console.log('Decoded Message:', decoded);
```

This javascript code snippet shows how to encode a string to base64 using a particular encoding. You may need to use such functionality depending on your email transport requirements. The functions provided also show how to decode the base64 string with the same encoding.

And here, lets see an example of encoding a string in java:

```java
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public class EncodingUtil {

    public static String encodeString(String message, String encoding) {
        try {
            byte[] encodedBytes = message.getBytes(encoding);
            return Base64.getEncoder().encodeToString(encodedBytes);
        } catch (Exception e) {
            System.err.println("Encoding error: " + e.getMessage());
            return null;
        }
    }

    public static String decodeString(String base64Message, String encoding) {
      try {
          byte[] decodedBytes = Base64.getDecoder().decode(base64Message);
          return new String(decodedBytes, encoding);
      } catch (Exception e) {
          System.err.println("Decoding error: " + e.getMessage());
          return null;
      }
  }

    public static void main(String[] args) {
        String message = "Hello, this is a string with some special characters like üöäáéíóúñ";
        String encodedMessage = encodeString(message, "UTF-8");
        System.out.println("Encoded Message: " + encodedMessage);

        String decodedMessage = decodeString(encodedMessage, "UTF-8");
        System.out.println("Decoded Message: " + decodedMessage);

    }
}
```
This java snippet shows how to encode and decode a string with base64 and a specific encoding. This particular java snippet shows the basic operations involved when using base64 encoding and decoding using the java standard library.

**General recommendations**

When handling emails, you should always explicitly set the encoding, ideally to `utf-8`, both in email headers and the body of the message. You should also try to use a modern email client and libraries that support the newest standard encodings. If you're dealing with user-generated content from multiple sources (like a chat app), you will need to make sure to be very explicit and consistent in the encoding that you are using. Be careful with auto-conversion tools, as they may introduce unexpected issues.

For a deeper dive into character encoding, i'd highly recommend “Unicode Explained” by Jukka K. Korpela, it's a fantastic book, and you can also refer to the unicode consortium website at unicode.org, which contains all the information about the unicode standards.

Oh and by the way, i once got a character encoding error so bad it tried to display my cat as a bunch of greek symbols. that was fun.

In essence, character encoding might look like a low level thing to worry about, but getting it wrong can make your application unusable, it can create an avalanche of issues in what may look like a very small problem. By understanding how encodings work and being very explicit and rigorous in their usage, you'll save yourself tons of debugging time.
