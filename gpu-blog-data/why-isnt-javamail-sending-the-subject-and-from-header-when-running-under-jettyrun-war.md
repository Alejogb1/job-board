---
title: "Why isn't JavaMail sending the subject and from header when running under jetty:run-war?"
date: "2025-01-26"
id: "why-isnt-javamail-sending-the-subject-and-from-header-when-running-under-jettyrun-war"
---

JavaMail, when functioning within a `jetty:run-war` context, can exhibit peculiar behavior regarding email headers, specifically the "Subject" and "From" fields. This is often not a failure of JavaMail itself, but rather an interaction with the configuration, the embedded Jetty environment, and the way the application obtains resources within it. The core problem stems from how the web application is initialized and how the JavaMail session is defined.

Here's a detailed breakdown based on my troubleshooting history, specifically a problematic project involving a complex transactional email system: When a web application operates in a traditional application server environment like Tomcat or Glassfish, resources (like a JavaMail `Session`) are often managed by the container, either through JNDI lookups or via server-level configuration files. However, when using `jetty:run-war`, which effectively runs an embedded instance of Jetty, the container does not always perform the same resource initialization process. This can lead to inconsistencies in how the `Session` is created, and crucially, how its properties are populated. The default initialization routines in most JavaMail applications, including those with `javax.mail.internet.MimeMessage` class, assumes that these required header properties like the Subject and from address are included in the properties of a mail `Session`. This assumption is often true when the mail session object is acquired from a JNDI lookup, since the container’s configuration includes these. However, when manually creating the `Session` using a default constructor with an incomplete properties object, these critical headers are left to be specified in code before sending the message.

Typically, the JavaMail API relies on a `java.util.Properties` object to configure the mail session. This properties object contains key-value pairs that define the mail server's address, authentication details, and other relevant settings. When running within a web application container, these properties are frequently obtained from a JNDI context (a naming directory). The mail session is retrieved from the JNDI directory, and it encapsulates all the required headers. When using `jetty:run-war`, the JNDI context is rarely configured or even utilized in the same way, leading to a properties object without the essential properties.

The symptom, observed through debugging, is that the `MimeMessage` being constructed lacks the required header information during email construction, and the email is sent without them by the JavaMail API. The email server receives a message that either discards these messages or inserts default headers. The issue is typically not a bug in JavaMail itself but rather a consequence of an incomplete session properties configuration in an environment where no default JNDI is available to provide the complete properties. The code might look correct to an inexperienced developer, because in their testing environments, the mail session is setup correctly by the container.

Now, let’s examine some code examples to illustrate these points:

**Example 1: Incorrect Session Setup**

This example demonstrates the common mistake of creating a JavaMail session with an incomplete properties set:

```java
import java.util.Properties;
import javax.mail.*;
import javax.mail.internet.*;

public class EmailService {

    public void sendEmail(String to, String body) throws MessagingException {
        Properties props = new Properties();
        props.put("mail.smtp.host", "smtp.example.com"); // Only host is specified
        props.put("mail.smtp.port", "587");
		props.put("mail.smtp.auth", "true");
		props.put("mail.smtp.starttls.enable", "true");
		props.put("mail.smtp.user", "username");
		props.put("mail.smtp.password", "password");

        Session session = Session.getInstance(props, new Authenticator() {
        	@Override
        	protected PasswordAuthentication getPasswordAuthentication() {
        		return new PasswordAuthentication("username", "password");
        	}
        });

        Message message = new MimeMessage(session);
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        // Subject and from are NOT set
        message.setText(body);
        Transport.send(message);
    }
}
```

Here, while SMTP connection details are provided, the `Session` does not implicitly understand the "From" address or subject that the server needs, nor that a particular account should be used.  This results in an email being sent without these headers when executed through `jetty:run-war` because there's no JNDI context or externally configured settings, relying purely on the limited properties provided.

**Example 2: Corrected Session Setup with manual header configuration**

This example shows the solution, configuring the "From" address and subject directly in the message object itself and not relying on a session properties configuration:

```java
import java.util.Properties;
import javax.mail.*;
import javax.mail.internet.*;

public class EmailService {

    public void sendEmail(String to, String subject, String body) throws MessagingException {
        Properties props = new Properties();
        props.put("mail.smtp.host", "smtp.example.com");
        props.put("mail.smtp.port", "587");
		props.put("mail.smtp.auth", "true");
		props.put("mail.smtp.starttls.enable", "true");
		props.put("mail.smtp.user", "username");
		props.put("mail.smtp.password", "password");

        Session session = Session.getInstance(props,  new Authenticator() {
        	@Override
        	protected PasswordAuthentication getPasswordAuthentication() {
        		return new PasswordAuthentication("username", "password");
        	}
        });


        Message message = new MimeMessage(session);
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setFrom(new InternetAddress("myemail@example.com"));  // Explicitly set From
		message.setSubject(subject); // Explicitly set subject
        message.setText(body);
        Transport.send(message);
    }
}
```

In this corrected version, the `setFrom` and `setSubject` methods of the `MimeMessage` are used to explicitly define the missing information. This approach works regardless of the deployment environment, as it is independent of container-specific configurations. The `Session` is still created in the same manner as in the original example with the required connection properties specified in the properties object.

**Example 3: Correct Session Setup using a properties configuration**

This final example illustrates a method of setting up the session object by including the required headers as properties:

```java
import java.util.Properties;
import javax.mail.*;
import javax.mail.internet.*;

public class EmailService {

    public void sendEmail(String to, String body) throws MessagingException {
		String fromAddress = "myemail@example.com";
		String mailSubject = "My Test Subject";
        Properties props = new Properties();
        props.put("mail.smtp.host", "smtp.example.com");
        props.put("mail.smtp.port", "587");
		props.put("mail.smtp.auth", "true");
		props.put("mail.smtp.starttls.enable", "true");
		props.put("mail.smtp.user", "username");
		props.put("mail.smtp.password", "password");
		props.put("mail.from", fromAddress);
		props.put("mail.subject", mailSubject);

        Session session = Session.getInstance(props, new Authenticator() {
        	@Override
        	protected PasswordAuthentication getPasswordAuthentication() {
        		return new PasswordAuthentication("username", "password");
        	}
        });

        Message message = new MimeMessage(session);
        message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
        message.setText(body);
        Transport.send(message);
    }
}
```

Here, the `mail.from` and `mail.subject` properties are set directly within the properties object that creates the `Session`.  This method works as well when creating the session object directly, and avoids the repeated configuration of headers before sending every message. This is preferable for some situations.

In summary, when running JavaMail with `jetty:run-war`, it's crucial to be mindful of how the `Session` is configured, as the usual JNDI lookup might not be available. Ensuring that all required properties, including the "From" address and subject, are either provided in session configuration properties or explicitly set within the `MimeMessage` is essential for proper email delivery. This can be accomplished as shown in Example 2 or 3.

For additional study and reference, consider exploring JavaMail API documentation, specifically the classes `javax.mail.Session`, `javax.mail.internet.MimeMessage`, and `java.util.Properties`. Also, consult resources that detail best practices when using JavaMail in different server environments. Furthermore, detailed books on Java web application servers can provide additional knowledge regarding configuration and deployment options. Investigating specific documentation from the mail provider can clarify any specific required settings for outgoing messages as well, such as specific ports or authentication methods.
