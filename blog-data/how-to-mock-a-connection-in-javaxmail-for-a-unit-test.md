---
title: "How to mock a connection in javax.mail for a Unit test?"
date: "2024-12-14"
id: "how-to-mock-a-connection-in-javaxmail-for-a-unit-test"
---

alright, so you're looking to mock a connection in javax.mail for unit testing, yeah? i've been there, done that, got the t-shirt, and probably spilled coffee on it too. it's a pretty common hurdle when you're trying to isolate your email-sending logic and avoid the headache of actually sending emails during tests.

first off, let's be clear, javax.mail isn't exactly built for easy mocking out of the box. it uses concrete classes and final methods in some places which makes it tricky. you can't just extend `javax.mail.Transport` and override methods, or something like that because they are final methods, and that's a real problem if you want to control the behavior of the email send process from a unit test. this limitation is why we need to get creative, basically we have to bypass it using different techniques.

i remember one particularly annoying project years ago. i was working on an application that sent out notification emails for a big e-commerce platform. the email sending logic was all intertwined with the rest of the system, and every time i ran my tests, it was also triggering email sends. talk about a spam machine. plus, each test would take forever because of the i/o and networking involved, not ideal for fast feedback in development. at one point, i even accidentally spammed my co-worker, they thought we were under attack by some evil email bot. i learned my lesson after that day. unit tests should be fast, repeatable, and not trigger external side effects, like spamming my coworker.

so, after that experience, i spent some time investigating how to get around this javax.mail mocking problem. what we need to do is control and isolate external dependencies, and then fake those in tests, so lets go over some of the most common approaches.

one of the most straightforward methods is to use a mocking library, like mockito. this is my usual first go-to because i am comfortable with mockito and it is a standard de-facto option in the java world. the key idea is to mock the `javax.mail.Session` and the `javax.mail.Transport` objects and make them act as if the mail was sent successfully, without making the actual smtp call. here is some sample code:

```java
import org.junit.jupiter.api.Test;
import static org.mockito.Mockito.*;
import javax.mail.*;
import javax.mail.internet.MimeMessage;

public class EmailServiceTest {

    @Test
    public void testEmailSending() throws Exception {
        // mock session
        Session session = mock(Session.class);

        // mock transport
        Transport transport = mock(Transport.class);
        when(session.getTransport(anyString())).thenReturn(transport);

        // create and inject mail sender
        EmailService emailService = new EmailService(session);

        // create a message
        Message message = new MimeMessage(session);
        message.setRecipient(Message.RecipientType.TO, new javax.mail.internet.InternetAddress("test@example.com"));
        message.setSubject("test subject");
        message.setText("test message");

        // send message using mocked transport
        emailService.sendEmail(message);

        // verify that the transport.connect and transport.sendMessage were invoked
        verify(transport, times(1)).connect(anyString(), anyInt(), anyString(), anyString());
        verify(transport, times(1)).sendMessage(any(), any());
        verify(transport, times(1)).close();
    }

    // your email service
    public class EmailService {
         private final Session session;

         public EmailService(Session session){
           this.session = session;
         }

        public void sendEmail(Message message) throws MessagingException {
            Transport transport = session.getTransport("smtp");
            transport.connect("localhost", 25, "user", "password");
            transport.sendMessage(message, message.getAllRecipients());
            transport.close();
        }
    }
}

```

in this code snippet, we're using mockito to create a mock `session` object and a mock `transport` object. we configure the mock `session` to return our mocked `transport` when `session.getTransport("smtp")` is called and then verify that `connect`, `sendMessage`, and `close` where called using mockito's `verify` method, but in reality, no connection is done at all. this is the trick here, mockito records and checks that the method calls are being made.

another method, which is sometimes used in legacy codebases, or when mocking libraries aren't an option for some reason, involves creating a custom transport that just stores the sent message. this approach requires creating a class that extends javax.mail.Transport, which might not be ideal for new projects and some people dont like it due to the nature of javax.mail classes.

```java
import javax.mail.*;
import javax.mail.internet.MimeMessage;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MockTransportTest {

    @Test
    public void testCustomMockTransport() throws Exception {
         Properties props = new Properties();
         props.put("mail.transport.protocol", "custommock");

         //create custom session
         Session session = Session.getInstance(props);

         //set the mock transport
         session.setProvider(new Provider(Provider.Type.TRANSPORT,"custommock","testmock", MockTransport.class.getName(),null));

         EmailService emailService = new EmailService(session);
         Message message = new MimeMessage(session);
         message.setRecipient(Message.RecipientType.TO, new javax.mail.internet.InternetAddress("test@example.com"));
         message.setSubject("test subject");
         message.setText("test message");

         // send email using our custom transport
         emailService.sendEmail(message);

         //check that the email was intercepted by custom mock
         List<Message> sentMessages = MockTransport.getSentMessages();
         assertEquals(1, sentMessages.size());
         assertEquals("test subject", sentMessages.get(0).getSubject());

    }

    public static class MockTransport extends Transport{
        private static List<Message> sentMessages = new ArrayList<>();

        public MockTransport(Session session, URLName urlname) {
            super(session, urlname);
        }

        @Override
        public void connect(String host, int port, String username, String password) throws MessagingException {
            //do nothing.
        }

       @Override
       public void connect() throws MessagingException{
          //do nothing.
       }

       @Override
       public void sendMessage(Message message, Address[] addresses) throws MessagingException {
          sentMessages.add(message);
       }

       @Override
       public void close() throws MessagingException {
           //do nothing
       }

       public static List<Message> getSentMessages(){
           return sentMessages;
       }
   }

    public class EmailService {
         private final Session session;

         public EmailService(Session session){
           this.session = session;
         }

        public void sendEmail(Message message) throws MessagingException {
            Transport transport = session.getTransport("custommock");
            transport.connect();
            transport.sendMessage(message, message.getAllRecipients());
            transport.close();
        }
    }
}

```

in this sample, we create a custom `MockTransport` class that extends `javax.mail.Transport` and instead of sending emails, it stores the sent messages in a static list. then the tests can be done to assert the behavior of this `MockTransport`. this method requires creating a custom mail provider and setting it on the `session`. the email service uses this `custommock` protocol to send emails. again, it is a more involved solution but it works. some people prefer this way because they don't like the mocks approach.

a third alternative is to wrap the `javax.mail` operations into an interface, and then mock the implementation of this interface. this method is usually better than the previous two because you have better flexibility and control over what you want to do in the mock, and in general, is the best from a design perspective. for instance:

```java
import javax.mail.*;
import javax.mail.internet.MimeMessage;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.Arrays;
import java.util.List;


public class MockMailInterfaceTest{

  @Test
    public void testMailInterfaceMock() throws Exception{
        MailSender mailSender = mock(MailSender.class);
        EmailService emailService = new EmailService(mailSender);

        Message message = new MimeMessage(Session.getDefaultInstance(System.getProperties()));
        message.setRecipient(Message.RecipientType.TO, new javax.mail.internet.InternetAddress("test@example.com"));
        message.setSubject("test subject");
        message.setText("test message");


        emailService.sendEmail(message);
        //verify that the send method is called exactly once
        verify(mailSender, times(1)).send(any());

        //add a mock behavior that returns the sent email
        when(mailSender.send(any())).thenReturn(Arrays.asList(message));

        List<Message> sentMessages = emailService.sendEmail(message);
        assertEquals(1, sentMessages.size());
        assertEquals("test subject", sentMessages.get(0).getSubject());
    }


    public interface MailSender {
       List<Message> send(Message message) throws MessagingException;
    }

    public static class JavaMailSender implements MailSender {
        private final Session session;
        public JavaMailSender(Session session){
            this.session = session;
        }

        @Override
        public List<Message> send(Message message) throws MessagingException {
             Transport transport = session.getTransport("smtp");
             transport.connect("localhost", 25, "user", "password");
             transport.sendMessage(message, message.getAllRecipients());
             transport.close();
            return  Arrays.asList(message);
        }
    }

    public class EmailService {
         private final MailSender mailSender;

         public EmailService(MailSender mailSender){
           this.mailSender = mailSender;
         }

        public List<Message> sendEmail(Message message) throws MessagingException {
            return mailSender.send(message);
        }
    }

}

```

in this solution we define a `MailSender` interface that represents the operation of sending a message. then we have a class `JavaMailSender` that uses the real `javax.mail` and an implementation of `MailSender`. in tests, we use a mock of `MailSender` interface, and we can easily verify that the `send` method was called. we have now all the flexibility of mocking with an interface, and we can easily create different mocks for specific behavior.

this is my preferred approach because you gain a better decoupling and modularity of the email sending logic. in fact, you can replace the current `javax.mail` with a more modern library or even a third-party email service, without breaking the application, only having to change the implementation of the `MailSender` interface. this is the big advantage here, better decoupling.

when it comes to resources, i'd recommend checking out "effective java" by joshua bloch. while not specifically about javax.mail mocking, it's packed with good practices about interfaces and how to write testable code. also look at "xunit test patterns", by gerard meszaros, as it has a lot of useful patterns about how to design a test suite.

as a final note, i know it's tempting to always use the simplest solution, but take some time to consider the implications. mocking directly the `javax.mail.Transport` or creating custom implementations are quick solutions, but using an interface like the third approach is a more robust solution in the long run.

and lastly, and probably my only joke about this: email sending code... it's like trying to herd cats, but at least you can mock the cats for testing. i hope this helps, and let me know if you have more questions.
