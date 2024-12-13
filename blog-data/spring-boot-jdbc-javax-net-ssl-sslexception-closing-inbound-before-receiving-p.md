---
title: "spring boot jdbc javax net ssl sslexception closing inbound before receiving p?"
date: "2024-12-13"
id: "spring-boot-jdbc-javax-net-ssl-sslexception-closing-inbound-before-receiving-p"
---

Okay so you got that dreaded "javax net ssl SSLException closing inbound before receiving p" thing huh Been there done that got the t-shirt and the therapy bills honestly I feel your pain Its like the internet just decided to ghost you mid-handshake Not cool I tell you not cool

So lets break this down because this isnt a simple "just restart your server" type of thing Yeah I wish it was We are diving into the slightly messy world of SSL/TLS handshakes gone south

First off this error is pretty much what it sounds like a javax net ssl SSLException that occurs when the inbound side the part of the connection you are receiving data on closes before a full SSL/TLS handshake has been completed before receiving whatever the system expects that p thing to be which in most cases is the application data after that SSL handshake the "p" part is just the way java prints it I guess The handshake is the dance your app does with the database or external service to make sure they can talk to each other securely with encryption and such. When this handshake is interrupted before it’s finished that exception happens and its never pretty

Now I've seen this happen for a bunch of reasons sometimes obvious sometimes not-so-obvious. One of the big culprits is a mismatch in the SSL/TLS versions that are being used. Old servers want SSLv3 or TLSv1.0 new servers want TLSv1.2 or TLSv1.3 and if your app and database are not on the same page it’s a straight-up failure to connect. This is by far the most common reason i encountered this issue so lets start with that first.

I remember one time I was working on this e-commerce app back in 2015 when TLS 1.0 and 1.1 were being deprecated it was a total nightmare We spent two days figuring out that the database servers were still running on TLS 1.0 and our Spring Boot app using Spring Security was trying to use TLS 1.2 They were just shouting at each other in different languages no surprise they didn't get along.

So first thing lets check our app configuration. We need to make sure that the SSL/TLS protocol is aligned with what the database server or whatever service your app is talking to wants You can specify the supported protocols in the `application.properties` or `application.yml` file or programatically. Check the Java doc API for the ssl and trust manager if you are doing it programatically.

Here is an example of how you configure it in a `application.properties` file:

```properties
server.ssl.enabled=true
server.ssl.key-store=classpath:keystore.jks
server.ssl.key-store-password=your_keystore_password
server.ssl.key-password=your_key_password
server.ssl.key-store-type=JKS
server.ssl.trust-store=classpath:truststore.jks
server.ssl.trust-store-password=your_truststore_password
server.ssl.trust-store-type=JKS
server.ssl.protocol=TLSv1.2
server.ssl.ciphers=TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384,TLS_RSA_WITH_AES_256_CBC_SHA256,TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256,TLS_RSA_WITH_AES_128_CBC_SHA256
```

This configuration sets up TLS v1.2 as the required SSL protocol and also defines the enabled ciphers in a safe manner you may have to change ciphers depending on your needs and the server you are connecting to. I’ve been there where I set the protocol to TLS 1.3 and the database refused to accept the request so this is often a negotiation between the server and client.

Now if its not that you need to check if your truststore and keystore is configured correctly this is where the certificates live that your server uses to identify itself and the other server to identify itself the keystore contains the server’s certificate and private key while the truststore contains the certificate authorities that you trust. If either of these is messed up the handshake will fail because neither side can verify the other side's identity.

Here's how to set this up when you are using the spring jdbc template when setting up your DataSource (using Hikari connection pool in this example),

```java
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.util.Properties;

@Configuration
public class DatabaseConfig {

    @Value("${spring.datasource.url}")
    private String dbUrl;

    @Value("${spring.datasource.username}")
    private String dbUsername;

    @Value("${spring.datasource.password}")
    private String dbPassword;

    @Value("${server.ssl.trust-store}")
    private String trustStorePath;

    @Value("${server.ssl.trust-store-password}")
    private String trustStorePassword;

    @Value("${server.ssl.key-store}")
    private String keyStorePath;

    @Value("${server.ssl.key-store-password}")
    private String keyStorePassword;


    @Bean
    public DataSource dataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(dbUrl);
        config.setUsername(dbUsername);
        config.setPassword(dbPassword);


        Properties sslProperties = new Properties();
        sslProperties.setProperty("javax.net.ssl.trustStore", trustStorePath);
        sslProperties.setProperty("javax.net.ssl.trustStorePassword", trustStorePassword);
        sslProperties.setProperty("javax.net.ssl.keyStore",keyStorePath);
        sslProperties.setProperty("javax.net.ssl.keyStorePassword", keyStorePassword);
        sslProperties.setProperty("javax.net.ssl.keyStoreType","JKS");
        sslProperties.setProperty("javax.net.ssl.trustStoreType","JKS");


        config.setDataSourceProperties(sslProperties);
        return new HikariDataSource(config);
    }

   @Bean
    public JdbcTemplate jdbcTemplate(DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }

}
```

The most important part here is in the `dataSource()` method where we set `javax.net.ssl.trustStore` and `javax.net.ssl.trustStorePassword` properties to your truststore and keystore locations. Make sure that the path to the store is valid inside your docker container or where you run the application. Also the keystore password and the key password should be exactly the same if you have created the keystore and the key at the same time using java keytool which is the common way of generating them.

If you are using RestTemplate or other types of HTTP clients then you will have to configure the SSLContext and the trust manager for those specific classes depending on the client.

Now if you are still hitting this issue you might want to check the database server logs because it might also be the case that the database server side has problems establishing the secure connection It might have expired certificates or misconfigured server side settings. I’ve had issues where the client cert was not being recognized by the database because the database trustore was outdated

And that's why I often make sure to have separate keystores and truststores just in case something goes wrong on the server or client side it also makes it easier to maintain different certificates and authorities. If you mess it up you will get that javax net ssl SSLException closing inbound before receiving p thing or the other way around if its the server having issues.

I remember once I was pulling my hair out over this and it turned out the database admin had swapped the truststore for a test one and forgot to put it back We wasted hours debugging that.

There is also the network side if you are deploying on cloud environments you will need to check the network policy to make sure that connections are being established properly and not being blocked at the load balancers or security groups. Sometimes you might have a firewall issue on the network where your app and database reside and you wont get this error message but the connection will simply fail. Usually you can trace the route by using `traceroute` or `mtr` commands from a linux environment.

You might need to also check your cipher suite settings as some ciphers can be vulnerable or not supported by some servers or clients and this might be causing this SSLException. You need to make sure both sides agree on some safe cipher suites. If you are not sure about this then its better to use the default ones that java or other tools recommend.

If after checking all of that you are still facing this issue, you can use a proxy tool like wireshark and enable the SSL debug to capture the traffic and analyze the handshake messages to see what is failing. You can also use the java system properties to enable SSL debug like this -Djavax.net.debug=ssl,handshake,data

```java
public class Main {
    public static void main(String[] args) {
        System.setProperty("javax.net.debug", "ssl,handshake,data");
        // Your connection code goes here
    }
}
```

This will spit out a ton of information on the console and can be quite verbose but this will provide very useful information about what exactly is going wrong. Remember to remove this flag when you go into production.

For further reading and detailed explanations on SSL/TLS handshake process I would highly recommend "Serious Cryptography A Practical Introduction to Modern Encryption" by Jean-Philippe Aumasson and “Cryptography Engineering” by Niels Ferguson, Bruce Schneier and Tadayoshi Kohno and to understand more on the SSL handshakes there is also RFC 5246 https://www.rfc-editor.org/rfc/rfc5246 which is the standard for TLS 1.2 and RFC 8446 which is the standard for TLS 1.3. I really like reading the RFCs for understanding the underlying mechanism of how the things work on a lower level. I've also used the book "Java Security" by Scott Oaks for java specific implementations of the security and trust management features. You might also find interesting papers on secure TLS configurations and certificates from the National Institute of Standards and Technology (NIST).

Debugging SSL/TLS issues can feel like pulling teeth but with a systematic approach you’ll find it Its just like when you are working with javascript and you forget a simple semicolon somewhere in your react application or even worse in node js and spend hours trying to figure out what is wrong Its a never ending story if you ask me

I hope this helps and good luck debugging. Let me know if you have other issues and I will try my best to assist.
