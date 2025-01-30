---
title: "How can I test Thorntail applications with Arquillian without using @DefaultDeployment?"
date: "2025-01-30"
id: "how-can-i-test-thorntail-applications-with-arquillian"
---
Arquillian's `@DefaultDeployment` annotation, while convenient, can lead to tightly coupled tests and hinder the independent verification of individual application components within a Thorntail application.  My experience working on large-scale microservices projects using Thorntail highlighted the critical need for more granular testing strategies, avoiding the monolithic approach inherent in relying solely on `@DefaultDeployment`. This response will outline effective alternatives, demonstrating how to leverage Arquillian's power for more focused and maintainable tests.

**1.  Understanding the Limitations of `@DefaultDeployment`**

The `@DefaultDeployment` annotation instructs Arquillian to deploy the entire application archive (e.g., the WAR file) for testing.  While this simplifies initial setup, it presents several drawbacks.  Firstly, tests become inherently dependent on the complete application context, making them slower and more difficult to debug.  A failure in one component can cascade and obscure issues within another. Secondly, it obstructs the possibility of targeted unit-like testing of individual components,  making it challenging to adopt Test-Driven Development (TDD) effectively. Finally, maintaining these tests becomes complex as the application grows; any change in one part of the application might necessitate widespread test modifications.  This experience led me to favor alternative approaches for better isolation and maintainability.

**2.  Alternative Approaches to Testing Thorntail Applications with Arquillian**

The key to overcoming the limitations of `@DefaultDeployment` lies in employing Arquillian's deployment mechanisms to selectively deploy only the necessary components for each test.  This is achievable through several approaches:

* **Using `@Deployment` with custom deployment configurations:** This offers precise control over what gets deployed. You can construct deployments from individual classes, archives, or even dynamically generated content.

* **Utilizing CDI beans and injection for testing specific services:**  This approach allows for dependency injection of mocked or stubbed services within the tested component, effectively isolating it from the rest of the application.

* **Leveraging Arquillian's extension framework:**  Custom extensions can provide sophisticated deployment strategies and test utilities tailored to your specific application architecture.


**3. Code Examples Illustrating Alternative Deployment Strategies**

Let's illustrate these alternatives with practical examples.  Assume we have a Thorntail application with a `UserService` and a `NotificationService`.


**Example 1: Testing `UserService` independently using `@Deployment`**


```java
import org.jboss.arquillian.container.test.api.Deployment;
import org.jboss.arquillian.junit.Arquillian;
import org.jboss.shrinkwrap.api.Archive;
import org.jboss.shrinkwrap.api.ShrinkWrap;
import org.jboss.shrinkwrap.api.asset.EmptyAsset;
import org.jboss.shrinkwrap.api.spec.JavaArchive;
import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertNotNull;

@RunWith(Arquillian.class)
public class UserServiceTest {

    @Deployment
    public static Archive<?> createDeployment() {
        return ShrinkWrap.create(JavaArchive.class)
                .addClass(UserService.class) // Only deploy UserService
                .addClass(User.class)       // Add necessary dependencies
                .addAsManifestResource(EmptyAsset.INSTANCE, "beans.xml");
    }

    @Inject
    private UserService userService;

    @Test
    public void testCreateUser() {
        User user = userService.createUser("testuser", "password");
        assertNotNull(user);
    }
}
```

This example demonstrates deploying only the `UserService` and its dependencies. This isolation ensures that the test focuses solely on the `UserService` functionality, avoiding any interference from the `NotificationService` or other parts of the application.  The use of `ShrinkWrap` allows fine-grained control over the deployment archive.

**Example 2:  Testing with Mocked Dependencies via CDI Injection**


```java
import org.jboss.arquillian.container.test.api.Deployment;
import org.jboss.arquillian.junit.Arquillian;
import org.jboss.shrinkwrap.api.Archive;
import org.jboss.shrinkwrap.api.ShrinkWrap;
import org.jboss.shrinkwrap.api.asset.EmptyAsset;
import org.jboss.shrinkwrap.api.spec.JavaArchive;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import javax.inject.Inject;

import static org.mockito.Mockito.verify;

@RunWith(Arquillian.class)
public class UserServiceTest {

    @Mock
    private NotificationService notificationService; // Mock the dependency

    @InjectMocks
    private UserService userService; // Inject the mock into UserService

    @Deployment
    public static Archive<?> createDeployment() {
        return ShrinkWrap.create(JavaArchive.class)
                .addClass(UserService.class)
                .addClass(User.class)
                .addAsManifestResource(EmptyAsset.INSTANCE, "beans.xml");
    }

    @Test
    public void testCreateUserWithNotification() {
        MockitoAnnotations.initMocks(this); // Initialize mocks
        userService.createUser("testuser", "password");
        verify(notificationService).sendNotification("testuser");
    }
}
```

Here, `NotificationService` is mocked using Mockito. This enables testing `UserService`'s interaction with `NotificationService` without actually deploying or interacting with the real implementation. This approach is crucial for isolating unit-like tests and ensuring that changes in one service do not affect the tests of another.

**Example 3:  A more complex scenario using a custom Arquillian extension (Conceptual Outline)**


This example outlines the concept; the actual implementation would be significantly more involved and depend on the specifics of the extension.

A custom extension could provide a mechanism for dynamically configuring deployments based on test annotations or parameters.  For instance, an annotation `@DeployComponent(component = "NotificationService")` could be created.  The custom extension would intercept this annotation and selectively deploy the specified component.  This level of control empowers dynamic test configurations, particularly beneficial for complex applications.


```java
//This is a conceptual outline, the actual implementation requires a custom Arquillian Extension.

@RunWith(Arquillian.class)
public class ComplexServiceTest {

    @Test
    @DeployComponent(component = "NotificationService")
    public void testNotificationService(){
        //Test logic interacts with the deployed NotificationService
    }

    @Test
    @DeployComponent(component = "UserService")
    public void testUserService(){
        //Test logic interacts with the deployed UserService
    }
}
```


**4.  Resource Recommendations**

Thorntail documentation,  Arquillian documentation, the JBoss ShrinkWrap documentation, and a good understanding of CDI (Contexts and Dependency Injection) are invaluable for effective Thorntail testing with Arquillian. Mastering these resources is crucial for constructing well-structured, maintainable test suites.  Additionally, familiarizing oneself with mocking frameworks like Mockito or EasyMock proves beneficial for isolating components during testing.

By abandoning the reliance on `@DefaultDeployment` and embracing these alternative techniques, one can significantly improve the quality, maintainability, and efficiency of their Thorntail testing strategy. The benefits extend beyond simple unit testing, enabling comprehensive integration testing with far greater control and precision. The granular nature of these approaches promotes a more robust and sustainable testing framework, a crucial factor in the long-term success of any project.
