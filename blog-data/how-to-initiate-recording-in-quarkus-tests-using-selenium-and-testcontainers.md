---
title: "How to initiate recording in Quarkus tests using Selenium and Testcontainers?"
date: "2024-12-23"
id: "how-to-initiate-recording-in-quarkus-tests-using-selenium-and-testcontainers"
---

,  It's a scenario I've encountered quite a few times in past projects, particularly when dealing with web application testing that requires actual browser interaction. The challenge, as you've likely found, is smoothly integrating Selenium for UI actions with Testcontainers for a reproducible and isolated environment, specifically within a Quarkus test setup. It's not always straightforward, but there are several approaches that work effectively, and I've found a consistent method to achieve it.

Essentially, we need to juggle several components: the Quarkus application under test, a Selenium-driven browser instance, and a Testcontainers-managed environment, often a Docker container running a browser like Chrome or Firefox. The goal is to orchestrate these so that we can programmatically initiate recording from the Selenium side, leveraging the isolation and lifecycle management provided by Testcontainers.

The first hurdle is setting up Testcontainers to host the browser. I typically opt for a standalone Selenium grid setup, primarily because it offers more control and separation of concerns. It’s also beneficial when you need different browser versions or want to scale your testing horizontally. I’ve used the ‘selenium/standalone-chrome’ and ‘selenium/standalone-firefox’ Docker images extensively; they’re robust and well-documented.

Here's a look at how I'd typically set this up within a Quarkus test class, using JUnit 5 and the `quarkus-test` extension:

```java
import org.junit.jupiter.api.*;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import io.quarkus.test.junit.QuarkusTest;
import java.net.MalformedURLException;
import java.net.URL;

@QuarkusTest
@Testcontainers
public class RecordingTest {

    private static final DockerImageName CHROME_IMAGE = DockerImageName.parse("selenium/standalone-chrome:4.15.0-20231024");
     private static final DockerImageName FIREFOX_IMAGE = DockerImageName.parse("selenium/standalone-firefox:4.15.0-20231024");

    @Container
    private static final GenericContainer<?> chromeContainer = new GenericContainer<>(CHROME_IMAGE)
          .withExposedPorts(4444);
     @Container
    private static final GenericContainer<?> firefoxContainer = new GenericContainer<>(FIREFOX_IMAGE)
          .withExposedPorts(4444);


    private RemoteWebDriver driver;

     @BeforeEach
    void setup() throws MalformedURLException {
        ChromeOptions chromeOptions = new ChromeOptions();
        String chromeUrl = "http://" + chromeContainer.getHost() + ":" + chromeContainer.getMappedPort(4444) + "/wd/hub";
        driver = new RemoteWebDriver(new URL(chromeUrl), chromeOptions);
        //alternative driver to firefox
        //FirefoxOptions firefoxOptions = new FirefoxOptions();
        //String firefoxUrl = "http://" + firefoxContainer.getHost() + ":" + firefoxContainer.getMappedPort(4444) + "/wd/hub";
        //driver = new RemoteWebDriver(new URL(firefoxUrl), firefoxOptions);


    }


   @AfterEach
    void teardown() {
      if (driver != null) {
        driver.quit();
      }
    }


    @Test
    void testRecording() {
        driver.get("http://localhost:8081/hello"); // Assume your application runs on 8081
         //insert selenium code for assertions and web ui actions here
        Assertions.assertEquals("Hello, World!", driver.getPageSource());

    }
}
```

Here's what's going on: We're using the `@Testcontainers` annotation to signal that we're employing Docker containers. The `@Container` annotation marks the `chromeContainer` (and alternatively the firefoxContainer) variable, which uses `GenericContainer` to instantiate a Selenium standalone server. I've exposed the standard Selenium port, 4444. Inside `setup`, I set up the remote web driver to connect to this Selenium server, and I explicitly include the webdriver quit within the `teardown` to ensure proper resource cleaning. This is crucial, as failing to terminate the driver will result in resource leaks. Notice I am also establishing the URL based on the mapped port of the running container, ensuring test stability as ports are dynamic, not fixed. The test case itself, `testRecording` would contain the desired browser automation logic.

Now, regarding recording initiation, you'll typically not find a "start recording" command directly within the core Selenium API. Instead, you'd leverage browser-specific capabilities or external tools. For example, some browser drivers expose performance logs or allow capturing screenshots at intervals, which can then be assembled into a recording. This is where tools like the WebDriverManager by bonigarcia can become indispensable, as they take care of setting up specific drivers for each browser. However, it would not be used directly to record in itself, but rather provide the necessary browser driver. A specific implementation example would require the implementation of a more complex framework or library which is beyond the scope of this answer, but I have used libraries similar to `monte-screen-recorder` to implement automated recording on selenium tests.

Let's consider an alternative, simpler scenario: capturing screenshots during the test to simulate a basic recording. This can often be adequate for debugging or documentation purposes. Here's how you could modify the above example:

```java
import org.junit.jupiter.api.*;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.chrome.ChromeOptions;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import io.quarkus.test.junit.QuarkusTest;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import org.apache.commons.io.FileUtils;

@QuarkusTest
@Testcontainers
public class RecordingTest {

    private static final DockerImageName CHROME_IMAGE = DockerImageName.parse("selenium/standalone-chrome:4.15.0-20231024");

    @Container
    private static final GenericContainer<?> chromeContainer = new GenericContainer<>(CHROME_IMAGE)
          .withExposedPorts(4444);

    private RemoteWebDriver driver;

     @BeforeEach
    void setup() throws MalformedURLException {
        ChromeOptions chromeOptions = new ChromeOptions();
        String chromeUrl = "http://" + chromeContainer.getHost() + ":" + chromeContainer.getMappedPort(4444) + "/wd/hub";
        driver = new RemoteWebDriver(new URL(chromeUrl), chromeOptions);

    }

   @AfterEach
    void teardown() {
      if (driver != null) {
        driver.quit();
      }
    }

    @Test
    void testRecording() throws IOException {
        driver.get("http://localhost:8081/hello");

        takeScreenshot("screenshot1");

        // Perform some UI actions here that you want to capture.
         Assertions.assertEquals("Hello, World!", driver.getPageSource());

         takeScreenshot("screenshot2");
    }

    private void takeScreenshot(String fileName) throws IOException {
    File screenshot = ((TakesScreenshot) driver).getScreenshotAs(OutputType.FILE);
    FileUtils.copyFile(screenshot, new File("target/screenshots/" + fileName + ".png"));
    }
}
```
Here, I've added the `TakesScreenshot` interface and the `takeScreenshot` helper method. This captures a screenshot using `getScreenshotAs(OutputType.FILE)`, and saves it to a temporary file using `FileUtils`. This method is called at different times of the test to capture different browser states. The `apache-commons-io` dependency is used to simplify the saving of the screenshot.

Another approach that is worth exploring, especially if the end goal is video recording, would be to interact with browser developer tools. Modern browsers provide a rich set of APIs accessible through the devtools protocol, and libraries like selenium-devtools are available to simplify interaction with them, allowing access to data streams which can then be converted into video. Using this approach is more advanced but offers better granularity.

In terms of resources, I strongly recommend the official Selenium documentation, as well as the Testcontainers documentation, as primary resources for understanding these tools. For browser driver management, check out the WebDriverManager project documentation. Finally, if you are interested in a deeper understanding of browser performance and video recording using devtools, research the Chrome DevTools Protocol documentation and libraries such as `puppeteer` (though not applicable directly to java or selenium). These are the foundational technologies you will require to effectively implement recording within your test scenarios.

It is worth noting that implementing a full video recording in selenium can be complex as the selenium driver does not contain this functionality by default. This can be achieved by combining the above techniques, external libraries, and custom implementations. If you are in search for a simpler solution, consider integration with cloud based testing platforms that might offer these tools as part of their services.

Remember, consistent practice and understanding of the underlying mechanisms are key to mastering these techniques.
