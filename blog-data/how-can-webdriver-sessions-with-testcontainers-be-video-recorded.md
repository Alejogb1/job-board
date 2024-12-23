---
title: "How can WebDriver sessions with TestContainers be video-recorded?"
date: "2024-12-23"
id: "how-can-webdriver-sessions-with-testcontainers-be-video-recorded"
---

,  It's a common challenge I've seen crop up quite a few times, especially when debugging those tricky, intermittent UI test failures. Capturing the full session, in video, provides invaluable context. I recall back at 'XyloTech', we had a complex e-commerce platform, and hunting down the cause of random checkout errors was like chasing shadows. We finally implemented this recording solution and it significantly accelerated our diagnosis speed. Let's break down how you can achieve this with WebDriver and Testcontainers.

The core issue is that Testcontainers, while fantastic for managing containerized dependencies like browsers, doesn't inherently provide video recording capabilities for WebDriver sessions. Therefore, we need to introduce a mechanism that sits between the WebDriver actions and the browser, capturing the display output. A relatively straightforward approach is using a VNC server within the container coupled with recording software, or, perhaps more elegantly, leveraging the built-in screen recording capabilities of a suitable headless browser image. My preference leans towards the latter, as it simplifies the setup.

We’ll focus on using a browser image that already has screen recording baked in; selenium/standalone-chrome or selenium/standalone-firefox often provides capabilities for this (check the specific tags on docker hub for 'recording' options). The key is that we need to configure the browser's options so that the recording starts alongside the browser session and ensure the resulting video file is accessible from your host machine.

Here's the general process, followed by specific examples using Python, Java, and Node.js:

1.  **Select a suitable Docker image:** As previously mentioned, ensure the base docker image has recording capabilities.
2.  **Configure WebDriver Options:** When launching the WebDriver, ensure you pass specific command-line arguments (or browser options) that initiate the recording. This step is browser-specific and requires careful attention to detail.
3.  **Retrieve the Recording:** After the test session, extract the recorded video file from the container. Typically, the video is saved to a well-defined location within the container filesystem, allowing us to map this volume using Testcontainers to retrieve it onto your host.
4.  **Cleanup:** It's also important to ensure proper cleanup of the containers after testing.

**Example 1: Python**

```python
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


def record_webdriver_session_python():
    video_path = os.path.abspath("videos")
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    chrome_options = ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument(f"--screen-record-file={video_path}/recording.mp4")
    chrome_options.add_argument("--screen-record-format=mp4")

    with DockerContainer("selenium/standalone-chrome:latest") as chrome_container:

        wait_for_logs(chrome_container, "Selenium Server is ready", timeout=30)

        driver = webdriver.Remote(
            command_executor=f"http://{chrome_container.get_container_host_ip()}:{chrome_container.get_exposed_port(4444)}/wd/hub",
            options=chrome_options
        )

        driver.get("https://example.com")
        time.sleep(5)
        driver.quit()

        volume_mount = f"{video_path}:/videos"
        chrome_container.add_volume(volume_mount)

        chrome_container.stop()  #stop the container to save the video
        print("Video Recording Completed: check your local folder 'videos'")

if __name__ == "__main__":
    record_webdriver_session_python()
```

This Python example demonstrates how to initiate screen recording on a headless chrome container, save it locally to a 'videos' directory, and how to configure the driver. Note the specific chrome options related to screen recording. This leverages `testcontainers.core` to manage the selenium container directly, instead of using `testcontainers.selenium`.

**Example 2: Java**

```java
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.MountableFile;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;
import java.time.Duration;

public class WebDriverRecordingJava {

    public static void main(String[] args) throws MalformedURLException {
        String videoDir = Paths.get("videos").toAbsolutePath().toString();
        File videoFolder = new File(videoDir);
        if (!videoFolder.exists()) {
            videoFolder.mkdirs();
        }

        ChromeOptions chromeOptions = new ChromeOptions();
        chromeOptions.addArguments("--no-sandbox");
        chromeOptions.addArguments("--disable-dev-shm-usage");
        chromeOptions.addArguments("--headless=new");
        chromeOptions.addArguments("--screen-record-file=/videos/recording.mp4");
        chromeOptions.addArguments("--screen-record-format=mp4");

        try (GenericContainer<?> chromeContainer = new GenericContainer<>("selenium/standalone-chrome:latest")
                .withExposedPorts(4444)
                .waitingFor(Wait.forLogMessage("Selenium Server is ready", 1))
        )
        {
            chromeContainer.start();

            URL seleniumUrl = new URL("http://" + chromeContainer.getHost() + ":" + chromeContainer.getMappedPort(4444) + "/wd/hub");
            WebDriver driver = new RemoteWebDriver(seleniumUrl, chromeOptions);

            driver.get("https://example.com");
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            driver.quit();

            MountableFile mountedVolume = MountableFile.forHostPath(videoDir);
            chromeContainer.addFileSystemBind(mountedVolume.getMountPath(), "/videos");
            chromeContainer.stop();
             System.out.println("Video Recording Completed: check your local folder 'videos'");


        }
    }
}
```

Here, the Java code uses similar ChromeOptions and specifies a volume mount to pull the video from the docker container. It uses the Testcontainers `GenericContainer` which is more flexible than the `BrowserWebDriverContainer`. This example shows explicitly how to use mounting with `addFileSystemBind`.

**Example 3: Node.js**

```javascript
const { GenericContainer } = require('testcontainers');
const { Builder, Capabilities } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const fs = require('fs');
const path = require('path');

async function recordWebDriverSessionNode() {

    const videoPath = path.resolve('videos');
    if (!fs.existsSync(videoPath)) {
        fs.mkdirSync(videoPath, { recursive: true });
    }

    const chromeOptions = new chrome.Options();
    chromeOptions.addArguments('--no-sandbox');
    chromeOptions.addArguments('--disable-dev-shm-usage');
    chromeOptions.addArguments('--headless=new');
    chromeOptions.addArguments(`--screen-record-file=/videos/recording.mp4`);
    chromeOptions.addArguments('--screen-record-format=mp4');


    const chromeContainer = await new GenericContainer('selenium/standalone-chrome:latest')
        .withExposedPorts(4444)
        .withWaitStrategy({
            waitUntil: 'log',
            logMessage: 'Selenium Server is ready',
        })
        .start();



    const capabilities = Capabilities.chrome()
    capabilities.set(chrome.Options.name, chromeOptions);


    const driver = await new Builder()
        .usingServer(`http://${chromeContainer.getHost()}:${chromeContainer.getMappedPort(4444)}/wd/hub`)
        .withCapabilities(capabilities)
        .build();

    await driver.get('https://example.com');
    await new Promise(resolve => setTimeout(resolve, 5000));
    await driver.quit();

    const volumeMount = `${videoPath}:/videos`;
    chromeContainer.addFileSystemBind(volumeMount);
    await chromeContainer.stop();

    console.log("Video Recording Completed: check your local folder 'videos'");
}

recordWebDriverSessionNode();
```

This Node.js example, uses `testcontainers` library to start the chrome container and the selenium-webdriver to connect to it. It ensures the recording options are set, and, just like the other examples, maps a volume to retrieve the video. The `withWaitStrategy` is explicitly provided here which mirrors the java `wait.forLogMessage` approach.

**Important Notes and Further Reading**

*   **Browser Specifics:** The exact flags and options for screen recording may differ slightly between Chromium-based browsers (Chrome, Edge) and Firefox. Always consult the documentation for your specific browser version.
*   **Performance:** Video recording can impact performance and memory usage. Optimize your recording settings, such as frame rate and resolution, according to your requirements.
*   **File Management:** Implement robust logic to handle scenarios where no recording is generated (e.g., due to session failures) and implement proper cleanup of video files over time.

For further reading, I'd recommend diving deep into the WebDriver W3C specifications, specifically the capabilities for browser options and arguments which dictates how to manipulate them. The Selenium documentation on advanced browser settings is also a valuable resource. Additionally, studying the Docker documentation related to volume mounts will enhance your understanding of how the video files are extracted. You might also find it beneficial to explore the specific capabilities of the chrome or firefox driver you are using, or the specific `Dockerfile` for your `selenium/standalone-chrome` image on docker hub. Finally, always refer to the Testcontainers documentation to deeply understand what features are available.

By combining the power of Testcontainers with browser-specific recording capabilities, you can build robust UI testing setups that provide the vital debugging information needed to keep your projects running smoothly. It's an investment that pays dividends in reduced debugging time and improved confidence in your tests.
