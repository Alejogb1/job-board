---
title: "Why is Cliver::Dependency::NotFound occurring in production Rails Capybara tests?"
date: "2024-12-23"
id: "why-is-cliverdependencynotfound-occurring-in-production-rails-capybara-tests"
---

Alright, let's talk about that pesky `Cliver::Dependency::NotFound` error you’re seeing in your production Rails Capybara tests. This isn't an uncommon issue, and I’ve certainly battled it myself on several occasions – notably during that migration of our legacy system to a containerized environment a few years back. It's frustrating because, locally, things usually work just fine, but as soon as you push to production, bam, it hits you with the dependency error. Let’s unpack why this happens, and more importantly, how to fix it.

The core of the issue usually boils down to missing or incorrectly configured dependencies required by the `Cliver` gem, which Capybara sometimes uses for driver-specific functionality, particularly when you're dealing with headless browsers like Chrome or Firefox. Cliver’s job is to locate these binaries – like `chromedriver` or `geckodriver` – on the system. When it can't find them in the places it expects, it throws that very `NotFound` error. The problem isn’t always about the binaries themselves, sometimes it's about how and where those binaries are being looked for in different environments, which is where local dev differs from production server setups.

In a typical development environment, these dependencies might be installed directly on your machine, often via gem dependencies or manual installation. But production servers are typically more isolated, perhaps using Docker containers, or with stricter filesystem policies. The gem might be present, but the crucial executable might not be accessible in the container or on the path where `Cliver` is searching for it. This disparity leads to the error.

Here are the scenarios I've encountered, along with solutions that I’ve found effective.

**Scenario 1: Missing Executable on the Server**

The most frequent issue is simply that the driver executable (e.g., `chromedriver`, `geckodriver`) isn’t installed or present in a directory where `Cliver` will look for it.

Here’s a simple code snippet demonstrating how to set the path explicitly in your `rails_helper.rb` or a similar configuration file, which is often a good starting point for debugging:

```ruby
# rails_helper.rb or similar setup file
require 'capybara/rails'

Capybara.register_driver :chrome_headless do |app|
  options = Selenium::WebDriver::Chrome::Options.new
  options.add_argument('--headless')
  options.add_argument('--no-sandbox') # Recommended for Docker/CI
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument('--window-size=1280,800')

  chrome_bin = ENV.fetch('GOOGLE_CHROME_BIN', nil)

  if chrome_bin
      options.binary = chrome_bin
  end


  Capybara::Selenium::Driver.new(app, browser: :chrome, options: options)
end

Capybara.default_driver = :chrome_headless
Capybara.javascript_driver = :chrome_headless

# Explicitly set chromedriver path
ENV['PATH'] = "#{ENV['PATH']}:/path/to/your/chromedriver_directory" # Adapt this path for your environment
```

In the above snippet, notice we added an extra line that prepends the path to your `chromedriver` directory to the environment's `PATH`. This ensures that `Cliver` has a clear path to find the executable. Remember to adjust `/path/to/your/chromedriver_directory` to the actual path on your server.

**Scenario 2: Incorrect Path in Docker Environment**

When you're using Docker, the path to your driver executable might not be the same as on your local machine or even on a traditional server. The above fix still applies, but you might need to incorporate the executable into the Docker image itself.

Here’s how to modify your `Dockerfile` to ensure `chromedriver` is installed and accessible:

```dockerfile
# Dockerfile example
FROM ruby:3.1-slim

RUN apt-get update && apt-get install -y wget unzip
# Install Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
RUN apt-get update && apt-get install -y google-chrome-stable

# Install chromedriver
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip -d /opt/
RUN chmod +x /opt/chromedriver
ENV PATH="${PATH}:/opt" # Add to path

WORKDIR /app
COPY Gemfile Gemfile.lock ./

RUN bundle install
COPY . .

# other build instructions
```

This `Dockerfile` snippet performs several steps: it downloads and installs Google Chrome and then downloads the compatible `chromedriver`, places it in `/opt/`, makes it executable, and adds `/opt` to the `PATH` environment variable so that `Cliver` can find it when running the tests in your container. Again, verify the `chromedriver` URL matches the chrome version installed for compatibility.

**Scenario 3: The `chromedriver` Version Is Mismatched**

Another less obvious but equally problematic issue is when the version of `chromedriver` doesn't align with the version of Chrome you’re using. This mismatch leads to unpredictable behavior and often manifests as the `Cliver::Dependency::NotFound` error or related issues during test execution.

To avoid this, it's crucial to ensure that the installed version of the `chromedriver` matches the version of Google Chrome installed in your environment. A discrepancy can happen if, for instance, a gem update pulls in a driver that is not compatible, or if a new Chrome version is automatically installed in your environment, breaking compatibility.

```ruby
# rails_helper.rb, or similar configuration file
Capybara.register_driver :chrome_headless do |app|
  options = Selenium::WebDriver::Chrome::Options.new
  options.add_argument('--headless')
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument('--window-size=1280,800')
  chrome_bin = ENV.fetch('GOOGLE_CHROME_BIN', nil)

  if chrome_bin
      options.binary = chrome_bin
  end


    driver_path = ENV.fetch('CHROMEDRIVER_PATH', nil)
    if driver_path
        Selenium::WebDriver::Chrome.path = driver_path
    end


  Capybara::Selenium::Driver.new(app, browser: :chrome, options: options)
end

Capybara.default_driver = :chrome_headless
Capybara.javascript_driver = :chrome_headless
```

Here, we added a check for `CHROMEDRIVER_PATH` in the environment. If set, we instruct selenium directly to look for the driver at that location. This adds further flexibility especially when you start managing multiple versions across your infrastructure.

**Recommendations**

For further exploration of these topics, I recommend a few key resources:

*   **"Selenium with Ruby" by Bhasin, Sumit** – A very practical book on setting up and configuring selenium drivers and specifically talks about managing browser and driver interactions across different platforms.
*   **SeleniumHQ Documentation** - Specifically the documentation for setting up drivers with selenium, which is what `Cliver` is using under the hood.
*   **Docker Official Documentation** - For general Docker configuration and best practices on how to build robust and maintainable container images, especially related to pathing and environment variables.

In summary, `Cliver::Dependency::NotFound` in your production Rails Capybara tests primarily results from the Cliver gem being unable to locate required executable dependencies. The fixes involve ensuring correct path configurations, accurate placement of executables in Docker environments, and strict version alignment between your browser (e.g., Chrome) and its associated driver (`chromedriver`). When you see this error, don’t panic. Work methodically, checking the path, versions, and configurations I’ve detailed above, and you should be able to nail it down. Good luck.
