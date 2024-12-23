---
title: "How can I install an older version of Laravel using Sail?"
date: "2024-12-23"
id: "how-can-i-install-an-older-version-of-laravel-using-sail"
---

Alright, let's tackle this. I remember a project a while back, where we needed to maintain a legacy system built on Laravel 5.8, and getting that to play nice with modern docker tooling like Sail was… a journey. The default sail setup tends to target the latest stable release, so rolling back to an older version needs a bit of manual intervention. It's not terribly complex, but it requires understanding a few moving parts. Essentially, we need to explicitly specify the Laravel version during the project creation process and tailor the docker configuration accordingly.

First, let's focus on the initial scaffolding. When you start a new Laravel project with sail, the command `curl -s "https://laravel.build/example-app" | bash` grabs the latest build, which will install the most current Laravel version. To get the older version we need to use the --laravel argument, like so:

`curl -s "https://laravel.build/example-app" | bash -s -- --laravel=5.8`

This command tells Laravel installer to specifically setup a version 5.8 project and does not use the latest version by default. This is the first crucial step. Once the project is created the docker files will be created as well, but the Dockerfiles created here will use a php version that is compatible with Laravel 5.8, which at the time was PHP 7.2.

Now, even though the application will be on an older version, there are no other changes that have to be done to the docker configuration, as long as the version is a supported version. Older versions of php like 7.2 are still available in docker so this shouldn't present a problem.

Let's consider a slightly more complex scenario, say you need to install Laravel 6.20, which was a specific patch release. While there isn’t an officially supported way of doing this with the basic laravel setup, you could change the composer.json file directly and install it as such. This is a more advanced, but completely workable, method:

1. **Project Creation:** Start by creating a new Laravel project *without* specifying a version, let's use the example command for the latest version
`curl -s "https://laravel.build/example-app" | bash`
2. **Adjust composer.json:** Navigate to the project's root directory where the composer.json lives. Change the line for the `"laravel/framework"` to specify the specific version, for example `"laravel/framework": "6.20.*"`. It's important to use a wildcard here because specific patch versions are often not needed, and Composer is intelligent about resolving to the newest patch.
3. **Composer Update:** Run `composer update` in the project's root. This command will use the updated composer.json and install the specific version specified.

Here’s a working code snippet demonstrating the modified `composer.json`:

```json
{
    "name": "laravel/laravel",
    "type": "project",
    "description": "The Laravel Framework.",
    "keywords": [
        "framework",
        "laravel"
    ],
    "license": "MIT",
    "require": {
        "php": "^7.2|^8.0",
        "fideloper/proxy": "^4.4",
        "fruitcake/laravel-cors": "^2.0",
        "guzzlehttp/guzzle": "^7.0.1",
        "laravel/framework": "6.20.*", // SPECIFIC VERSION HERE
        "laravel/tinker": "^2.5"
    },
    "require-dev": {
        "facade/ignition": "^2.0",
        "fakerphp/faker": "^1.9.1",
        "mockery/mockery": "^1.3.1",
        "nunomaduro/collision": "^5.0",
        "phpunit/phpunit": "^9.3.3"
    },
    "config": {
        "optimize-autoloader": true,
        "preferred-install": "dist",
        "sort-packages": true
    },
    "extra": {
        "laravel": {
            "dont-discover": []
        }
    },
    "autoload": {
        "psr-4": {
            "App\\": "app/"
        },
        "classmap": [
            "database/seeds",
            "database/factories"
        ]
    },
    "autoload-dev": {
        "psr-4": {
            "Tests\\": "tests/"
        }
    },
    "minimum-stability": "dev",
    "prefer-stable": true,
    "scripts": {
        "post-autoload-dump": [
            "Illuminate\\Foundation\\ComposerScripts::postAutoloadDump",
            "@php artisan package:discover --ansi"
        ],
        "post-root-package-install": [
            "@php -r \"file_exists('.env') || copy('.env.example', '.env');\""
        ],
        "post-create-project-cmd": [
            "@php artisan key:generate --ansi"
        ]
    }
}
```

After modifying composer.json you would proceed with `composer update` to apply the changes.

This method gives you greater control but requires that you are comfortable editing the `composer.json` file directly and updating using the cli.

Now, if you are working with very old versions (e.g. Laravel 5.5 or earlier), you might encounter compatibility issues with the default Sail configurations. In that scenario you may need to explicitly modify the Dockerfile, or use an older version of PHP and other dependencies within your docker setup. Here's a simplified example of what that might look like. This example is a slightly modified dockerfile from a legacy project. You may have to adjust the node version according to your specific needs. This is something that I have personally encountered:

```dockerfile
FROM php:7.2-fpm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libpng-dev \
    libzip-dev \
    zip \
    unzip \
    nano

# Install PHP extensions
RUN docker-php-ext-install pdo_mysql mbstring exif pcntl bcmath zip

# Install Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Set working directory
WORKDIR /var/www/html

# Add user to www-data group
RUN usermod -u 1000 www-data && chown -R www-data:www-data /var/www/html

# Install Node.js and npm
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs

# Install yarn
RUN npm install -g yarn

USER www-data
```

This example demonstrates a Dockerfile that specifies php 7.2, installs system dependencies, and the correct php extentions. This Dockerfile could be used in your specific docker folder along side the docker-compose.yml file. The important thing is to correctly map the location where you keep your dockerfile to the compose file. The relevant section in docker-compose.yml would be similar to:

```yaml
  app:
      build:
        context: ./docker
        dockerfile: Dockerfile
      ports:
          - '${APP_PORT:-80}:80'
      environment:
          WWWUSER: '${WWWUSER:-1000}'
```

In conclusion, while Sail is designed to streamline development with the latest Laravel version, it's absolutely manageable to work with older iterations. The key is to be precise with your `composer.json` file and understand that when dealing with very old Laravel versions, you might have to delve into Dockerfile adjustments. This is not the norm for most common versions and use cases. Always test your configurations in isolated environments first. If you’re getting involved with legacy projects, the Laravel documentation for specific versions is crucial, as well as understanding the compatibility of each Laravel version and its required PHP version. There are also excellent resources, such as the “Laravel Up & Running” book by Matt Stauffer, which has sections that dive deep into composer and deployments. Also consider looking into the excellent "Pro Docker" book from O'Reilly for a deep dive into docker if you feel the need to fine tune your docker experience. These can help greatly if you're working with older versions or needing to tailor the underlying configuration.
