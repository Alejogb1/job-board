---
title: "How do I configure Nginx with Passenger to serve a Rails 6 app using Webpacker?"
date: "2024-12-23"
id: "how-do-i-configure-nginx-with-passenger-to-serve-a-rails-6-app-using-webpacker"
---

Alright,  It's a question I've encountered countless times, and honestly, each project throws its unique curveball. The combination of Nginx, Passenger, and Webpacker in a Rails 6 environment definitely has some nuances, but it's manageable if we approach it systematically.

From memory, back when I was optimizing a high-traffic e-commerce site – lets call it ‘Project Phoenix’ – we initially struggled with slow asset loading and deployment headaches. We ultimately settled on this architecture, and I learned some solid lessons along the way. The configuration revolves around several key aspects: ensuring Passenger plays nicely with Webpacker's output, setting up the necessary Nginx directives, and understanding the interplay between these components.

First, let's address the core idea: Passenger essentially acts as an application server, tightly coupled with Nginx. It understands how to run your Rails application efficiently, but we need to guide it on where to find the actual application code and importantly, where to locate the compiled assets produced by Webpacker.

The pivotal point here is the `public` directory, which Webpacker outputs its generated assets into, specifically under `public/packs`. Passenger, by default, serves everything from the `public` directory, which is precisely what we want, but its important that the `root` directive is configured to reflect the correct path on the system. Let's take a look at some Nginx configuration snippets.

Here's a basic server block configuration:

```nginx
server {
  listen 80;
  server_name your_domain.com;
  root /path/to/your/rails/app/public;

  passenger_enabled on;
  passenger_ruby /path/to/your/ruby/executable;
  passenger_app_env production;

  location ~ ^/packs/ {
        gzip_static on;
        expires max;
        add_header Cache-Control public;
   }
}
```

Key points:

*   `listen 80;` obviously dictates that we are listening on standard http port, this should be changed to `443` for https traffic.
*   `server_name your_domain.com;` should be replaced with the domain you will be serving.
*   `root /path/to/your/rails/app/public;` This is crucial, as it directs Nginx (and thus, Passenger) to use the `public` directory as the root for static file serving. Replace `/path/to/your/rails/app/public` with the correct path on your server.
*   `passenger_enabled on;` turns on passenger integration.
*   `passenger_ruby /path/to/your/ruby/executable;` specifies the Ruby executable which passenger should be using. Replace `/path/to/your/ruby/executable` with the correct path on your server.
*   `passenger_app_env production;` denotes the environment which is being used, which should match the environment used in your rails app.
*   The `location ~ ^/packs/ { ... }` block is designed to handle the `packs` directory specifically. The `gzip_static on;` directive allows Nginx to pre-compress and serve pre-compressed assets, reducing server load. The `expires max;` and `add_header Cache-Control public;` directives optimize caching, drastically improving load times. This block will mean that any URL starting with `/packs/` will have optimal caching, thus reducing load times on subsequent requests.

Now, let’s enhance this. In `Project Phoenix`, we needed to handle different environments and custom configurations. We adopted a more structured approach:

```nginx
server {
  listen 80;
  server_name your_domain.com;
  root /path/to/your/rails/app/public;

  passenger_enabled on;
  passenger_ruby /path/to/your/ruby/executable;
  passenger_app_env production;
  
  # ensure passenger picks up any changes in the root folder
  passenger_restart_dir /path/to/your/rails/app/tmp;
  
  location / {
        # This directive is added to prevent passenger errors in the nginx error logs.
        try_files $uri @passenger;
   }

    location @passenger {
        passenger_app_env production;
    }
  
    location ~ ^/packs/ {
        gzip_static on;
        expires max;
        add_header Cache-Control public;
    }

  error_log /var/log/nginx/error.log;
  access_log /var/log/nginx/access.log;
}
```

The changes are important:

*   `passenger_restart_dir /path/to/your/rails/app/tmp;` This directive is essential. If you make changes to your Ruby files, then passenger needs to be restarted, this directive means that passenger will automatically restart if any changes are made within the directory, as determined by passenger.
*   `location / { try_files $uri @passenger; }` The try_files directive is key here. It tells Nginx to first check if a file exists at the requested URI; if not, then it passes the request to the named location @passenger.
*  `location @passenger { passenger_app_env production; }` This is the named location `try_files` will use.

This ensures that Passenger handles all dynamic requests efficiently. Also, log files are specifically included, for debugging purposes.

Finally, let's look at a more detailed configuration where we might be hosting multiple applications behind the same Nginx server, using subdomains. This is a common scenario, and it often calls for slight variations. Assume we’re using subdomains `app1.example.com` and `app2.example.com`:

```nginx
server {
    listen 80;
    server_name app1.example.com;
    root /path/to/app1/public;

    passenger_enabled on;
    passenger_ruby /path/to/app1/ruby/executable;
    passenger_app_env production;
    passenger_restart_dir /path/to/app1/tmp;

    location / {
         try_files $uri @passenger_app1;
    }

    location @passenger_app1 {
         passenger_app_env production;
     }

    location ~ ^/packs/ {
        gzip_static on;
        expires max;
        add_header Cache-Control public;
    }
     
   error_log /var/log/nginx/app1-error.log;
    access_log /var/log/nginx/app1-access.log;
}

server {
    listen 80;
    server_name app2.example.com;
    root /path/to/app2/public;

    passenger_enabled on;
    passenger_ruby /path/to/app2/ruby/executable;
    passenger_app_env production;
    passenger_restart_dir /path/to/app2/tmp;

    location / {
       try_files $uri @passenger_app2;
    }

   location @passenger_app2 {
       passenger_app_env production;
   }

    location ~ ^/packs/ {
        gzip_static on;
        expires max;
        add_header Cache-Control public;
    }
    
     error_log /var/log/nginx/app2-error.log;
     access_log /var/log/nginx/app2-access.log;
}
```

Here, each server block corresponds to a specific application, with unique root paths, ruby executables, application names and log files. Each application is now independent from the other. This structure is essential for managing multiple Rails applications using the same Nginx instance.

Several resources were invaluable in getting ‘Project Phoenix’ up and running and stable. I would strongly recommend:

*   **The official Phusion Passenger documentation:** This is, without question, the most authoritative source. It covers everything in detail, including troubleshooting, and is readily available from Phusion themselves. It would be best to consult the Nginx documentation on the relevant modules to understand what each directive does.
*   **"The Nginx Cookbook" by Derek DeJonghe:** This book provides a comprehensive guide to Nginx, including a plethora of practical examples and best practices. It provides clarity on some of the core aspects of Nginx and how to deploy and structure configurations.
*   **"Programming Ruby" by Dave Thomas, Chad Fowler, and Andy Hunt:** While not solely focused on deployment, understanding Ruby is crucial for effectively troubleshooting deployment issues. Having an in-depth understanding of your programming language allows you to better understand errors and issues.

Configuration can seem complicated at first glance, but it’s just about understanding the interactions between the tools, and ensuring the configuration reflects the architecture and needs of your application. It's important to test, test, and retest any configuration changes in a staging environment before moving to production. You will encounter issues, but having the basic foundations down means debugging will be much easier.
