---
title: "How to configure Nginx reverse proxy to redirect subdomains to a dynamic port?"
date: "2024-12-23"
id: "how-to-configure-nginx-reverse-proxy-to-redirect-subdomains-to-a-dynamic-port"
---

Alright, let's tackle this subdomain redirection via nginx to a dynamic port; it's a fairly common scenario in microservices architectures or development environments where services are constantly spinning up on different ports. From my experience, there's a few ways to go about it, each with their own nuances and tradeoffs, and the optimal choice often depends on the specifics of your setup. I’ve certainly tripped over my fair share of configuration files getting it just so.

First, let's dissect what we actually mean by "dynamic port." Typically, when we think of port assignments, we tend to associate a service with a specific port, say, port 8080 for a web application. However, for various reasons, these ports might not be fixed. Perhaps you have a Docker swarm dynamically assigning ports, or your application server’s runtime selects them on each startup. In these scenarios, relying on static port configurations in nginx will obviously fail.

The core problem then becomes how to tell nginx, at runtime, where to proxy the requests to if the destination port changes constantly. The solution often revolves around leveraging nginx's ability to make decisions based on variables, combined with a mechanism to update those variables as needed. We aren’t going to magically guess ports of course, so something has to feed that data to nginx.

Let's look at the most common approach, using an upstream block with variables based on some external input. I've used this approach extensively in scenarios where a separate service provides the port mapping, which is typically my preferred pattern.

Here's a simplified example. Imagine we have a simple REST API that gives us the current port for each service. Let’s say this lives at `portmapper.internal/getport?service=service_a` and returns, for example, `{"port": 3001}`. It's not the only way to solve it, but it's scalable, and you are not reliant on things like DNS or other workarounds which are less flexible.

```nginx
http {
    # Define an upstream block, using a variable in the server address
    upstream service_a_dynamic {
      resolver 127.0.0.11 valid=10s; # Use docker’s internal dns if relevant, with a validity time.
      set $service_a_port 3000; # Default port.
      
      # Fetch the dynamic port from the port mapper api.
      js_content_by_lua_block {
        local http = require "resty.http"
        local httpc = http.new()
        local res, err = httpc:request_uri("http://portmapper.internal/getport?service=service_a",{
          method="GET",
          headers = {
              ["Content-Type"] = "application/json",
            },
        })

        if not res then
           ngx.log(ngx.ERR, "Failed to fetch port for service_a:", err)
           return ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
        end
        local body = res.body
        if not body then
           ngx.log(ngx.ERR, "No response from port mapper for service_a.")
           return ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
        end

        local cjson = require "cjson"
        local json_data = cjson.decode(body)
        if json_data and json_data.port then
          ngx.var.service_a_port = json_data.port
        else
          ngx.log(ngx.ERR, "Invalid response from port mapper for service_a:", body)
          return ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
        end
      end

      server  service_a:$service_a_port; # Dynamic port here!
    }


    server {
        listen 80;
        server_name service_a.example.com;

        location / {
            proxy_pass http://service_a_dynamic;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

This snippet uses `js_content_by_lua_block`, which requires the `ngx_http_lua_module` to be compiled into your nginx. Inside this block, we're making a simple http request to our hypothetical `portmapper.internal` API. We parse the JSON and extract the port, setting the `service_a_port` variable. Finally, we pass `service_a:$service_a_port` as our server, giving us that dynamically resolved port. Keep in mind this is using lua for inline request handling; it works well but can add more complexity into nginx setup. This approach avoids the need to restart nginx for changes, and updates happen roughly every 10s.

However, let's say you want to avoid the lua dependencies. You could alternatively rely on a server that provides this port information via DNS TXT records. I have used this method successfully when dealing with environments where service discovery was already managed via DNS. I would not use it if you already have other API access patterns in place, such as the one above.

```nginx
http {
    # Define a map to dynamically fetch the port from a dns record.
    map $host $service_a_port {
        default "";
        service_a.example.com dns_txt_record.service_a._tcp.example.com;
    }
    
    # Define an upstream block, using a variable in the server address
    upstream service_a_dynamic_dns {
        resolver 127.0.0.11 valid=10s; # Use docker’s internal dns if relevant, with a validity time.
        server  service_a:$service_a_port; # Dynamic port here!
    }

    server {
        listen 80;
        server_name service_a.example.com;

        location / {
            proxy_pass http://service_a_dynamic_dns;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

This solution relies on the `map` directive to convert the hostname to a dns TXT record to query, which contains only the required port, as defined by some upstream service. This keeps things simple, but does require dns setup and can be harder to debug when something goes wrong, as it adds an extra layer between nginx and the destination server. It’s also important that this dns TXT record resolves promptly.

Finally, let's look at a slightly different approach using environment variables, suitable if your application environment setup allows such configuration to be applied. This is perhaps the simplest method, but relies on the environment variables to be correct. When containers are created, they may have their own dynamically generated ports, passed down via environment variables.

```nginx
http {
    # Define an upstream block, using an env variable.
    upstream service_a_env {
        server  service_a:${SERVICE_A_PORT}; # Dynamic port here from an env variable.
    }


    server {
        listen 80;
        server_name service_a.example.com;

        location / {
            proxy_pass http://service_a_env;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```
Here, `${SERVICE_A_PORT}` refers to an environment variable, and nginx will read it during startup. This approach is simple, but restarting nginx is required every time the port changes. In development or situations where ports change very infrequently, this method may be adequate.

When choosing between these approaches, you should assess your specific needs. The Lua based approach offers dynamic updates and real-time checking, making it ideal for highly dynamic environments, though requires more complexity. The dns based approach can be simpler, if your existing platform uses dns already. Environment variables are a good choice when the port is defined by the host environment.

For further reading, I'd recommend exploring "Nginx HTTP Server" by Richard Albury. It provides an in-depth look at nginx's architecture and configuration, and the nginx documentation itself is incredibly thorough. Also, if you're interested in advanced Lua scripting within nginx, "Programming in Lua" by Roberto Ierusalimschy is very helpful. These resources can provide a good foundation for working with complex nginx setups.

Remember, the best approach is always the one that best fits your specific scenario, balancing flexibility with simplicity.
