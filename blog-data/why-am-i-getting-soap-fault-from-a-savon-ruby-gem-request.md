---
title: "Why am I getting SOAP fault from a Savon Ruby gem request?"
date: "2024-12-16"
id: "why-am-i-getting-soap-fault-from-a-savon-ruby-gem-request"
---

Alright, let's tackle this soap fault issue you're encountering with Savon. I've definitely been down that road before – more times than i care to recall, to be honest. Soap, while powerful in its own right, can be particularly finicky when it comes to implementation. So, let’s break down some common culprits for why you might be seeing those irritating `soap:fault` responses when using Savon in Ruby.

First, it's important to remember that a soap fault isn’t just a generic error. It’s a formal response indicating the server rejected your request for a specific reason, and those reasons can be quite varied. Think of it as a detailed, albeit sometimes frustrating, error message directly from the web service itself. The core of debugging these issues, as I've often found, lies in scrutinizing not only your request but also the web service definition you’re interacting with—typically, the WSDL file.

One of the most frequent offenders, in my experience, is a mismatch between the data structure you’re sending in your savon request and the data structure the web service expects. Soap is notoriously strict about data types, element names, and namespaces. I once spent hours troubleshooting a seemingly straightforward request because I had overlooked the fact that an integer field, as defined in the wsdl, was nested inside an extra layer of an object in the request sent to the api. The service, of course, faulted, and rightfully so. You’ve got to pay close attention to the expected data types and hierarchies as described in the WSDL. I’d recommend reviewing the WSDL file manually or through a tool that provides a graphical representation of the schema. Consider resources such as "Web Services Description Language (WSDL) 1.1" by the W3C for a solid understanding of the specification if you’re new to that. It's dry, but fundamental.

Another common error, particularly when working with more complex soap services, involves incorrect namespaces or prefixes within your request. Savon generally handles this well, but discrepancies between your client configuration and server expectations can cause problems. For example, if the wsdl specifies a particular namespace as the default and your savon request uses a different one, even for the same element name, it will likely fail. This can be a subtle error to catch when looking at the raw xml, but a diff of an example from a correct call (if you can get one) versus your request will usually flag this immediately.

Let's consider a few concrete examples using code. These examples illustrate common fault causes and demonstrate how to address them within Savon. I’ll be using a simplified, hypothetical web service for clarity.

**Example 1: Data Type Mismatch**

Let’s say the WSDL specifies a 'quantity' field as an integer, but we are sending it as a string, wrapped in some other object:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl") # Replace with actual wsdl path

begin
  response = client.call(:create_order) do |soap, header|
    soap.body = {
        order_details: {
          quantity: "10", # Incorrect: Should be an integer
          product_id: 123,
        }
    }
  end
  puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```

The web service will likely return a fault since `quantity` is specified as integer in the wsdl. Here's the corrected version:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl")

begin
  response = client.call(:create_order) do |soap, header|
    soap.body = {
        order_details: {
          quantity: 10, # Correct: Sending as an integer
          product_id: 123,
        }
    }
  end
  puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```
The change here involves removing the quotation marks around the value `10`, ensuring it's sent as an integer as specified by the WSDL.

**Example 2: Incorrect Namespace**

Imagine a scenario where the WSDL defines a specific namespace, say `http://example.com/orders`, for order elements, but we're neglecting to use it correctly in our request:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl")

begin
  response = client.call(:create_order) do |soap, header|
    soap.body = {
        order_details: {
          quantity: 10,
          product_id: 123,
          "ns:customer_id": 456 # Incorrect: missing namespace specification for other elements, assuming ns is the correct namespace
        }
    }
  end
  puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```

While we've correctly used the `ns` prefix for the `customer_id`, the other fields should likely also have a namespace if that’s what the WSDL specifies. To resolve this, we need to properly define the namespace for all elements that require it, either through specific prefix application, or using the default namespace option of Savon. The second version is usually cleaner.

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl", namespace: "http://example.com/orders")


begin
  response = client.call(:create_order) do |soap, header|
    soap.body = {
      order_details: {
          quantity: 10,
          product_id: 123,
          customer_id: 456 # Correct: Assumes default namespace applies now to order_details content
      }
    }
  end
    puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```
Here, by specifying `namespace` in the client definition, we avoid repeating the `ns:` prefix. This simplifies the code and ensures consistency with the service requirements.

**Example 3: Missing or Incorrect Header Information**

Often, web services require specific header data, like authentication tokens or routing information. Omitting or incorrectly formatting these can also trigger soap faults. This is often where the `header` block is used in Savon.

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl")

begin
  response = client.call(:create_order) do |soap, header|
    soap.body = {
        order_details: {
          quantity: 10,
          product_id: 123,
          customer_id: 456
        }
    }
  end
    puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```
This request may fault, especially if an authentication token is expected in the soap header. Let’s add a fictional authentication token:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/service.wsdl", namespace: "http://example.com/orders")

begin
  response = client.call(:create_order) do |soap, header|
       header["wsse:Security"] = {
        "wsse:UsernameToken" => {
          "wsse:Username" => "my_username",
          "wsse:Password" => "my_password"
          }
        }
    soap.body = {
        order_details: {
          quantity: 10,
          product_id: 123,
          customer_id: 456
        }
    }
  end
   puts response.body
rescue Savon::SOAPFault => fault
  puts "SOAP Fault: #{fault.to_s}"
end
```

The corrected request now incorporates the `wsse:Security` header, which is a common way for services to authenticate a request, preventing a potential fault due to missing credentials. This also assumes that these credentials are what are expected from the WSDL, and that the namespace prefixes are also correct, so this might require more debugging.

Debugging soap faults frequently requires a step-by-step approach. Start by reviewing the wsdl thoroughly. Then inspect the raw soap request and response using Savon’s logging capability (`log: true` in the client configuration) can help with pinpointing discrepancies. Additionally, the soap fault message itself often provides helpful details about which specific element or attribute triggered the problem. I’ve found that tools like SoapUI or Postman, configured to read the service’s WSDL, can be helpful to validate request structures prior to implementing them into savon clients, providing a 'golden path' to reference against your own requests. Lastly, and this is often overlooked, always check the web service documentation if it is available for examples and guidance on using it's specific implementation of soap. The "Soap 1.2 Specification" from the W3C is also a good resource to have if you work with soap services regularly.

In conclusion, while soap faults can be initially frustrating, careful analysis of the WSDL, attention to data types, namespaces, headers, and thoughtful debugging will usually reveal the underlying cause. It’s a meticulous process, but with practice, you'll be able to navigate these challenges more efficiently.
