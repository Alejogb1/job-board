---
title: "Why am I getting Savon::SOAPFault on request with Savon ruby gem?"
date: "2024-12-23"
id: "why-am-i-getting-savonsoapfault-on-request-with-savon-ruby-gem"
---

Okay, let's tackle this. Savon::SOAPFault errors… ah, yes, I remember a particularly thorny case back when I was working on integrating a legacy enterprise system with our newer platform. Those errors can feel incredibly vague initially, but they usually boil down to a handful of common issues. I've seen it enough times to have a decent handle on debugging them.

The `Savon::SOAPFault` exception in Ruby’s Savon gem fundamentally means the SOAP server you're communicating with has explicitly rejected your request. It’s not a network problem per se, but a problem with the *contents* of your SOAP envelope as understood by the server. Let's dissect the most probable causes and how I've approached them in the past.

First, the most frequent culprit is incorrect soap envelope construction. This means that the XML structure you are sending, which includes the SOAP header and body, doesn't match what the service is expecting. Even small deviations can cause the server to reject the message. When dealing with complex web service definitions (WSDLs), there can be subtle nuances in expected namespaces, attribute order, and data types.

Let's say I was, back in the day, trying to fetch some customer data. Suppose the server requires a specific namespace declaration for its elements, or a specific data type for a customer ID:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/your/service.wsdl") #replace wsdl path with real path to your wsdl

begin
  response = client.call(:get_customer_data) do
    message(
      {
        'GetCustomerData' => {
          'customerID' => 12345, # Assuming customer ID is an integer
        }
      },
       namespaces: { "xmlns:ns1" => "http://example.com/customer/schema" } # Assuming a specific namespace
    )
  end
  puts response.body

rescue Savon::SOAPFault => error
  puts "SOAP Fault Error: #{error.to_s}"
  puts "Fault Code: #{error.fault.faultcode}"
  puts "Fault String: #{error.fault.faultstring}"
  puts "Fault Detail: #{error.fault.detail}" # Check fault detail, it provides important error information
end
```

In this first example, notice the usage of `namespaces:` inside the `message` method, as well as the assumed data type for `customerID`. Without this precise namespace declaration, the server might simply not understand which schema to validate against, leading to a fault. It might be expecting something like `ns1:customerID` and that’s why you would see a fault, because the server doesn't recognize the `customerID` element.

Another common cause is incorrect or missing authentication details. Many SOAP services require specific credentials within the SOAP header, often using WS-Security. These could be username/password pairs or more complex security tokens. If these are absent or incorrect, the server will rightly reject the request.

Here’s a look at what that would look like in code, this time with basic authentication in the SOAP header:

```ruby
require 'savon'
require 'base64'

client = Savon.client(wsdl: "path/to/your/service.wsdl")

begin
    username = 'myusername'
    password = 'mypassword'
    auth_token = "Basic " + Base64.encode64("#{username}:#{password}").strip
    
  response = client.call(:get_customer_data) do
    header "Authorization" => auth_token,
         "Content-Type" => 'text/xml; charset=UTF-8'
    message({
        'GetCustomerData' => {
            'customerID' => 12345
        }
    })

  end
  puts response.body

rescue Savon::SOAPFault => error
  puts "SOAP Fault Error: #{error.to_s}"
  puts "Fault Code: #{error.fault.faultcode}"
  puts "Fault String: #{error.fault.faultstring}"
  puts "Fault Detail: #{error.fault.detail}" # Always examine the fault detail.
end
```

In this snippet, I'm constructing a basic authentication header and explicitly setting a Content-Type. WS-Security can get complex quickly, involving timestamp generation and encryption, so referring to specific documentation (perhaps the WS-Security specification itself) is essential. In real projects, I've had to dig through server logs and Wireshark captures to get authentication headers formatted precisely.

Finally, parameter mismatches are also prime contributors to `Savon::SOAPFault`. Even if the schema structure seems correct, you might be sending the wrong data types or too few/too many parameters for a given operation. Always verify the data types you are providing exactly match the xsd of the WSDL you are using. For instance, the server might expect a string value for an identifier but receive an integer, or a specific date format that’s not respected by your request.

Here’s an example demonstrating how incorrect parameter values can lead to a soap fault:

```ruby
require 'savon'

client = Savon.client(wsdl: "path/to/your/service.wsdl")

begin
  response = client.call(:create_customer) do
    message({
      'CreateCustomer' => {
        'customer_details' => {
           'name' => "John Doe",
           'birthdate' => "1980/01/15", # Incorrect Date format
           'customer_type' => 1 # Incorrect data type for customer type
        }
      }
    })
  end
  puts response.body

rescue Savon::SOAPFault => error
  puts "SOAP Fault Error: #{error.to_s}"
  puts "Fault Code: #{error.fault.faultcode}"
  puts "Fault String: #{error.fault.faultstring}"
  puts "Fault Detail: #{error.fault.detail}"
end
```

In this case, the server expects a specific date string format, like “1980-01-15,” or might require the ‘customer_type’ to be a specific string literal rather than an integer. The fault details typically provides clues about parameter validation issues.

When debugging, I’ve found that the following resources are extremely valuable:

1.  **The WSDL (Web Service Definition Language) document itself:** This is your primary reference. Pay close attention to the data types defined within the XML schema, the required namespaces, and the parameter lists for each operation. Tools like XMLSpy can help navigate and understand WSDL files.
2.  **The SOAP server's documentation:** This may include specifics beyond the WSDL that might not be readily apparent (e.g. specific authentication mechanisms).
3.  **Wireshark:** A network protocol analyzer. Capturing and analyzing the actual SOAP request and response payloads provides concrete evidence of how the request was formulated by Savon and how the server interpreted it. This is invaluable for diagnosing issues.
4.  **“Programming Web Services with SOAP,” by James Snell, Doug Tidwell, and Pavel Kulchenko:** This provides a very clear and complete understanding of SOAP protocol itself and can explain some of the nuances which can cause this type of errors
5.  **The official Savon documentation:** Which, despite being very good, sometimes needs to be paired with good understanding of SOAP itself.

In short, `Savon::SOAPFault` errors often result from minor inconsistencies between your SOAP requests and the server’s expectations. By meticulously inspecting the WSDL, meticulously checking your requests against the expected structure, and making use of the debug information within the `Savon::SOAPFault` error objects, as well as other tools mentioned above, you can almost always uncover the underlying cause. Remember to break down the issue, and don't be afraid to examine the raw XML to spot the source of the error.
