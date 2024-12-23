---
title: "Why is my Chainlink node job failing with an invalid character '_' error during JSON payload interpolation?"
date: "2024-12-23"
id: "why-is-my-chainlink-node-job-failing-with-an-invalid-character--error-during-json-payload-interpolation"
---

Alright, let's get into this. Seeing an "invalid character '_'" error during Chainlink job runs, specifically within JSON payload interpolation, generally points to a common but often overlooked nuance in how Chainlink handles variables within these templates. From my experience, having spent a few years wrestling with various integration challenges in decentralized applications, I've seen this issue pop up more than once, usually when transitioning from initial development to more complex, real-world usage. The underlying problem essentially boils down to misinterpretations or conflicts within the variable substitution process inside the `jsonparse` adapter or when constructing the JSON payload directly.

The core of the issue usually involves attempting to use variables with characters that aren't considered valid identifiers within a JSON context, especially when they appear *within* the string values of your json structures. Chainlink’s JSON processing logic for template interpolation expects these variables to be referenced within the Jinja2 templating system using the `{{variable_name}}` notation, but it’s quite sensitive to characters within variable names themselves and also where you are trying to insert the result of a variable evaluation. Often we will find that the problematic characters are not in the variable names itself, but in the result of that evaluation, which is not escaped to be compatible with json strings. The underscore character '_', while perfectly acceptable in many programming contexts, can become a source of trouble in json payloads specifically when they exist outside the Jinja2 placeholder but are still part of the result.

Here's what I mean. In a typical Chainlink job spec, you're using variables that are sourced either from your adapter results or directly from your contract calls, and these variables must be handled carefully. Let's consider that you're constructing the body of an http POST request as an example.

Let's imagine this specific scenario from a past project: I needed to pull some data from an external API that returned an address with a specific format, that included an underscore, and inject it into a json request. My initial approach was naive, something like this:

```yaml
# Bad example that will lead to error
- type: http
  url: "https://api.example.com/submit"
  method: POST
  body: '{"account": "{{.result.address}}", "other_data": "some value"}'
  headers:
    Content-Type: application/json
  jsonParse: true
  resultPath: "success"
```

Now, the variable being substituted `{{.result.address}}` might, for the sake of argument, resolve to something like `"0x123_456_789"`. The server on the other side, when receives this, will fail, because even if the json parsing is successful, the server will reject the address, because it has underscores, but in the context of this question it’s the chainlink job that will fail in the json interpolation step with an invalid character "_". This would be equivalent to manually trying to create json like so: `{"account": "0x123_456_789", "other_data": "some value"}`. This is an invalid json! Json allows underscores only in the key, but not in the values, unless it's escaped or part of a number.

The fix here is not only to ensure that the variable that is inserted into the json document has no underscores, but that it is also correctly escaped according to json standards, making sure it's properly formatted. We can use a new template. This is how it would look with a preprocessor:

```yaml
# correct example using a preprocessor to escape underscores from address
- type: http
  url: "https://api.example.com/submit"
  method: POST
  body: '{"account": {{.result.escaped_address}}, "other_data": "some value"}'
  headers:
    Content-Type: application/json
  jsonParse: true
  resultPath: "success"
- type: preprocessor
  name: json_escaper
  params:
    resultPath: "escaped_address"
    input: '{{.result.address}}'
    template: '"{{input | string | escape_json}}"'
```

In this corrected version, we introduce a preprocessor of the type template. This preprocessor will take the result of `{{.result.address}}` and escape any json control characters, such as double quotes, backslashes and, yes, underscores, ensuring the result is a valid json string value when inserted.

Let's move to another scenario that I also encountered before. Imagine you’re using a `jsonparse` adapter and, again, expect the incoming JSON from the external API to contain an address like `0xabc_def_ghi`. And say you try to access a nested field with underscores in the keys themselves and fail:

```yaml
# another bad example that will also lead to error
- type: http
  url: "https://api.example.com/data"
  method: GET
  jsonParse: true
  resultPath: "nested_data.address_with_underscore.value"
```

In this case the json structure from the api might have the following structure:

```json
{
  "nested_data": {
    "address_with_underscore": {
       "value": "0xabc_def_ghi"
    }
  }
}
```

The problem here is not that the value contains underscores, but the `resultPath` is trying to access a json key with underscores. `jsonParse` expects the keys to be valid javascript like names, without underscores. The solution is to access the data using a different path, or to create a preprocessor template to rename the keys to remove the underscores before passing to the `jsonParse`. Let’s use an alternative path here:

```yaml
# fixed example using a simpler path to avoid underscore keys.
- type: http
  url: "https://api.example.com/data"
  method: GET
  jsonParse: true
  resultPath: "nested_data.address_with_underscore"
- type: preprocessor
  name: get_address_value
  params:
    resultPath: "address_value"
    input: '{{.result.value}}'
```

In this case, we use a simpler result path to get the structure that has the underscored keys, and then another preprocessor to access the `value` field. The error, in this case, comes from trying to access a key with underscores in the `jsonparse` adapter, it does not come from the data itself containing underscores.

Let’s consider another common issue: trying to build dynamic json with nested fields with variable keys, also using a template for it. Let's say we want to construct a JSON object where the keys themselves are dynamically determined, and these keys *might* include underscores:

```yaml
# bad example with dynamic keys containing underscores
- type: preprocessor
  name: build_dynamic_json
  params:
    resultPath: "dynamic_json"
    input: '{ "prefix_{{.result.key}}": "{{.result.value}}"}'
    template: '"{{input}}"'
```
If the `{{.result.key}}` is resolved to "some_key", the json that is generated will be `{"prefix_some_key":"some_value"}` and everything will work as expected since the underscored key is inside a valid json document. The problem is if the result is not wrapped in quotes, since the resulting json would look like this: `{"prefix_some_key":some_value}`, which will result in a json error.

The fix here again is to correctly escape the variable result.

```yaml
# correct example using dynamic keys with proper json escaping
- type: preprocessor
  name: build_dynamic_json
  params:
    resultPath: "dynamic_json"
    input: '{"prefix_{{ input | string | escape_json }}": "{{.result.value}}" }'
    template: '"{{input}}"'
```

In the fix, we again make use of a `template` type preprocessor. The key here is the addition of the `escape_json` filter. By applying this filter to the `input`, we ensure that any special characters, including underscores that may be present in the value are properly escaped, allowing them to be safely included in the dynamic JSON key.

In practical terms, what you see as an invalid character error can be due to an unescaped value used to create json, the result from variable substitution having an underscore where it shouldn't, or trying to access json keys that are not valid javascript identifiers. Careful use of string escaping and preprocessors to shape data, can provide a powerful way to handle data formatting. It always involves careful inspection of how you use your variables, both in the `resultPath` and the templates, to understand what data is going where and if it requires escaping or any kind of processing.

To deepen your understanding further, I’d recommend exploring "The Definitive Guide to JSON" by David Crockford, for the basics, and the Jinja2 documentation, specifically on templating, filtering, and the details of its syntax. I also highly recommend reading the Chainlink documentation on job specs and adapters, which will explain in greater detail how these pieces interact together. Finally, experimenting with simple jobs to print out the result of intermediate steps will allow you to pinpoint where the problem is by inspecting what values are getting injected where in the job spec. This hands-on approach will be your best tool for debugging and prevent these errors in the future.
