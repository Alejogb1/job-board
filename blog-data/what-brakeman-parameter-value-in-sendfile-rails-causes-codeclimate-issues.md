---
title: "What Brakeman parameter value in send_file Rails causes CodeClimate issues?"
date: "2024-12-23"
id: "what-brakeman-parameter-value-in-sendfile-rails-causes-codeclimate-issues"
---

Alright, let's unpack this. It's been a while since I've actively worked with a Rails monolith that uses `send_file` so extensively it started triggering CodeClimate alerts related to Brakeman, but the memory’s still there. The specific issue we're likely hitting revolves around the `disposition` parameter when it's used improperly. I recall a particularly hairy incident back in '17 at a startup where a carelessly handled download feature nearly exposed sensitive data through a misconfigured header. It was a learning experience, to say the least.

The `send_file` method in Rails is generally used to stream file contents as a response. The `disposition` parameter dictates how a browser should handle the file. If left unspecified, the default, typically ‘inline,’ attempts to display the file within the browser if possible. However, when dealing with non-viewable files, or when the user explicitly needs to download the file, setting the `disposition` to `attachment` is the standard practice. The problem isn’t necessarily using `attachment`, it’s what happens when you couple it with user-supplied parameters, making it vulnerable.

Brakeman flags this scenario because, without careful sanitization, an attacker could manipulate the filename included in the Content-Disposition header through the `filename` option of the `disposition` parameter. This isn't a direct code execution risk. Instead, it's more related to potential information disclosure and, in specific scenarios, to denial of service if the generated filename is long or contains special characters which could confuse the browser. Consider that most browser implementations interpret filenames with a limited set of characters so anything outside that set may have unintended consequences.

Let’s look at some code examples. First, a typical example that *doesn't* cause issues:

```ruby
def download_report
  report_path = Rails.root.join('tmp', 'generated_report.pdf')
  send_file(report_path,
            filename: 'report.pdf',
            type: 'application/pdf',
            disposition: 'attachment')
end
```

This snippet downloads `generated_report.pdf` as a file named `report.pdf`. It’s simple and safe because `report.pdf` is hardcoded. Brakeman isn't going to complain about this one because there's nothing dynamically injected by a user. Now, let's introduce the problematic scenario that triggers alerts.

```ruby
def download_user_file
  user_file = params[:filename]
  file_path = Rails.root.join('uploads', user_file) # assuming files are uploaded into `/uploads`
  send_file(file_path,
            filename: user_file,
            disposition: 'attachment')
end
```

This second example takes the filename directly from user input. An attacker could send a request where `params[:filename]` contains something like: `../../../../etc/passwd`. Even though the file system access may fail because Rails is accessing the file through `Rails.root`, which is usually confined to the application's root, the risk is in the header. Although the browser will eventually be forced to download 'passwd' which will most likely not be readable, the attacker gains knowledge of what file exists on that file system. A better attacker could also use that knowledge to try other attacks. Furthermore, an attacker can insert special characters into the filename causing a denial of service on the client by injecting an excessively long file name.

To mitigate this, we need to sanitize the input:

```ruby
def download_user_file_safe
  user_file = params[:filename].to_s.gsub(/[^a-zA-Z0-9._-]/, '') # sanitize input!
  file_path = Rails.root.join('uploads', user_file) # assuming files are uploaded into `/uploads`

    if File.exist?(file_path)
      send_file(file_path,
                filename: user_file,
                disposition: 'attachment')
    else
      render plain: "File not found", status: :not_found
    end
end
```

Here, we use a regular expression to remove any characters that aren’t alphanumeric, periods, underscores, or hyphens. While there are more complex sanitization strategies, this should eliminate most problematic cases. Also, note the addition of `File.exist?` to check whether the file exists before attempting the `send_file` call. That can help preventing some errors.

This highlights the importance of always validating user input, but also of making sure the application responds appropriately, with an error, when a resource is not available.

The core takeaway from the Brakeman warning isn’t just to avoid using `params` directly in the `filename` option. It's about understanding the potential for manipulation through dynamically generated headers. The browser doesn't validate content types based on the extension of the file name, it relies on what's sent in the content type header. That's why we have to manually set it when we `send_file`. It’s crucial to understand that we are interacting with two types of potentially vulnerable parameters here: the filename that goes into the header and the file path used to read the file from disk. We need to address both.

For further reading on secure file handling practices and web application security in general, I'd recommend starting with the following:

*   **"The Web Application Hacker's Handbook: Finding and Exploiting Security Flaws" by Dafydd Stuttard and Marcus Pinto:** This book provides a comprehensive guide to understanding web vulnerabilities, including header manipulation and file handling issues, and how to exploit them. It’s not specific to Rails but fundamental concepts it exposes still apply to any web framework.
*   **OWASP (Open Web Application Security Project) documentation:** Specifically, the section on input validation and encoding. OWASP provides resources and guides for secure coding practices. The Cheat Sheet Series is a good starting point, especially the Input Validation Cheat Sheet.
*   **"Secure Programming with Static Analysis" by Brian Chess and Jacob West:** This one delves into static analysis tools and how to use them effectively to find vulnerabilities, which is the very problem Brakeman is trying to address in Rails applications. Although it’s not specific to web applications, it covers the static analysis concepts that apply to Brakeman.

Understanding that Brakeman is trying to highlight security concerns, not just stylistic problems, is key. The `send_file` method is very useful, but as with any feature that touches the file system and headers, it needs to be handled cautiously with a strong focus on sanitization. I hope this helps clarify the issue. Let me know if you have more questions.
