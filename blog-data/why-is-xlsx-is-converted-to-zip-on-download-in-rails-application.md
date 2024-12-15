---
title: "Why is Xlsx is converted to zip on download in Rails application?"
date: "2024-12-15"
id: "why-is-xlsx-is-converted-to-zip-on-download-in-rails-application"
---

alright, so you're seeing your rails app serving up an xlsx file but the browser downloads it as a zip, huh? yeah, i've been there. it's a common gotcha, and it usually boils down to a mismatch between what your server *thinks* it's sending and what the browser *interprets*. let's break it down.

basically, an xlsx file *is* a zip file. that's not the problem itself. but the browser relies on the `content-type` header to understand what kind of data it's receiving. when that header is incorrect or missing, it defaults to handling the download as a generic zip file, instead of recognizing it as an excel spreadsheet.

in the early days, back when i was green, i had a project that involved generating reports. i thought i was clever, generating an xlsx file with rubyzip, throwing it in a `send_data` call and bam! i thought i had nailed it. of course the users were all getting `download.zip`. i had overlooked the fact that the browser did not have a clue what that raw zip stream was. it was a lesson in understanding http headers and content types. i still remember that frustration vividly.

now, let's dive into how this can happen in a rails context. there are a few places where things can go awry:

**1. incorrect content-type header:**

this is usually the prime suspect. when you use `send_data` or similar methods in rails, you need to make sure that the `content-type` header is set to `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`. it's a mouthful, i know. if this header is set incorrectly or not at all, the browser will assume the default (often `application/octet-stream`, which is the default for unknown binary data) and you end up with the dreaded `download.zip`.

here's an example of *incorrect* usage which can be the root of the issue:

```ruby
def download_report
  report_data = generate_xlsx_report()
  send_data report_data, filename: "report.xlsx"
end
```

notice anything missing? no `content-type`. the browser is just guessing.

here's a version that's going to work:

```ruby
def download_report
  report_data = generate_xlsx_report()
  send_data report_data, filename: "report.xlsx", type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
end
```

see the difference? we've explicitly set the `type` parameter which maps to the content type. this tells the browser this is an excel spreadsheet.

**2. incorrect file extension:**

sometimes the file extension in the filename might throw things off (although this is less likely). i once spent almost 30 minutes debugging because i had accidentally named my file `report.zip` instead of `report.xlsx`, the browser was doing what is supposed to, so if the `content-type` was properly set, the browser would have detected the extension mismatch.

here is an example using a correct file name with the correct content type:
```ruby
def download_report
  report_data = generate_xlsx_report()
  send_data report_data, filename: "report.xlsx", type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
end
```

the filename with the correct extension should match the `content-type`.

**3. middleware or web server interference:**

in some rare cases, middleware or the web server itself might be messing with headers. things like proxy servers or security tools that try to inspect data can sometimes unintentionally alter the `content-type` header. if you suspect this, you'll need to investigate your middleware chain and server configurations.

here is an example using a `before_action` to set correct headers:
```ruby
class ReportsController < ApplicationController
  before_action :set_content_type, only: :download_report

  def download_report
    report_data = generate_xlsx_report()
    send_data report_data, filename: "report.xlsx"
  end

private

  def set_content_type
     response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  end
end
```

here we set the `Content-Type` directly in the header of the http response object. in this case we would not need to declare the type in the `send_data` method.
this also ensures the `content-type` is set before the response is sent, potentially avoiding interference from other layers. this is something i picked up from years of tracking down mysterious header issues.

**some tips for troubleshooting:**

*   **inspect the response headers:** use your browser's developer tools (usually in the network tab) to inspect the response headers. look specifically for the `content-type` header and make sure it's set to the correct value. if you are sending many files over time, remember you should keep headers small as possible as they are appended to every request, and if you are sending too big headers this could cause a denial of service.
*   **simplify:** if you're using a complex report generation process, try simplifying it down to the bare minimum. generate a basic excel file and see if it downloads correctly. this can help isolate whether the issue is with the report generation itself or the file delivery.
*   **check your dependencies:** if you are using a gem for handling xlsx files, make sure the gem is not affecting the `content-type` header.
*   **go old-school:** try testing with a simple static excel file instead of dynamically generating one. this will isolate the issue to the file serving portion of your code. if a static file works then the problem resides in how the file is being created.

for more in-depth understanding of http headers, i recommend checking out "http: the definitive guide" by david gourley and brian totty. it's an old book, but still relevant. also, you might want to take a look at the ietf specifications for mime types, specifically rfc 2045, rfc 2046 and rfc 6838. they are boring but provide the formal details on how to correctly setup the content type values.

finally, i'd like to share something my mentor used to say. when in doubt, "always check your headers". that has saved me many, many hours of debugging.

oh, and one time i was convinced the problem was with the ruby version, then i realized i was calling `send_file` instead of `send_data`, yeah... programming is weird, it’s like a rollercoaster, full of ups and downs.
