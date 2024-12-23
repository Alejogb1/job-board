---
title: "How can uploads across subdomains be accessed using active storage?"
date: "2024-12-23"
id: "how-can-uploads-across-subdomains-be-accessed-using-active-storage"
---

Alright,  I recall a project from a few years back, where we had a sprawling microservices architecture. Part of that involved users uploading files across various subdomains, and needing a unified way to access them, which of course, we opted to manage using active storage. It wasn't a completely straightforward setup, and it presented some interesting challenges with browser security and cors, so let me walk you through my experience and some solutions.

Fundamentally, the issue stems from the browser's same-origin policy. When an upload happens from `app.example.com`, the browser will block requests to fetch that file from, say, `api.example.com` or `cdn.example.com` without explicit permissions. Active storage itself, as a tool to abstract away file management, handles the saving and processing, but doesn’t inherently solve the cross-domain accessibility problem. You need to configure the webservers appropriately to allow cross-origin resource sharing (cors).

My initial approach, as with many things, was iterative. We started simple, placing all uploads in a single, centralized storage location managed by our main rails application. While manageable initially, this created scalability and performance bottlenecks as the userbase grew. We then transitioned to utilizing subdomain-specific storage buckets within AWS S3, while maintaining our core app to orchestrate access.

The key is in correctly configuring your CORS policies to let the other subdomains access these resources. We ended up opting for a fairly open policy for our cdn subdomain, given that these files were meant to be publicly available, whereas api subdomain access was gated behind authentication.

Here's how I configured the AWS S3 CORS settings for the `cdn.example.com` subdomain, as an example. I'm showing a simplified version to illustrate the core concepts. Please note that you should always evaluate the most secure policy suitable to your requirements.

```xml
<CORSConfiguration>
    <CORSRule>
        <AllowedOrigin>https://cdn.example.com</AllowedOrigin>
        <AllowedMethod>GET</AllowedMethod>
        <MaxAgeSeconds>3000</MaxAgeSeconds>
        <AllowedHeader>*</AllowedHeader>
    </CORSRule>
    <CORSRule>
        <AllowedOrigin>https://app.example.com</AllowedOrigin>
        <AllowedMethod>GET</AllowedMethod>
        <MaxAgeSeconds>3000</MaxAgeSeconds>
        <AllowedHeader>*</AllowedHeader>
    </CORSRule>
      <CORSRule>
        <AllowedOrigin>https://api.example.com</AllowedOrigin>
        <AllowedMethod>GET</AllowedMethod>
         <AllowedHeader>Authorization</AllowedHeader>
        <MaxAgeSeconds>3000</MaxAgeSeconds>
    </CORSRule>
</CORSConfiguration>
```

In this simplified s3 cors configuration, you see three `corrule` elements. Each allows specified domains to access s3 resources. Note the usage of the `<allowedorigin>` tag, where we specify which domains can interact with files hosted in this bucket. The `<allowedmethod>` tag determines which http request method are supported (`get`, `post`, `put`, etc). The `authorization` header is specified as allowed for the api subdomain, as most requests here would require this header to authenticate.

Next up, we had to make sure our rails app, the entry point for our user facing service, would also handle the storage urls generation correctly. Our application primarily resided at `app.example.com`, and we wanted to create a user experience where the user can upload files seamlessly from within our application. The rails application, given it was the authority for generating pre-signed urls for our active storage managed uploads, needed to be aware of the correct bucket to target. We set up our `storage.yml` file to be more dynamic based on the application's environment.

```ruby
# config/storage.yml
amazon:
  service: s3
  access_key_id: <%= ENV['AWS_ACCESS_KEY_ID'] %>
  secret_access_key: <%= ENV['AWS_SECRET_ACCESS_KEY'] %>
  region: <%= ENV['AWS_REGION'] %>
  bucket: <%=  if Rails.env.production?
                    'production-bucket'
               elsif Rails.env.staging?
                    'staging-bucket'
               else
                 'development-bucket'
                end %>
  # In other environments, this could vary depending on the subdomain in use
  # For example you could use subdomains instead of environment:
   # bucket:  <%= if request.subdomain == 'api'
    #                   'api-bucket'
     #           elsif request.subdomain == 'cdn'
      #                 'cdn-bucket'
       #         else
        #             'default-bucket'
         #        end %>
```
This snippet shows the `storage.yml` file, where we define different s3 buckets for different `rails.env` environments. You can also see an commented out section using request subdomains instead of rails environments. This snippet shows how you can use ruby's embedded ruby (erb) template functionality to dynamically determine which bucket to target. This way, regardless of where the user is interacting with your system from, the pre-signed url will point to the appropriate storage bucket.

Moving on, when the user interfaces with the application at `app.example.com`, our rails application generates pre-signed urls. These pre-signed urls permit access to the object on a one off basis, without requiring public access and without making a resource available directly. These pre-signed urls would target, in this case, the `production-bucket` if in production, which is configured to allow the necessary domains to access resources. Here's an example of how you can generate these pre-signed urls with rails:

```ruby
class UploadController < ApplicationController

  def create
    blob = ActiveStorage::Blob.create_before_direct_upload!(
      filename: params[:file].original_filename,
      byte_size: params[:file].size,
      content_type: params[:file].content_type
      )
    render json: {
      signed_id: blob.signed_id,
      direct_upload_url: blob.service_url_for_direct_upload,
      headers: blob.service_headers_for_direct_upload
    }
  end

  def show
      @attachment = ActiveStorage::Attachment.find_by_signed_id(params[:id])
      redirect_to rails_blob_url(@attachment.blob, only_path: true)
  end
end

```
In this `uploadcontroller` example, the `create` action generates a pre-signed url, which is sent back to the user interface for uploading a file. The `show` action handles displaying a file when you have a `signed_id`. This utilizes active storage's `service_url_for_direct_upload` to provide the necessary url for the upload. When redirecting to the url of the attachment itself, rails takes care of setting the headers correctly to allow access across domains.

Now, what if you need more granular control? You might need to introduce a proxy layer, possibly another service acting as an intermediary to further control access to the uploaded files. However, that adds complexity and could be a further point of failure in your infrastructure. The described method has generally worked well for us across multiple production deployments.

For more in-depth understanding of cors, I highly recommend reading the official Mozilla Developer Network documentation on CORS. Furthering the discussion around s3's capabilities, the official aws documentation for s3 should be studied, especially its section on bucket policies and access control lists. Finally, for a solid grasp on active storage within the rails ecosystem, consult the official rails guide documentation specific to active storage.

Remember, the exact configuration will vary based on your setup, particularly if you're not using AWS S3 or if you have additional access control constraints. However, the core principles of understanding the same-origin policy and proper CORS configuration still hold true. The examples provided should provide a starting point for any similar architectural implementations.
