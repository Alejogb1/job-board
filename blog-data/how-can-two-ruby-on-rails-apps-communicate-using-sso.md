---
title: "How can two Ruby on Rails apps communicate using SSO?"
date: "2024-12-23"
id: "how-can-two-ruby-on-rails-apps-communicate-using-sso"
---

, let's unpack this. I recall a rather involved project back in '16, where we had to integrate two separate Rails applications – a customer-facing portal and an internal admin system. Both needed shared user sessions, meaning, in essence, single sign-on (SSO). The naive approach would have been to just share the user database, but that quickly gets messy and defeats the purpose of having two distinct applications. So, we opted for a more robust and flexible approach, leveraging an identity provider.

Now, when talking about SSO in this context, we’re essentially dealing with the process where a user authenticates once and can then access multiple applications without re-authenticating. The core idea is to delegate authentication to a central service, often called an identity provider (IdP), rather than each application handling it independently. There are several industry-standard protocols for accomplishing this, but in our case, and given the nature of many Rails environments, we opted for the Security Assertion Markup Language (SAML) protocol, specifically SAML 2.0, given its maturity and widespread support.

The essence of SAML is that the user first requests access to a protected resource on a service provider (SP), which in our case are the Rails apps. If not already authenticated, the SP redirects the user to the IdP. Upon successful authentication at the IdP, an authentication assertion, signed by the IdP, is sent back to the SP. The SP verifies the signature and the assertion, then establishes a local session for the user.

Let's outline the steps a bit more granularly, specifically using SAML 2.0, within the framework of two Rails applications.

1.  **Setup**: Each Rails app (the customer portal and the admin system) needs to be configured as a service provider (SP) with a SAML client library. For Rails, `ruby-saml` gem tends to be a reliable choice. On the IdP side, you'll configure these as registered service providers. We’ll use Okta in my fictional experience as the IdP, given its solid support and developer-friendly nature, although other IdPs like Keycloak or Auth0 could fit as well.

2.  **Authentication Request**: When a user tries to access a protected resource in one of your Rails apps (say, the customer portal), the app acts as an SP and checks if the user has an active session. If not, the app generates a SAML authentication request and redirects the user to the IdP (Okta in our case).

3.  **IdP Authentication**: The user lands at the Okta login page. After successful authentication with Okta, Okta sends a SAML assertion back to the Rails app’s assertion consumer service endpoint (ACS).

4.  **Assertion Processing**: The Rails app receives the SAML assertion. It verifies the digital signature, ensuring the message originated from the expected IdP. It also validates the assertion’s timestamps, audience restriction, and subject identifier (the user identifier). Finally, the application creates a local session for the user.

5.  **Access Granted**: With a valid local session established, the user can now access the initially requested protected resource. The other Rails application will go through a similar authentication workflow, leveraging Okta as its central authority and SSO is effectively working across both apps.

Here are the practical examples, showing the pertinent parts in Ruby using the `ruby-saml` gem:

**Code Snippet 1: Generating a SAML Authentication Request in a Rails Controller**

```ruby
  require 'ruby-saml'

  class SessionsController < ApplicationController
    def new
      settings = {
        assertion_consumer_service_url: "https://your_portal.example.com/saml/acs", #ACS url of the portal
        sp_entity_id: "https://your_portal.example.com",
        idp_sso_target_url: "https://your_okta_domain.okta.com/app/some_id/sso/saml", #Okta url
        idp_cert_fingerprint: "some_fingerprint",
        name_identifier_format: "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified"
      }

      request = OneLogin::RubySaml::Authrequest.new(settings)
      redirect_to request.create
    end
  end
```

In this example, you're setting up the `settings` hash with crucial SAML parameters. The `assertion_consumer_service_url` points to where Okta will send the assertion after authentication. The `sp_entity_id` uniquely identifies the service provider, and the `idp_sso_target_url` specifies Okta’s SSO endpoint. The `idp_cert_fingerprint` is critical for signature verification, and `name_identifier_format` specifies how the user identifier is formed in the SAML assertion. We're also leveraging `ruby-saml` to create an authentication request and redirect the user.

**Code Snippet 2: Handling a SAML Assertion in a Rails Controller (Assertion Consumer Service)**

```ruby
 class SamlController < ApplicationController
    def acs
      settings = {
        assertion_consumer_service_url: "https://your_portal.example.com/saml/acs",
        sp_entity_id: "https://your_portal.example.com",
        idp_sso_target_url: "https://your_okta_domain.okta.com/app/some_id/sso/saml",
        idp_cert_fingerprint: "some_fingerprint",
        name_identifier_format: "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified"
      }

      response = OneLogin::RubySaml::Response.new(params[:SAMLResponse], settings: settings)
       if response.is_valid?
         session[:user_id] = response.nameid
         redirect_to root_path, notice: "Successfully logged in"
       else
         render plain: "Invalid SAML Assertion", status: :unauthorized
       end
     end
  end
```

Here we receive the SAML assertion (`params[:SAMLResponse]`) from Okta and instantiate a `OneLogin::RubySaml::Response` to process it, validating the signature, timestamp, and other checks. If the assertion is valid, we retrieve the `nameid` (user identifier) and store it in a session to establish the user session. It’s essential to have robust error handling here to deal with invalid assertions.

**Code Snippet 3: Ensuring Authentication is Required by a Rails Controller**

```ruby
class ApplicationController < ActionController::Base
   before_action :authenticate_user!

   private

   def authenticate_user!
     unless session[:user_id]
      redirect_to new_session_path, alert: "Please log in"
    end
  end

  def current_user
     @current_user ||= User.find_by(uid: session[:user_id])
  end
end
```

This controller sets the baseline. Before any action on most controllers, the `authenticate_user!` method checks if a session exists. If not, the user is redirected to the login path, which will start the SAML flow if required. Furthermore, the `current_user` provides a convenient way to load the currently logged-in user based on the `user_id` stored in the session.

Key to succeeding with this setup is making the necessary configuration on the Okta side, setting up a SAML integration for each of the Rails applications with their respective metadata, ensuring the correct certificate and fingerprint is in place. Moreover, metadata exchange between SP and IdP is usually needed in initial configuration process. Each service provider (Rails app) will have metadata XML file that needs to be uploaded to IdP. IdP will also generate a metadata file, which might be needed in SP configuration. It’s a crucial part for establishing communication between IdP and SP.

For further reading on SAML 2.0, I'd recommend consulting the official OASIS specifications (look for “OASIS SAML specification”). Additionally, "Understanding SAML" by Paul Madsen is a comprehensive guide. For a deeper understanding of authentication in web applications, check out "Web Security: A Step-by-Step Approach" by Michael Howard and David LeBlanc. For specific guidance on Ruby on Rails security, the Rails Security Guide, which is part of the official Rails documentation, is invaluable. These will provide a solid foundation for understanding the subtleties and nuances involved in achieving secure and seamless SSO across different applications. This is the approach we used, and it proved stable and scalable, keeping the security requirements of the project very much in mind.
