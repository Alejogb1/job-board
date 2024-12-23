---
title: "How can I disable username/password authentication in Keycloak?"
date: "2024-12-23"
id: "how-can-i-disable-usernamepassword-authentication-in-keycloak"
---

Alright, let's tackle this. Disabling username/password authentication in Keycloak isn't a single toggle; it requires a careful understanding of the authentication flows and how Keycloak handles identity brokering. In my experience, I've seen this requested primarily in two contexts: either migrating to a solely federated authentication model or simplifying a development/testing environment. Neither is trivial, and both require a solid plan to prevent inadvertently locking out users, especially in production scenarios.

My journey with this started during a rather complex migration project for a financial services application. We were moving away from local database user management to Azure AD for all authentications. The initial attempt at simply disabling the username/password login form directly resulted in a frustrating lockout situation, underscoring the importance of understanding the underlying authentication mechanisms. Therefore, let’s walk through how to accomplish this while maintaining a functional system.

The crux of the matter is that Keycloak utilizes authentication flows, which are configurable sequences of authentication steps. These flows determine how a user proves their identity. For username/password authentication, there's usually a ‘browser flow’ with ‘Username Password Form’ as one of the steps. To effectively disable username/password authentication, you have to either modify this flow or configure alternative flows that bypass this step altogether. Simply turning off the ‘Username Password’ authenticator, for example, might cause unintended consequences if that step is a prerequisite for other authentication methods.

Let’s break down the common strategies.

**Strategy 1: Modifying the Browser Flow (Less Recommended for Full Disable)**

This is the path of least resistance but not the most robust for a complete disabling. Within the Keycloak administration console, navigate to *Authentication* -> *Flows*. Select *browser flow*. You will see a sequence of actions. You might find the *Username Password Form* under the 'Browser Forms' provider. While you could *remove* this, it's a brittle approach because other flows might rely on it. More importantly, if no other authentication option is present, users simply will not be able to log in. A much more sound approach here would be to configure alternative authentication paths like an Identity Provider and configure the flows such that the username password step is skipped when a user logs in through an IdP.

**Example Code Snippet (Keycloak Admin CLI - `kcadm`) to retrieve an existing browser flow and make changes (illustrative and not complete removal of username/password):**

```bash
# Assumes you're authenticated with `kcadm config credentials`

# Get the browser flow id
browser_flow_id=$(kcadm get flows -r myrealm --fields id --query "alias=browser" | jq -r '.[].id')

# Get the execution for the username password form in that flow
username_password_execution_id=$(kcadm get flow-executions -r myrealm --flow ${browser_flow_id} --fields id --query "providerId=auth-username-password-form" | jq -r '.[].id')

# Disable the username password form execution - ONLY if another authentication method is present. This approach can backfire.
kcadm update flow-execution -r myrealm ${username_password_execution_id} -o disabled=true

# Verify the updated execution
kcadm get flow-executions -r myrealm ${browser_flow_id}  --query "providerId=auth-username-password-form"
```
*Note:  This cli example requires `jq` to parse the JSON outputs and you might need to use different realm names or ids in the command. Disabling the execution is not a full removal and it can be easily reverted. This highlights why simply modifying a built-in flow is often insufficient.*

**Strategy 2: Utilizing Alternative Flows and Identity Providers (Preferred Approach)**

The more reliable strategy is to configure an *Identity Provider* (IdP) and modify flows to bypass the username/password form if the user authenticates through the IdP. This approach provides a clear path and ensures that you are not entirely dependent on username/password authentication. Keycloak provides pre-built support for numerous IdPs like Google, GitHub, Azure AD, etc. You can configure one or more IdPs and then configure your *Browser* flow to utilize it.

Here’s the process:

1.  **Configure an IdP:** In the Keycloak Admin console, navigate to *Identity Providers*. Add the provider that you need, for example, *OpenID Connect v1.0*. Keycloak will ask you for details about your IdP configuration - Client ID, Client Secret, endpoints, etc. Ensure you obtain these details from your IdP.
2.  **Adjust Browser Flow:** Edit your *browser flow* and add the *Identity Provider Redirector* as the *first* authenticator. Configure it so that it is set to trigger for an unspecified provider. Alternatively, you can create a separate flow (e.g., ‘IdP Browser Flow’) and set it as the default browser flow for your realm.
3.  **Set Default Authentication:** In your realm settings, under the *Authentication* tab, set this new browser flow (if you created one) as the *Browser Flow* or make sure that your browser flow has a conditional step that is met when the user attempts to log in.

**Example Code Snippet (Keycloak Admin CLI - Illustrative, assumes Identity Provider "MyAzureAD" is already configured):**

```bash
#Assumes you have an existing Azure AD idp with alias 'MyAzureAD'
# Get the browser flow id
browser_flow_id=$(kcadm get flows -r myrealm --fields id --query "alias=browser" | jq -r '.[].id')

#get the provider id for 'Identity Provider Redirector' (find the id using `kcadm get authenticators -r myrealm` and pick the correct one). Assuming it's 'identity-provider-redirector'
redirector_provider_id="identity-provider-redirector"

# Create a new execution for the identity provider redirector. Make it a top-level requirement and first step.
kcadm create flow-execution -r myrealm  --flow ${browser_flow_id} --provider ${redirector_provider_id} --priority 1 --requirement REQUIRED  --alias "IDP Redirector"
```

This setup ensures that users are redirected to the configured IdP upon initiating authentication. This effectively bypasses the username/password form for users who use that specific IdP, creating an indirect, conditional disabling of the username/password authentication for some, but not all users, as some users might still use local account authentication if they do not have the IdP configured.

**Strategy 3: Dedicated Flows for Specific Scenarios (Recommended for Complete Disable)**

For complete disabling of username/password, particularly useful in pure federated environments or specific test instances, create a dedicated authentication flow. This will give you a clear separation, making sure there's no accidental overlap with other authentication methods.

1.  **Create New Flow:** Under *Authentication* -> *Flows*, create a new flow, e.g., 'Federated Flow'.
2.  **Add IdP Redirector:** Make the *Identity Provider Redirector* the only authenticator in this flow. Configure it to handle all IdPs, or a specific one.
3.  **Set as Default:** Under *Authentication*, set the *Browser Flow* for the realm (or the client if you need client-specific auth) to this new flow. Ensure that no other clients use the default browser flow that might have password-based authentication enabled.

**Example Code Snippet (Keycloak Admin CLI - Illustrative, creates new flow and sets for the realm):**

```bash
# Assuming provider id for identity provider redirector is 'identity-provider-redirector'
redirector_provider_id="identity-provider-redirector"
# Create a new flow 'federated-flow'
kcadm create flows -r myrealm --alias 'federated-flow'

# Get the flow id
federated_flow_id=$(kcadm get flows -r myrealm --fields id --query "alias=federated-flow" | jq -r '.[].id')


# Add the IdP redirector execution
kcadm create flow-execution -r myrealm  --flow ${federated_flow_id} --provider ${redirector_provider_id} --priority 1 --requirement REQUIRED  --alias "IDP Redirector"

# Set this flow as the default browser flow for the realm. This will prevent password-based login for ALL users of this realm
kcadm update realm -r myrealm --browserFlow ${federated_flow_id}
```

By configuring this new flow as default, all authentication attempts will redirect to the configured IdP immediately. This approach achieves a complete disabling of username/password login.

**Important Notes**

*   Always test changes thoroughly in a non-production environment.
*   Have a backup plan to regain access if you inadvertently misconfigure settings.
*   Use an alternative admin authentication method if you need to make changes.

**Recommended Resources:**

*   **The official Keycloak documentation:** It is the definitive guide for all aspects of the system. Specifically, the sections on authentication flows and identity brokering.
*   **“Keycloak: Identity Management for Modern Applications” by Pedro Igor Silva:** A great starting point that provides a practical overview of Keycloak's core concepts.
*   **RFC 6749 (OAuth 2.0):** Essential for understanding the fundamental flows involved in identity brokering and authentication.

In conclusion, disabling username/password authentication in Keycloak requires more than just a simple checkbox. It requires a careful understanding of the authentication flows, an informed configuration of your IdPs, and a planned rollout. By taking a structured approach, using new flows, and thorough testing, you can successfully disable this authentication mechanism and migrate to a more modern approach. Remember that direct modifications to core flows can be risky and a dedicated flow strategy offers both safety and clarity in the long run.
