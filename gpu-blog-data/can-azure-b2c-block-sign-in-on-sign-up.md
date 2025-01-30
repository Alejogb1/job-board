---
title: "Can Azure B2C block sign-in on sign-up?"
date: "2025-01-30"
id: "can-azure-b2c-block-sign-in-on-sign-up"
---
Azure B2C does not directly offer a single setting to prevent sign-in concurrently with sign-up.  However, achieving this functionality requires a nuanced understanding of its custom policies and the judicious use of claims transformations and orchestration.  My experience building and maintaining authentication systems for several large-scale applications within the Azure ecosystem has shown that this limitation can be effectively overcome using a multi-step approach leveraging custom policies.


**1.  A Clear Explanation of the Approach**

The core strategy revolves around manipulating claims within the user journey.  The fundamental idea is to create a custom policy where the initial sign-up flow sets a specific claim indicating a "newly registered" status. This claim is then checked at the sign-in stage. If the claim is present, the sign-in attempt is blocked; otherwise, standard sign-in proceeds. This necessitates creating a sophisticated custom policy incorporating multiple orchestration steps and claims transformations.


This is more complex than a simple configuration change because Azure B2C is inherently designed for both registration and sign-in. Its default policy assumes a user might alternate between the two actions.  Thus, preventing simultaneous access requires extending the default flow and implementing a conditional logic based on newly created user attributes.  The process involves careful modification of the self-asserted and relying party claims, guaranteeing that the newly registered status is correctly propagated and assessed.


Failure to implement this rigorously can lead to security vulnerabilities. For example, without strict claim management, a malicious actor could potentially bypass the blocking mechanism by manipulating the claim values.  Therefore, careful attention to data validation and claim transformations is paramount.


**2. Code Examples with Commentary**

The following examples illustrate crucial snippets within a custom policy XML configuration.  Note that these are simplified excerpts and require integration within a complete policy structure.


**Example 1: Setting the "Newly Registered" Claim in the Sign-up Flow**

This snippet shows how to add a claim to the user's profile immediately after successful registration. We'll use the claim `isNewlyRegistered` set to `true`.

```xml
<ClaimsTransformation>
  <TechnicalProfile Id="AddNewlyRegisteredClaim">
    <OutputClaims>
      <OutputClaim ClaimTypeReferenceId="isNewlyRegistered" PartnerClaimType="isNewlyRegistered" />
    </OutputClaims>
    <ClaimsTransformationLogic>
      <Script>
        <![CDATA[
          // Assign the value "true" to the isNewlyRegistered claim.
          context.identityData.isNewlyRegistered = true;
        ]]>
      </Script>
    </ClaimsTransformationLogic>
  </TechnicalProfile>
</ClaimsTransformation>

<!-- ... subsequent steps to orchestrate calling this Technical Profile after successful user creation ... -->
```

This `TechnicalProfile` utilizes a JavaScript script to directly manipulate the claims.  Its execution is crucial; it must be called *after* user creation within the sign-up orchestration.


**Example 2: Checking the "Newly Registered" Claim During Sign-in**

This snippet demonstrates how to intercept sign-in attempts and deny access if the `isNewlyRegistered` claim is `true`.

```xml
<ClaimsTransformation>
  <TechnicalProfile Id="CheckNewlyRegisteredClaim">
    <InputClaims>
      <InputClaim ClaimTypeReferenceId="isNewlyRegistered" PartnerClaimType="isNewlyRegistered"/>
    </InputClaims>
    <ClaimsTransformationLogic>
      <Script>
        <![CDATA[
          if (context.identityData.isNewlyRegistered === 'true') {
            // Block sign-in.  This may require setting a specific error code.
            context.error = {
              code: 'newlyRegistered',
              message: 'Please complete your profile before signing in.'
            };
          }
        ]]>
      </Script>
    </ClaimsTransformationLogic>
  </TechnicalProfile>
</ClaimsTransformation>

<!-- ... place this Technical Profile before the actual authentication step in the sign-in flow ... -->
```

This `TechnicalProfile` checks the value of the `isNewlyRegistered` claim. If true, it sets an error, effectively blocking the sign-in attempt. A custom error message is provided for improved user experience.


**Example 3:  Orchestration using Orchestration Steps**

This highlights how to sequence the above transformations within the overall custom policy.

```xml
<OrchestrationStep Order="1" Type="ClaimsTransformation">
  <ClaimsTransformation Id="AddNewlyRegisteredClaim" />
</OrchestrationStep>

<OrchestrationStep Order="2" Type="SelfAsserted-LocalAccountSignUp">
  <!-- ... standard sign-up orchestration steps ... -->
</OrchestrationStep>

<!-- ...In the Sign-in policy -->

<OrchestrationStep Order="1" Type="ClaimsTransformation">
  <ClaimsTransformation Id="CheckNewlyRegisteredClaim" />
</OrchestrationStep>

<OrchestrationStep Order="2" Type="SelfAsserted-LocalAccountSignIn">
  <!-- ... standard sign-in orchestration steps ... -->
</OrchestrationStep>

```

This illustrates how the `AddNewlyRegisteredClaim` transformation is placed after the successful user creation during sign-up, and `CheckNewlyRegisteredClaim` is placed before the authentication step in the sign-in policy. This precise ordering is crucial for the correct functioning of the mechanism.


**3. Resource Recommendations**

For further details on building custom policies in Azure B2C, consult the official Microsoft documentation. Focus on the sections covering claims transformations, orchestration steps, and error handling.  Pay close attention to security best practices related to custom policy development. Additionally, review examples of advanced custom policies provided in the official documentation; studying these will provide valuable context for implementing more complex scenarios. Familiarize yourself with the various claim types available within Azure B2C, particularly those related to user attributes and authentication status.  Finally, rigorous testing is crucial; test your custom policy thoroughly with various scenarios to ensure it functions as expected and does not introduce vulnerabilities.  Understanding JavaScript within the context of Azure B2C claims transformation scripts is also essential.
