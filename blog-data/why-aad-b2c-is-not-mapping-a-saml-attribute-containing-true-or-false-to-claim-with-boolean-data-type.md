---
title: "Why AAD B2C is not Mapping a SAML Attribute containing True or False to Claim with Boolean Data Type?"
date: "2024-12-15"
id: "why-aad-b2c-is-not-mapping-a-saml-attribute-containing-true-or-false-to-claim-with-boolean-data-type"
---

alright, so, you’re hitting that classic snag with azure ad b2c and saml attribute mapping to boolean claims, and yeah, it's a head-scratcher initially. i've been down this rabbit hole more times than i care to count, and it usually boils down to a few common culprits. let's unpack it, based on my own painful learnings and lots of late-night debugging sessions.

first off, the core problem isn't really that b2c *can't* map a 'true' or 'false' value to a boolean claim; it's more about how b2c interprets what it receives from the saml identity provider (idp) and what it expects for a boolean data type on the claim. by default, b2c treats all incoming saml attributes as string values. so, if your saml idp sends 'true' as a string, b2c simply sees "t", "r", "u", "e" as a string of characters. it doesn't inherently know that string should be interpreted as a boolean. this is the fundamental impedance mismatch we’re dealing with.

i recall a project, maybe around 2018 or 2019, where we were integrating with a third-party identity provider for a large e-commerce platform. the idp kept sending an `is_premium_user` attribute in the saml assertion, which would always be either "true" or "false". we figured, piece of cake, just map that to a boolean claim in b2c and be done with it. but, no dice. the claim always ended up as a string in our application, and it caused all sorts of logic errors down the line. we spent a couple of days pulling our hair out trying to figure out what went wrong, till it became clear that b2c isn’t doing the implicit type conversion we were expecting. it felt like banging my head against a wall made out of very technical documentation.

now, let's talk practical solutions. the simplest one is using a claims transformation policy in b2c. this policy will explicitly transform the incoming string value to a boolean value during the user journey. you'll be creating a policy that looks for the incoming string claim value and based on its value of 'true' or 'false', or whatever variations your idp sends, then converts it to true or false boolean data type. here’s how you’d approach that:

```xml
<ClaimsTransformation Id="ConvertStringToBoolean" TransformationMethod="ConvertClaimToBoolean">
    <InputClaims>
        <InputClaim ClaimTypeReferenceId="is_premium_user" TransformationClaimType="inputClaim" />
    </InputClaims>
    <OutputClaims>
        <OutputClaim ClaimTypeReferenceId="is_premium_user_boolean" TransformationClaimType="outputClaim" />
    </OutputClaims>
</ClaimsTransformation>

<ClaimsSchema>
  <ClaimType Id="is_premium_user_boolean">
    <DataType>boolean</DataType>
    <DefaultPartnerClaimTypes>
      <DefaultPartnerClaimType>is_premium_user</DefaultPartnerClaimType>
    </DefaultPartnerClaimTypes>
    <UserHelpText>Premium user flag</UserHelpText>
  </ClaimType>
</ClaimsSchema>
```

this policy snippet does a couple of things: first, it defines a `claimstransformation` that takes the input claim `is_premium_user` (the string value from your saml assertion) and converts it to a boolean claim called `is_premium_user_boolean`, which is also declared as a schema claim with boolean type. that is the new claim to be mapped to your application claims configuration. b2c knows to treat `is_premium_user_boolean` as a true boolean, which is crucial for downstream consumption. remember to insert this claims transformation call in a suitable orchestration step within your user journey.

sometimes the issue isn't a straight 'true' or 'false'. your idp might send '1' and '0', or even custom strings like 'yes' and 'no'. in those cases, you need a slightly different transformation:

```xml
<ClaimsTransformation Id="ConvertCustomStringToBoolean" TransformationMethod="ConvertClaimToBoolean">
    <InputClaims>
        <InputClaim ClaimTypeReferenceId="user_status" TransformationClaimType="inputClaim" />
    </InputClaims>
    <OutputClaims>
        <OutputClaim ClaimTypeReferenceId="user_active_boolean" TransformationClaimType="outputClaim" />
    </OutputClaims>
	<InputParameters>
        <InputParameter Id="trueValue" DataType="string" Value="active" />
		<InputParameter Id="falseValue" DataType="string" Value="inactive" />
    </InputParameters>
</ClaimsTransformation>

<ClaimsSchema>
  <ClaimType Id="user_active_boolean">
    <DataType>boolean</DataType>
	<DefaultPartnerClaimTypes>
      <DefaultPartnerClaimType>user_status</DefaultPartnerClaimType>
    </DefaultPartnerClaimTypes>
	<UserHelpText>User status flag</UserHelpText>
  </ClaimType>
</ClaimsSchema>

```

in this case, the `inputparameter` allows you to define the input strings, `active` and `inactive`. if the input claim called `user_status` is equal to `active`, it'll map to a true boolean value, otherwise false, and map to the claim called `user_active_boolean`. this gives a much more robust mechanism for dealing with non-standard true/false representation.

and yeah, i did once encounter a system where the values were literally "yep" and "nope". thankfully, the above strategy is flexible enough to handle these bizarre cases.

another layer to this whole thing is how the saml provider is configured. the idp's saml metadata needs to correctly declare the `datatype` of the attribute being sent. some idps have poor or missing configuration in the saml metadata. that's another place where errors can creep in. always triple check both your saml metadata and your b2c policies for any discrepancies.

also, keep in mind that during development, you may want to inspect the actual saml response to understand what is being sent from the idp. a tool like saml tracer or the browser's network dev tools come handy. by inspecting the saml response, you get a clearer understanding of the claim names and their respective values, making debugging less of a guessing game.

finally, here’s another example, this one checks if the claim is not empty, which is useful if some claims can be null and you need to map them to a boolean indicating whether the value is present or not.

```xml
    <ClaimsTransformation Id="ClaimIsNotEmpty" TransformationMethod="IsStringClaimNotEmpty">
      <InputClaims>
        <InputClaim ClaimTypeReferenceId="user_profile_id" TransformationClaimType="inputClaim"/>
      </InputClaims>
      <OutputClaims>
        <OutputClaim ClaimTypeReferenceId="user_profile_id_present" TransformationClaimType="outputClaim"/>
      </OutputClaims>
    </ClaimsTransformation>
<ClaimsSchema>
  <ClaimType Id="user_profile_id_present">
    <DataType>boolean</DataType>
	<DefaultPartnerClaimTypes>
      <DefaultPartnerClaimType>user_profile_id</DefaultPartnerClaimType>
    </DefaultPartnerClaimTypes>
	<UserHelpText>User profile present flag</UserHelpText>
  </ClaimType>
</ClaimsSchema>
```

here the transformation `isstringclaimnotempty` returns true if the input claim `user_profile_id` is not empty, and maps the value to a new boolean claim called `user_profile_id_present`.

in terms of reading materials, i’d strongly suggest taking a deep dive into the official azure ad b2c documentation on custom policies. the documentation can be a bit dense, but it's an invaluable resource to really understand the nuts and bolts. specifically the section on claim transformations and defining claims schemas, that's where you'll find the information on the transformation methods that i have shown above. also, i recommend going through any online saml documentation. understanding the saml structure and how claims are delivered can help you identify some common issues. and you know, there are also a few books about web application security, that cover protocols like saml and oauth, but it takes a lot of time to go through one of those, so i would prioritize the other two resources first.

remember, troubleshooting these type of issues is often about breaking things down to their core. check the saml response, verify the idp config, ensure your b2c policies are correct, and use the right transformation. and if all else fails, a fresh pot of coffee and rubber duck debugging can do wonders. it is always amazing how describing the problem out loud to a rubber duck can make it suddenly clear, isn't it?
