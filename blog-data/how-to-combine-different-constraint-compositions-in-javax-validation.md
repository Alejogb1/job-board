---
title: "How to Combine different constraint compositions in javax validation?"
date: "2024-12-14"
id: "how-to-combine-different-constraint-compositions-in-javax-validation"
---

i see you're hitting that fun wall with javax validation, the one where you want to mix and match different validation rules, almost like you are composing music notes, but instead of music you create software errors. i've been there, trust me. this isn't the kind of thing the standard documentation usually throws at you explicitly, but it's absolutely possible, and quite elegant once you get the hang of it.

let's break down what makes this tricky. usually, you slap annotations directly onto your fields, which is fine for basic cases. think `@notnull`, `@size`, or something simple. but what if your validation logic gets more complex? maybe you have a scenario where a field needs to be a valid email *only if* another field has a specific value. or perhaps you need to apply multiple levels of validation sequentially, where one rule depends on the successful validation of a previous rule. just annotating every field becomes a nightmare quickly.

i remember back in the mid-2000s, working on this web app, we had a form with address details. the user could select between "residential" and "business". if "business" was selected, we needed to enforce a valid company name and a company registration number. for “residential”, these fields should be empty. initially, we tried using if-else blocks in the controller. yeah, not a pretty sight. it became a sprawling mess. we had validation code duplication everywhere, which was a maintenance pain. that’s when i started getting into using custom validators and constraint composition to get a more modular and maintainable validation setup.

so, instead of a monolithic validation setup, javax validation lets you *compose* constraints. this means making reusable validators that can be combined in different ways to achieve the validation logic you need. the core idea is to break down complex requirements into smaller, manageable chunks, and then put those chunks together. the standard annotations give you some basic pieces to begin with. however, the full potential of javax validation lies in how you assemble them.

let’s start simple, you want to perform more than one type of validation at once. for instance you need that field to be not null and also a valid email. you can do it like this:

```java
import javax.validation.constraints.Email;
import javax.validation.constraints.NotNull;
public class User {
    @NotNull(message = "email cannot be null")
    @Email(message = "invalid email format")
    private String email;
}
```

this is a very simple way. when validating an instance of user, both rules will apply. so if it's null, it will fail the `@notnull` constraint and if it is not a valid email it will fail the `@email` constraint. now let's step up the game. what if you want to validate a certain field only if another field has a certain value? the answer for this is using groups. let’s imagine a situation where a user has two types of profile. let’s say one is "basic" and the other is "premium". premium users needs to provide more information than basic users.

this is how you could implement it:

```java
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;
import javax.validation.constraints.Email;
import javax.validation.groups.Default;

public class UserProfile {
  public interface PremiumProfile extends Default {}

  private String profileType;

  @NotBlank(message="username cannot be blank")
  private String username;

  @Email(message="invalid email format", groups = {PremiumProfile.class})
  private String email;

  @Size(min=10, max=10, message="phone number must be 10 digits", groups = {PremiumProfile.class})
  private String phoneNumber;


    public UserProfile(String profileType, String username, String email, String phoneNumber) {
        this.profileType = profileType;
        this.username = username;
        this.email = email;
        this.phoneNumber = phoneNumber;
    }

    public String getProfileType() {
        return profileType;
    }

    public void setProfileType(String profileType) {
        this.profileType = profileType;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }
}

```

in the `userprofile` class, we have declared an interface called premiumprofile. this interface extends the default validation group. then, we are specifying that the email and phone number should be validated only when the premiumprofile group is used during validation. this way, depending on which user profile type is set, you can activate different validation rules. when you validate this class you can do it in the following way, let’s assume you have a validator instance somewhere:

```java
    Validator validator = Validation.buildDefaultValidatorFactory().getValidator();
    UserProfile basicUser = new UserProfile("basic","john","john@example.com","1234567890");
    UserProfile premiumUser = new UserProfile("premium","john","john@example.com","1234567890");

    Set<ConstraintViolation<UserProfile>> basicViolations = validator.validate(basicUser);
    System.out.println("basic user violations: "+ basicViolations.size());

    Set<ConstraintViolation<UserProfile>> premiumViolations = validator.validate(premiumUser,UserProfile.PremiumProfile.class);
    System.out.println("premium user violations: "+ premiumViolations.size());
```

when you run this code, you'll see that the basic user has no violations since it doesn't validate the email and phone number because is not using the `premiumprofile` group. but when validating the premium user, we get 2 violations because both email and phone number have the wrong format and fail the validation. now, let's go even further. what if you want to create a custom validator that validates something more complex than what you can achieve with the standard annotations? you can do that too.

let’s suppose that we need to validate that the phone number entered is a valid local phone number. we have a custom class that determines that. first, we need to create the constraint annotation:

```java
import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.*;

@Target({ElementType.METHOD, ElementType.FIELD, ElementType.ANNOTATION_TYPE, ElementType.CONSTRUCTOR, ElementType.PARAMETER, ElementType.TYPE_USE})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = LocalPhoneNumberValidator.class)
@Documented
public @interface LocalPhoneNumber {

  String message() default "invalid local phone number";

  Class<?>[] groups() default {};

  Class<? extends Payload>[] payload() default {};

}
```

then we need to create the validator itself, this is the class that contains the validation logic.

```java

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

public class LocalPhoneNumberValidator implements ConstraintValidator<LocalPhoneNumber,String> {


  @Override
  public boolean isValid(String value, ConstraintValidatorContext context) {
      //replace this with your real phone number validation logic
    if (value == null) {
        return true;
      }
    return value.matches("\\d{3}-\\d{3}-\\d{4}");
  }
}
```

then, we can use this annotation as a normal validation annotation.

```java
import javax.validation.constraints.NotBlank;
import javax.validation.groups.Default;

public class UserProfile {
  public interface PremiumProfile extends Default {}

  private String profileType;

  @NotBlank(message="username cannot be blank")
  private String username;

  @Email(message="invalid email format", groups = {PremiumProfile.class})
  private String email;

  @LocalPhoneNumber(message="invalid local phone number", groups = {PremiumProfile.class})
  private String phoneNumber;

    public UserProfile(String profileType, String username, String email, String phoneNumber) {
        this.profileType = profileType;
        this.username = username;
        this.email = email;
        this.phoneNumber = phoneNumber;
    }

    public String getProfileType() {
        return profileType;
    }

    public void setProfileType(String profileType) {
        this.profileType = profileType;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }
}
```

now, when validating a premium user if the phone number does not match the format xxx-xxx-xxxx, then you will get a new validation error. and that is pretty much it, with the custom validator now you can add any validation logic you need.

when combining validations, think about breaking the problem down into separate, reusable validators. that way, you keep the complexity manageable, and it is easier to test. groups allow you to activate different validations depending on your context. for instance, you might have a "creation" group and an "update" group. each of them with its own specific validation requirements. there are some resources out there that have helped me a lot during my career. for a really thorough understanding of the bean validation specification, i'd recommend reading the official "bean validation specification (jsr 380)" document. another great resource is "java ee 7 development with wildfly" by michael d. remijan, it has a chapter on bean validation that goes pretty deep into custom validations and compositions. for a broader understanding of the concepts you can read "domain-driven design" by eric evans it is not specific to validation but it helps to separate your domain and business logic from implementation details, and that includes validation.

oh, and remember when they say, "don't reinvent the wheel?" well, sometimes it's fine to reinvent the wheel, as long as you understand how the previous wheel worked. don't go straight away writing custom validation if there is a default validator that does the same, unless you want to learn the inner workings. i’ve spent countless hours debugging silly errors because i forgot the constraint annotation is not what actually does the validation, it only references what does, a common mistake when you start using it. so, take it slow, step by step, and you'll soon have a robust and maintainable validation system.
