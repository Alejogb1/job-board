---
title: "Where should validation logic for disallowed words in comments be implemented: in the command handler or the domain model?"
date: "2024-12-23"
id: "where-should-validation-logic-for-disallowed-words-in-comments-be-implemented-in-the-command-handler-or-the-domain-model"
---

, let's tackle this one. I've seen this scenario play out countless times over the years, and it’s a classic example of where differing architectural philosophies can lead to distinct implementations. When it comes to validating disallowed words in comments, deciding between the command handler and the domain model isn't just about code placement—it’s about upholding the integrity of your application's design. My experience leans heavily toward placing this type of validation within the domain model, but let me explain why, step-by-step, and provide some illustrative code along the way.

I recall a particularly challenging project a few years back. We were building a large-scale forum application, and of course, we had a requirement to prevent certain offensive terms from being included in user comments. Initially, the validation was implemented in our command handlers. This made perfect sense at first; the command handler was orchestrating the actions, receiving the command, and seemingly in a prime position to validate the input. The code in the handler would, after receiving a 'PostCommentCommand', check the comment text against a list of disallowed words and either throw an exception or reject the command if any were found.

The problem, however, became evident as the application grew. We started introducing different user interfaces and different ways of posting comments, beyond the initial web frontend. Each time we added a new entry point, we found ourselves having to replicate the same validation logic within each handler. This, predictably, led to code duplication and a maintenance nightmare. It violated the DRY (Don't Repeat Yourself) principle and made the application incredibly brittle. Any change to the list of disallowed words would require updating multiple command handlers across the project. This also made it hard to centralize validation policy.

That's when we refactored, moving the core validation logic down into the domain model. The shift, while requiring some initial effort, dramatically simplified the overall system and reduced code redundancy.

Here’s the reasoning behind this approach and how we implemented it. The domain model should represent the core business rules and logic of your application. In this case, the ability to post a valid comment, free of disallowed words, is a fundamental business rule. Therefore, it is naturally a concern of our domain. A `Comment` entity should not exist in a state where it contains such words. The `Comment` itself, as a domain entity, should be responsible for ensuring its own validity. It should enforce the constraints of what constitutes a valid comment within its own methods.

Let’s see some code. Initially, our command handler might have looked something like this (pseudocode to illustrate, not language specific):

```pseudocode
class PostCommentCommandHandler {
    private DisallowedWordsRepository disallowedWordsRepository;

    public void handle(PostCommentCommand command) {
        var commentText = command.getText();
        var disallowedWords = disallowedWordsRepository.getDisallowedWords();

        foreach (var word in disallowedWords) {
             if (commentText.Contains(word)) {
                throw new InvalidCommentException("Comment contains disallowed word.");
            }
        }

        // create and persist the comment
        var comment = new Comment(command.getUserId(), commentText);
        commentRepository.save(comment);
    }
}
```

This snippet highlights the problem; the validation is intertwined with the command handling logic and needs access to infrastructure concerns. The handler knows about the `disallowedWordsRepository`—it’s responsible for fetching the prohibited words.

Now, let’s consider moving the validation into the domain model itself. We'd shift the code to the `Comment` entity:

```pseudocode
class Comment {
    private string text;
    private userId;

    public Comment(userId userId, string text, IDisallowedWordsPolicy policy) {
        this.userId = userId;
        if (!policy.isAllowed(text))
            throw new InvalidCommentException("Comment contains disallowed words.");
        this.text = text;
    }

   public string getText() { return text; }
}

interface IDisallowedWordsPolicy {
  boolean isAllowed(String text);
}

class DisallowedWordsPolicy implements IDisallowedWordsPolicy {
  private DisallowedWordsRepository disallowedWordsRepository;

  public DisallowedWordsPolicy(DisallowedWordsRepository disallowedWordsRepository) {
    this.disallowedWordsRepository = disallowedWordsRepository;
  }

  public boolean isAllowed(String text) {
        var disallowedWords = disallowedWordsRepository.getDisallowedWords();

        foreach (var word in disallowedWords) {
             if (text.contains(word)) {
                return false;
            }
        }
        return true;
    }
}

```

Now our command handler simplifies drastically:

```pseudocode
class PostCommentCommandHandler {
  private DisallowedWordsPolicy policy;

   public PostCommentCommandHandler (DisallowedWordsRepository disallowedWordsRepository) {
      this.policy = new DisallowedWordsPolicy(disallowedWordsRepository);
  }

    public void handle(PostCommentCommand command) {
      // domain will throw exception if invalid text
        var comment = new Comment(command.getUserId(), command.getText(), policy);
        commentRepository.save(comment);
    }
}
```

Here, we've delegated the validation of comment text to the `Comment` constructor. Now, the `Comment` entity itself is responsible for ensuring that its state is valid. Notice also the use of a separate Policy for the disallowed words; this keeps the `Comment` entity focused on the domain concept of a comment, rather than the specific rule implementation, further enhancing separation of concerns. This approach also allows for reuse of the policy in other domain methods, not just at creation time.

This revised structure has several benefits:

1.  **Encapsulation:** The validation logic is now encapsulated within the domain model. The command handler simply delegates to the domain object, focusing solely on coordinating application logic.
2.  **Single Source of Truth:** The validation logic is defined in one place, the `Comment` entity and `DisallowedWordsPolicy`. Changes to disallowed word lists are now easier to manage, and avoid code duplication.
3.  **Testability:** Domain objects can be tested independently of command handlers, simplifying unit testing.
4.  **Domain Language:** It aligns with domain-driven design (ddd), where the domain model accurately represents the business domain.
5.  **Reusability:** The validation can be easily reused within other domain contexts where a similar need exists (e.g., validating user bios).

A few points, further details, to clarify:

*   **Policy Implementation:** You'll notice we have created a `DisallowedWordsPolicy`. This policy object can encapsulate the actual logic related to accessing the list of prohibited words (often from a database or other configuration source). This further decouples the domain logic from infrastructure specifics.

*   **Dependency Injection:** When using a proper application framework, the DisallowedWordsRepository can be injected into the `DisallowedWordsPolicy` and `PostCommentCommandHandler`.

In my experience, this approach almost always leads to a more maintainable and robust system. While the command handler is a convenient place to implement initial validations, ultimately the responsibility for ensuring data validity rests with the domain. The domain model represents the core business rules, and validation logic is a fundamental aspect of these rules.

For deeper exploration into domain-driven design, I'd recommend reading *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans. For a practical look at how to apply these concepts in code, *Implementing Domain-Driven Design* by Vaughn Vernon is invaluable. Additionally, the *Patterns of Enterprise Application Architecture* by Martin Fowler offers extensive insights into various architectural patterns and their implications. These texts have shaped my approach to building software for many years and I believe they can greatly benefit anyone working in this field. These resources offer concrete examples that go far beyond hypothetical scenarios.
