---
title: "What phrases are used in the Better Specs example?"
date: "2024-12-23"
id: "what-phrases-are-used-in-the-better-specs-example"
---

,  It's a valid query, and it highlights a critical aspect of writing effective software specifications: understanding the language we use and its implications. I’ve spent years working through complex systems, and the language we adopt in specs has always been a cornerstone of smooth collaboration and successful deployments.

From what I've observed, the so-called "Better Specs" approach usually doesn't involve inventing entirely new language; rather, it refines how we use existing phrases and structures. It's about clarity, precision, and avoiding ambiguity. You won't find magic bullet phrases, but rather a collection of conventions, combined thoughtfully to create something much more useful. Let’s consider some of the common categories and then I'll dive into some examples.

First, you will see an emphasis on *active* voice. Instead of “the data will be processed,” we often state, “the system shall process the data.” The active voice places the action more clearly, and it also makes it easier to assign the action to an explicit entity, removing a lot of guesswork down the line.

Second, there’s the use of modal verbs. *Shall*, *should*, *may*, and *will* are incredibly important. Each signifies a different level of requirement:
    *   **Shall:** Indicates a mandatory requirement. “The system *shall* authenticate the user.”
    *   **Should:** Indicates a recommendation, a best practice. “The system *should* log all failed authentication attempts.”
    *   **May:** Indicates optionality. “The system *may* provide a summary report.”
    *   **Will:** Indicates a declaration or promise. "The system *will* be deployed to a staging environment.”

Third, we have conditional statements, which use “if,” “when,” “then,” or “where.” These delineate specific conditions or triggers that initiate specific actions or outcomes.

Fourth, consistent terminology. Using the same words to describe the same thing across the entire specification makes it readable and clear. If we use “data element” in one place and then “field” in another when referring to the same thing, we end up with confusion. A glossary is vital, especially for larger projects.

Finally, avoiding vague terms. Phrases like “as quickly as possible” or “user-friendly” are subject to interpretation. Instead, we want to define specific metrics. Perhaps, “the page shall load in under 2 seconds” is far better than “as quickly as possible”.

Now, let's illustrate this with code examples. Remember, specifications aren't code, but they dictate how code should function. These code snippets aren't from a single project, but rather are illustrative of patterns I’ve seen repeatedly.

**Example 1: Authentication and Authorization**

Let’s imagine a scenario requiring user authentication and authorization, where the system should prevent unauthorized access to certain endpoints.

```python
# Specification-influenced code (Python example, could be anything)

class Authenticator:
    def authenticate(self, username, password):
        # Implements the "shall" - MUST authenticate users given correct credentials.
        # Failure cases will have to be implemented appropriately.
        if self._credentials_valid(username, password):
            print(f"User '{username}' authenticated.")
            return True
        else:
            print("Authentication failed.")
            return False

    def _credentials_valid(self, username, password):
       # Here are more implementation details.
       return username == "user" and password == "password" # Simplified for example.

class AuthorizationManager:

    def authorize(self, username, endpoint, role):
       # Implements the "should", and MAY need to be expanded later.
       # Failure cases will have to be handled appropriately.
       if self._access_allowed(username, endpoint, role):
           print(f"User '{username}' authorized for '{endpoint}'.")
           return True
       else:
           print("Authorization failed.")
           return False

    def _access_allowed(self, username, endpoint, role):
         # This implements the details of the "shall".
         # Here, the logic is simple. Roles could be pulled from a database
         # instead of being hard-coded as in the following example.

         if role == "admin" and endpoint == "/admin":
            return True
         if role == "user" and endpoint == "/user":
            return True
         return False

auth_manager = AuthorizationManager()
auth = Authenticator()
user = "user"
password = "password"
role = "user"
endpoint = "/user"
admin_role = "admin"
admin_endpoint = "/admin"

auth_result = auth.authenticate(user, password)
if auth_result:
    auth_manager.authorize(user, endpoint, role) # User authorization
    auth_manager.authorize(user, admin_endpoint, admin_role) # Fails Authorization

```

Here, an ideal specification might contain phrases like:

*   "The system *shall* authenticate users with valid credentials." This is enforced in the `Authenticator.authenticate` method.
*   "The system *should* log all failed authentication attempts." (This isn't implemented in the simplified example but would ideally be present.)
*   "The system *shall* prevent unauthorized access to restricted endpoints based on user roles.” This is enforced by the `AuthorizationManager`.

**Example 2: Data Processing**

Suppose we need to process financial transactions.

```python
# Example of some basic data processing logic
def process_transaction(transaction, valid_currencies):
    if not transaction:
        raise ValueError("Transaction cannot be empty.") # Condition where processing cannot be done
    if not transaction['currency'] in valid_currencies:
        print(f"Transaction currency {transaction['currency']} is invalid.")
        return False
    transaction['status'] = "processed" # Action done if condition was met
    return transaction

valid_currencies = ['USD', 'EUR', 'GBP']

transaction_1 = {'id': 1, 'amount': 100, 'currency': 'USD'}
transaction_2 = {'id': 2, 'amount': 200, 'currency': 'JPY'}
transaction_3 = {} # Empty transaction

processed_transaction = process_transaction(transaction_1, valid_currencies)
if processed_transaction:
    print(f"Transaction with id {processed_transaction['id']} processed successfully.")
else:
    print("Transaction processing failed.")

try:
    process_transaction(transaction_3, valid_currencies)
except ValueError as e:
    print(f"Error: {e}")

process_transaction(transaction_2, valid_currencies) # Triggers non-fatal case in this example
```

An example specification might include:
*   “The system *shall* process financial transactions."
*   “*If* a transaction's currency is not in the list of valid currencies, *then* the transaction *shall* be marked as invalid and processing *shall* stop.”
*   "The system *should* log all processed transactions, including success and failure cases."

**Example 3: Reporting Module**

Consider a simple reporting module.

```python
def generate_report(data, report_type):
    if not data:
        raise ValueError("No data to generate report.")

    if report_type == 'summary':
      return f"Summary Report: {len(data)} items found." # Output of report when condition met
    elif report_type == "detailed":
        report_output = "Detailed Report:\n"
        for item in data:
            report_output += f"- {item}\n"
        return report_output
    else:
      raise ValueError("Invalid report type specified") # Condition where report cannot be made

report_data = ["item 1", "item 2", "item 3"]
try:
  summary_report = generate_report(report_data, "summary")
  print(summary_report)
  detailed_report = generate_report(report_data, "detailed")
  print(detailed_report)
  invalid_report = generate_report(report_data, "invalid")
except ValueError as e:
   print(f"Error: {e}")
```

Here, a corresponding specification may include phrases like:
*   “The system *shall* generate reports based on the provided data."
*   “*If* the report type is ‘summary,’ *then* the report *shall* display a count of the data items.”
*   “*If* the report type is ‘detailed’, *then* the report *shall* list each data item individually.”
*   "The system *may* support additional report types."

These snippets, of course, simplify the real-world challenges, but I think they do show how the phrasing in the specifications translates into code implementation. These examples highlight how “better specs” don't rely on magic words, but rather, on the consistent and precise application of structured language.

For further reading, I highly recommend *Software Requirements* by Karl Wiegers and Joy Beatty; and *Writing Effective Use Cases* by Alistair Cockburn. These resources dive deep into the science and art of requirements elicitation and specification, and offer a great foundation for moving beyond simple sentences to well-crafted documentation. Also, ISO/IEC/IEEE 29148:2018, the standard for systems and software engineering, is an essential reference.

The most effective specifications come from combining those elements in a way that is easy to follow, unambiguous, and allows for precise coding. In my experience, the result is always a more accurate and robust system built with less friction and fewer surprises along the way.
