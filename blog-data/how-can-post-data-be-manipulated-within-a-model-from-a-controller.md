---
title: "How can POST data be manipulated within a model from a controller?"
date: "2024-12-23"
id: "how-can-post-data-be-manipulated-within-a-model-from-a-controller"
---

Okay, let's tackle this. I’ve definitely seen my fair share of situations where manipulating post data within a model feels necessary. It's a common scenario, and if not handled correctly, it can lead to all sorts of headaches, from unintended data modifications to tightly coupled architectures that become nightmares to maintain. We need to be methodical and strategic about how we approach this.

The core issue, as I see it, is that the controller's job is primarily about *orchestration*. It receives the request (including the post data), it validates the request (to a degree), and then it determines what action to take. Models, on the other hand, are about the *data itself* and any associated business logic. This distinction is fundamental, and blurring it by pushing post-data manipulation directly into the model is generally a bad idea. We want to avoid having our models overly reliant on the nuances of the request object.

Now, I'm not suggesting models should be completely oblivious. There *are* cases where a model needs to respond to data provided in a post request, but there's a clear separation of concerns we should strive for. My approach has generally been to utilize what I call ‘data shaping’ or ‘data mapping’ techniques between the controller and the model, ensuring that the model only sees well-defined, processed data relevant to its specific domain. This is crucial for long-term maintainability.

I remember a particularly challenging project a few years back where we initially fell into the trap of having our models directly accessing the `request` object to get at post variables. It worked fine… until we decided to introduce a web socket interface. Suddenly, the model logic that was coupled to the http request was entirely out of its depth. That's where the pain points started, teaching us a valuable lesson about keeping controller-level concerns separate. The solution was to refactor, introduce a transformation layer, and significantly improve the overall architecture.

The question isn’t *can* you manipulate post data in the model. The real question is: *should* you? And the answer, from my perspective, is almost always no, not directly. You should be transforming or sanitizing the incoming data beforehand.

So, how can we achieve this data transformation, ensuring our models receive only the data they need, in the format they expect? Let's look at three practical code snippets, using a fictional example of creating a user in a hypothetical system. We'll assume a simplified PHP environment here.

**Snippet 1: Basic Sanitization in the Controller**

This is the most straightforward approach. Before passing data to the model, the controller performs some basic transformations, such as trimming whitespace and basic type validation.

```php
<?php

class UserController {

    public function createUser(array $requestData) {
        $name = isset($requestData['name']) ? trim($requestData['name']) : '';
        $email = isset($requestData['email']) ? trim($requestData['email']) : '';
        $password = isset($requestData['password']) ? $requestData['password'] : null;

        if (!$name || !$email || !$password) {
            // Handle error, return to the view or whatever the proper error handling method is in your application.
            return 'Invalid data.';
        }


        $userData = [
            'name' => $name,
            'email' => $email,
            'password' => password_hash($password, PASSWORD_DEFAULT) // hashing in the controller is reasonable for this
        ];

        $user = new User();
        $user->create($userData);


        return 'User created'; // Or redirect somewhere, depends on the app
    }
}

class User {

    public function create(array $userData){
       //Insert user into the database with $userData
        // For brevity let's assume database operations are handled elsewhere in an abstracted way.
        echo 'User created with the following data: '. json_encode($userData);
    }
}

// Sample usage:
$controller = new UserController();
$postData = ['name' => '  John Doe  ', 'email' => '  john.doe@example.com  ', 'password' => 'Secret123'];
echo $controller->createUser($postData);
?>
```

In this example, the controller cleans the incoming data, even hashing the password. The model simply receives an associative array of processed data, which is significantly cleaner and more testable.

**Snippet 2: Using Data Transfer Objects (DTOs)**

As your application grows, this pattern might become a bit cumbersome. DTOs allow you to encapsulate the data and any associated transformation logic in a dedicated class. This improves code organization and promotes reusability.

```php
<?php
class UserData {

    public string $name;
    public string $email;
    public string $password;

    public function __construct(array $data)
    {
        $this->name = isset($data['name']) ? trim($data['name']) : '';
        $this->email = isset($data['email']) ? trim($data['email']) : '';
        $this->password = isset($data['password']) ? $data['password'] : null;

        if (!$this->name || !$this->email || !$this->password)
        {
            throw new InvalidArgumentException("Invalid user data");
        }

    }

    public function toArray():array{
        return [
            'name'=>$this->name,
            'email'=>$this->email,
            'password'=>password_hash($this->password, PASSWORD_DEFAULT)
        ];
    }

}

class UserController {

    public function createUser(array $requestData) {
        try{
            $userData = new UserData($requestData);
            $user = new User();
            $user->create($userData->toArray());

            return 'User created.';

        }catch (InvalidArgumentException $e) {
            return "Error: " . $e->getMessage();
        }


    }
}

class User {

    public function create(array $userData){
       //Insert user into the database with $userData
        // For brevity let's assume database operations are handled elsewhere in an abstracted way.
        echo 'User created with the following data: '. json_encode($userData);
    }
}

// Sample usage:
$controller = new UserController();
$postData = ['name' => '  Jane Doe  ', 'email' => '  jane.doe@example.com  ', 'password' => 'SuperSecret456'];
echo $controller->createUser($postData);
?>
```

Here, the `UserData` class takes care of creating the object based on the request data and handles the sanitization and data transformation, including password hashing, offering better type safety and validation directly in the DTO. The controller is more focused on the orchestration and remains simpler.

**Snippet 3: Using a Service or a Mapper**

In more complex applications, you might introduce a service or a mapper layer. This approach can be beneficial if the transformations are more intricate or involve multiple data sources.

```php
<?php
class UserService {

    public function createUser(array $requestData) : array|string
    {
        $name = isset($requestData['name']) ? trim($requestData['name']) : '';
        $email = isset($requestData['email']) ? trim($requestData['email']) : '';
        $password = isset($requestData['password']) ? $requestData['password'] : null;

        if (!$name || !$email || !$password) {
            // Handle error, return to the view or whatever the proper error handling method is in your application.
            return 'Invalid data.';
        }

       $userData = [
          'name' => $name,
          'email' => $email,
          'password' => password_hash($password, PASSWORD_DEFAULT),
          'created_at' => date("Y-m-d H:i:s"), //Added created at field
        ];
        
       $user = new User();
       $user->create($userData);

        return 'User created';
    }
}
class UserController {

    public function createUser(array $requestData) {
        $userService = new UserService();
        return $userService->createUser($requestData);
    }
}

class User {

    public function create(array $userData){
       //Insert user into the database with $userData
        // For brevity let's assume database operations are handled elsewhere in an abstracted way.
        echo 'User created with the following data: '. json_encode($userData);
    }
}


// Sample usage:
$controller = new UserController();
$postData = ['name' => '   Peter Jones  ', 'email' => '  peter.jones@example.com   ', 'password' => 'StrongPassword123!'];
echo $controller->createUser($postData);

?>
```

Here, the `UserService` acts as an intermediary, encapsulating all of the data preparation and manipulation logic prior to calling the create method on the user model. Note that in larger systems the `create` method of a model will rarely interact with the database directly, but rather with a repository or data layer.

In each of these cases, you notice that the model is not dependent on the incoming request directly. It is simply receiving a clean array that represents its own data requirements.

For further reading, I'd suggest exploring resources such as "Patterns of Enterprise Application Architecture" by Martin Fowler, which dives into different architectures and data management practices. Also, "Clean Architecture" by Robert C. Martin (Uncle Bob) is a great source on how to structure your application and adhere to the separation of concerns. Another solid reference is "Domain-Driven Design" by Eric Evans, which deals with modeling complex domains. These texts will offer a deeper understanding of the principles I've outlined here, helping you avoid those troublesome situations I’ve seen in the past. By using appropriate data shaping techniques before reaching your models, you'll build a more robust, maintainable, and adaptable system.
