---
title: "How to do MockMvc : handling unique constraints?"
date: "2024-12-14"
id: "how-to-do-mockmvc--handling-unique-constraints"
---

so, you're hitting that fun wall with `mockmvc` and unique constraints, huh? i've been there, many times. it's a classic "testing-the-edge-cases" scenario. it usually surfaces when you're trying to simulate concurrent requests or just trying to make sure your api doesn't let duplicates slip through.

my first time encountering this was back in the early days of a project building a user registration system. we were using spring boot and, of course, `mockmvc` for our tests. everything seemed to work perfectly until, boom, we started seeing weird database exceptions on our integration tests. turns out, we weren't properly handling the unique constraint on the email field. it was a good lesson in how easily things can go wrong.

so, here's the deal. `mockmvc` itself doesn't really know about database constraints or unique indexes. it's just simulating http requests and responses. the heavy lifting for handling unique constraints happens at your service or repository layer (sometimes even at the database itself).

what that means is that you can't directly tell `mockmvc`, "hey, make sure this request doesn't violate the unique constraint." instead, you have to craft your test scenarios to *cause* those violations and verify that your code handles them gracefully.

let's break this down, shall we?

first, let's assume you have a typical spring boot controller that takes some data, saves it into the database, and returns a response. a simplistic example would be something like this:

```java
@RestController
@RequestMapping("/users")
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User createdUser = userService.createUser(user);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdUser);
    }
}
```

and the `userService` layer is doing the persistence logic:

```java
@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User createUser(User user) {
        return userRepository.save(user);
    }
}
```

now, your `user` model might have a unique constraint, say on the email field, annotated using `@column` or the like in your entity class, or through the schema itself. you'll usually see a `dataIntegrityViolationException` being thrown when inserting a duplicate if the constraint is hit.

now, how do you test this with `mockmvc`? you will need to mock the service layer and provide a mock service that will throw the relevant exception if the duplicate is hit. then you can assert on this in your test using `mockmvc`. here is an example:

```java
@ExtendWith(MockitoExtension.class)
public class UserControllerTest {

    @Mock
    private UserService userService;

    @InjectMocks
    private UserController userController;

    private MockMvc mockMvc;

    @BeforeEach
    public void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(userController).build();
    }

    @Test
    public void testCreateUser_duplicateEmail_returnsConflict() throws Exception {
         User user = new User();
         user.setEmail("test@test.com");
         user.setUserName("test");

        when(userService.createUser(any(User.class))).thenThrow(new DataIntegrityViolationException("duplicate email"));


        mockMvc.perform(post("/users")
                .contentType(MediaType.APPLICATION_JSON)
                .content(asJsonString(user)))
                .andExpect(status().isConflict());

        verify(userService, times(1)).createUser(any(User.class));
    }


    static String asJsonString(final Object obj) {
        try {
            return new ObjectMapper().writeValueAsString(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}

```

this test does a couple of things. first it mocks the service class and throws the `dataintegrityviolationexception` then you mock a post request via the `mockmvc` object and verify the returned http status code.

the idea here is to simulate the behavior of your database (or whatever is throwing that specific exception), and test if you controller is handling it correctly. the `status().isConflict()` is asserting the correct http status.

another approach might involve checking your custom exception handling code. let's say you've written a `@controlleradvice` class to handle the `dataintegrityviolationexception` and return a specific error message:

```java
@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<String> handleDataIntegrityViolation(DataIntegrityViolationException ex) {
      return new ResponseEntity<>("duplicate data found", HttpStatus.CONFLICT);
    }
}
```

in this case you need to also check the error returned message within your `mockmvc` test. the following test shows how to do this.

```java
    @Test
    public void testCreateUser_duplicateEmail_returnsConflict_with_message() throws Exception {
         User user = new User();
         user.setEmail("test@test.com");
         user.setUserName("test");

        when(userService.createUser(any(User.class))).thenThrow(new DataIntegrityViolationException("duplicate email"));


        mockMvc.perform(post("/users")
                .contentType(MediaType.APPLICATION_JSON)
                .content(asJsonString(user)))
                .andExpect(status().isConflict())
                .andExpect(content().string(containsString("duplicate data found")));

        verify(userService, times(1)).createUser(any(User.class));
    }
```

now we are checking both the status code and the returned string in the response.

a point here is that testing in isolation is extremely helpful and speeds up the test writing process and makes your tests more readable. you could mock the repository itself but often the service layer is the one where you will be placing the bulk of the business logic, which is what your tests should try to exercise, not the database implementation details.

now, about resources, you won't find a single book that's just about `mockmvc` and unique constraints. it's more about combining different concepts. however, "testing spring boot applications" by greg l turnquist is a pretty solid book that covers many scenarios for integration testing in spring boot. i recommend that one. also, "clean code" by robert c. martin, it might not seem related directly but it is essential for writing maintainable and testable code. these books have saved me a lot of headaches when i started with integration testing.

finally, i remember once, back when i was working at the local pet shop, i was debugging an issue related to this for hours. turns out, the issue was that i was trying to save a cat with the same 'name' column twice and got the data integrity violation. the fix was to add the 'is_adopted' column to the uniqueness constraints. it turned out to be a good reminder that we should always consider the constraints within the domain model, not only the 'code level' ones. it was one of those times i felt very dumb, but then i laughed it out of course!
hope this helps, let me know if you have any other questions.
