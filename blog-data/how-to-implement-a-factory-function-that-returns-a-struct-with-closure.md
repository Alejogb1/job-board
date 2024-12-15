---
title: "How to Implement a factory function that returns a struct with closure?"
date: "2024-12-15"
id: "how-to-implement-a-factory-function-that-returns-a-struct-with-closure"
---

alright, let's break down this factory function returning a struct with closures thing. it's a pattern i've bumped into a bunch over the years, especially when trying to encapsulate state or behavior in a clean way. i'm gonna try to keep this practical and down-to-earth, like we're pair programming.

so, the core idea is that instead of directly creating a struct, we use a function (the "factory") to do that. this factory function can then set up the struct with specific initial values or, more importantly, with functions that close over some private state. this state isn't directly accessible from outside the struct, which gives you that data-hiding aspect, which is great for keeping your code modular.

i remember one time i was working on a custom rendering engine for a tiny embedded system, and i needed to manage different shader types. using a factory function for each shader became indispensable. each shader struct contained functions for things like "prepare," "render," "cleanup," etc, and these functions all closed over the specific shader program's id, its uniform locations, and other resources that we allocated behind the scenes. the main loop just worked with these structs, it had no idea of the internal plumbing. this really simplified debugging and prevented tons of accidental cross-contamination of resources, that was a win, i still remember this clearly today.

let's get into the code, we need to show some code here. imagine we want to create a struct that manages a simple counter. we’ll use go, because i've always found its syntax quite readable for this particular pattern, but this concept applies pretty universally.

```go
package main

import "fmt"

type Counter struct {
	increment func() int
	getValue  func() int
}

func NewCounter(initialValue int) Counter {
	var count = initialValue

	increment := func() int {
		count++
		return count
	}

	getValue := func() int {
		return count
	}

	return Counter{
		increment: increment,
		getValue:  getValue,
	}
}

func main() {
	myCounter := NewCounter(10)
    fmt.Println(myCounter.getValue()) // prints 10
    myCounter.increment()
    fmt.Println(myCounter.getValue()) // prints 11
	myCounter.increment()
	fmt.Println(myCounter.getValue()) // prints 12
}
```
in this example the `newcounter` function is our factory. it takes an `initialvalue`, initializes a `count` variable local to the function, and returns a struct containing two closure functions. the closures can access and modify that count variable, but code outside of the struct cannot access it directly. this is the crux of what makes the pattern so versatile.

one of the benefits of this factory function is that you can inject dependencies easily, this is not shown in the above example because it is a simple counter but is a huge advantage for this pattern. imagine that you have a database connection that you need to pass to the created struct methods. in that case, the factory would take the database connection as input.

```go
package main

import (
	"fmt"
	"database/sql"
	_ "github.com/lib/pq" // postgres driver
)

type User struct {
	getUserName func(int) (string, error)
}

func NewUser(db *sql.DB) User {
  getUserName := func(id int) (string, error){
    var name string
    err := db.QueryRow("select name from users where id = $1", id).Scan(&name)
	if err != nil {
		return "", err
	}
    return name, nil
  }

  return User{
    getUserName: getUserName,
  }
}


func main() {
	db, err := sql.Open("postgres", "user=youruser password=yourpassword dbname=yourdbname sslmode=disable")
	if err != nil {
		panic(err)
	}
    defer db.Close()

    user := NewUser(db)
	name, err := user.getUserName(1) // replace 1 with user id
	if err != nil {
		fmt.Println("error:",err)
	}
	fmt.Println("user name: ", name)
}
```
this is another useful practical example. in this example, the factory receives the database connection and each method of the struct has access to this injected database connection.

i've used this pattern in the past with complex objects like dealing with finite state machines, different types of protocol parsers, and even for implementing undo/redo mechanisms. in every one of these cases, the factory was useful because it allowed me to create isolated objects, each with its own enclosed state and behavior. this approach improved the organization and testability of my code a lot. a good implementation of a finite state machine can be found in "game programming patterns" by robert nystrom. also, you can look at patterns that use this strategy on other books and articles such as "design patterns: elements of reusable object-oriented software" by the gang of four.

now let’s consider a slightly more elaborate example. in this example, we will use a closure for caching operations:
```go
package main

import (
	"fmt"
	"time"
)

type CachedOperation struct {
	execute func(int) int
}

func NewCachedOperation(expensiveOperation func(int) int) CachedOperation {
	cache := make(map[int]int)
	execute := func(input int) int {
		if result, ok := cache[input]; ok {
			fmt.Println("returning cached value")
			return result
		}
		fmt.Println("calculating value")
		result := expensiveOperation(input)
		cache[input] = result
		return result
	}
	return CachedOperation{
		execute: execute,
	}
}

func expensiveOperation(input int) int {
	time.Sleep(time.Second) // Simulate a long operation.
	return input * 2
}

func main() {
	cached := NewCachedOperation(expensiveOperation)
	fmt.Println(cached.execute(5)) // first execution, calculates and caches
	fmt.Println(cached.execute(5)) // second execution, returns cached value
	fmt.Println(cached.execute(6)) // first execution with different input
}

```
in this scenario, the `newcachedoperation` is our factory. it takes a function `expensiveoperation` as a dependency, creates a cache, and creates the `execute` function which is a closure with access to both the cache and the dependency `expensiveoperation`. each call to `execute` checks if the input is already in the cache, and if so, returns the cached result. otherwise, it executes the passed expensive operation. so basically is a lazy operation and it's useful in cases in which you need to execute a computationally expensive operation that has the same output always and you want to cache the result.

a lot of the patterns discussed can be found in books. if you want to deepen your knowledge in this particular pattern i would recommend to check books that discuss object oriented programming or functional programming, since factory functions are widely used by both paradigms. but in any case, those books that i already mentioned will be really useful for your studies.

let's talk about some things that are good practice when using this pattern. first, clearly document what the factory function does and how to use the returned struct. this saves a bunch of headaches down the road. second, make sure to keep the scope of your private variables as small as possible to avoid any conflicts, so the only variables that are private are the ones required by the closure. third and last, if your factory gets too complex, consider breaking it down into smaller helper functions to keep it maintainable.

in my experience, this pattern has never let me down. it's one of those tools in my programming toolkit that i use all the time, and it never gets old. oh, and one thing i never got tired of, do you know why programmers prefer dark mode? because light attracts bugs hahahaha. anyway back to seriousness.

so, to wrap things up. a factory function that returns a struct with closure is a robust technique to encapsulate state and behavior. it provides a clean, organized, and testable way to structure your code. i would always recommend it for more complex projects. don’t hesitate to use this pattern as you go to develop all kinds of different structures or programs, and you will see how useful it is. let me know if you need clarification on a particular case or if i have missed something.
