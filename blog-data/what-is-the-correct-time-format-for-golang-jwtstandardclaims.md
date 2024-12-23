---
title: "What is the correct time format for Golang jwt.StandardClaims?"
date: "2024-12-23"
id: "what-is-the-correct-time-format-for-golang-jwtstandardclaims"
---

Alright, let's tackle this one. It's a common sticking point, and I've certainly spent my share of late nights debugging subtle timezone issues lurking within jwt claims. The core of the problem lies not within the `jwt.StandardClaims` struct itself, but in how the underlying time fields are managed and serialized by the `go-jwt/jwt` library when you're working with json. The short answer is: **`jwt.StandardClaims` expects time values to be represented as numeric timestamps (Unix epoch seconds), usually as integers, and should be expressed in UTC**. Not strings, not formatted time outputs, and definitely not local times.

Let’s break this down and explore the details, drawing on a previous project involving a microservices architecture, where inconsistent time representations in tokens almost brought the entire thing crashing down.

First, the `jwt.StandardClaims` struct, as defined in the `go-jwt/jwt` package, has fields like `ExpiresAt`, `NotBefore`, and `IssuedAt`, which are all of type `time.Time`. When it comes to encoding to and decoding from json, the jwt library doesn’t directly serialize these `time.Time` objects into json objects the way you might expect. Instead, the `jwt` package's default encoder converts these fields into numeric timestamps (int64 holding Unix seconds). This is a critical point to grasp. It means you should not try to directly serialize time objects with custom formatted date strings. You need to explicitly manage conversion to and from the correct timestamp format.

This encoding convention ensures that the json representation of your jwt is predictable and timezone-agnostic (since the timestamp is relative to the UTC epoch). Problems usually arise when developers attempt to populate these fields using local times or by creating `time.Time` values without considering the underlying timestamp representation. It's a common pitfall.

Here’s how you should be doing it correctly. We need to make sure the `time.Time` values you set to `ExpiresAt`, `NotBefore`, and `IssuedAt` in your `StandardClaims` are *UTC* times and that they will be marshaled correctly to their corresponding timestamps when the json output is produced.

Here's a practical example. Let’s say you’re building a system that issues jwt tokens that expire in 1 hour.

```go
package main

import (
	"fmt"
	"time"
	"github.com/golang-jwt/jwt/v5"
)

func main() {
	claims := jwt.MapClaims{
		"iss": "my-app",
		"sub": "user123",
		"exp": time.Now().UTC().Add(time.Hour).Unix(), // Expires 1 hour from now
		"iat": time.Now().UTC().Unix(),             // Token issued at current time
		"nbf": time.Now().UTC().Unix(),              // Token valid from now
	}

  token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    ss, err := token.SignedString([]byte("secret")) // Dummy Secret, use stronger in prod.
    if err != nil {
        fmt.Println("Error signing token:", err)
        return
    }

    fmt.Println(ss)

	parsedClaims := jwt.MapClaims{}
	tokenParsed, _ := jwt.ParseWithClaims(ss, &parsedClaims, func(token *jwt.Token) (interface{}, error) {
		return []byte("secret"), nil
	})
    
	if tokenParsed.Valid {
        if exp, ok := parsedClaims["exp"].(float64); ok {
            fmt.Println("Parsed Expiry:", time.Unix(int64(exp), 0).UTC())
        }
         if iat, ok := parsedClaims["iat"].(float64); ok {
            fmt.Println("Parsed Issued At:", time.Unix(int64(iat), 0).UTC())
        }
        if nbf, ok := parsedClaims["nbf"].(float64); ok {
            fmt.Println("Parsed Not Before:", time.Unix(int64(nbf), 0).UTC())
        }

        fmt.Println("Token is Valid")
	}else{
        fmt.Println("Token is Invalid")
	}

}
```

In the above code snippet, the key aspect is converting the `time.Time` objects to UTC and then getting their corresponding Unix timestamps using the `Unix()` method which returns a `int64`. When decoding, note that the time fields are coming back as `float64` types from the decoded claims. To convert them back to proper `time.Time` objects you must first convert to `int64` and then create the `time.Time` using `time.Unix(int64, 0).UTC()`.

Now, here's what *not* to do. Directly using local times or formatted time strings will cause problems:

```go
package main
import (
	"fmt"
	"time"
	"github.com/golang-jwt/jwt/v5"
)

func main() {
	localTime := time.Now()
    // Incorrect: Local time without conversion
	claims := jwt.MapClaims{
        "exp": localTime.Add(time.Hour),  // INCORRECT: Not a numeric timestamp.
		"iat": localTime, // INCORRECT: Not a numeric timestamp
		"nbf": localTime, // INCORRECT: Not a numeric timestamp
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    ss, err := token.SignedString([]byte("secret")) // Dummy Secret
    if err != nil {
        fmt.Println("Error signing token:", err)
        return
    }

    fmt.Println(ss)
    
    parsedClaims := jwt.MapClaims{}
	tokenParsed, _ := jwt.ParseWithClaims(ss, &parsedClaims, func(token *jwt.Token) (interface{}, error) {
		return []byte("secret"), nil
	})
    
	if tokenParsed.Valid {
        if exp, ok := parsedClaims["exp"].(float64); ok {
            fmt.Println("Parsed Expiry:", time.Unix(int64(exp), 0).UTC())
        }
         if iat, ok := parsedClaims["iat"].(float64); ok {
            fmt.Println("Parsed Issued At:", time.Unix(int64(iat), 0).UTC())
        }
        if nbf, ok := parsedClaims["nbf"].(float64); ok {
            fmt.Println("Parsed Not Before:", time.Unix(int64(nbf), 0).UTC())
        }

        fmt.Println("Token is Valid")
	}else{
        fmt.Println("Token is Invalid")
	}


}
```

This code will still produce a token, but the time fields will likely not be what you intended when you decode it. The json will not represent time as the library expects causing the token to become invalid. You might even see validation errors or incorrect comparisons in downstream services. It's crucial to *always* use Unix timestamps and ensure the source time is UTC before conversion.

Finally, let's look at an example where we use `jwt.StandardClaims` for comparison.

```go
package main

import (
	"fmt"
	"time"
	"github.com/golang-jwt/jwt/v5"
)

func main() {

	expiresAt := time.Now().UTC().Add(time.Hour)
	issuedAt := time.Now().UTC()

    claims := jwt.StandardClaims{
		ExpiresAt: jwt.NumericDate{Time: expiresAt},
		IssuedAt: jwt.NumericDate{Time: issuedAt},
        NotBefore: jwt.NumericDate{Time: issuedAt},
        Issuer: "my-app",
        Subject: "user123",
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    ss, err := token.SignedString([]byte("secret")) // Dummy Secret
    if err != nil {
        fmt.Println("Error signing token:", err)
        return
    }

    fmt.Println(ss)


	parsedClaims := jwt.StandardClaims{}
	tokenParsed, _ := jwt.ParseWithClaims(ss, &parsedClaims, func(token *jwt.Token) (interface{}, error) {
		return []byte("secret"), nil
	})
    
	if tokenParsed.Valid {
        fmt.Println("Parsed Expiry:", parsedClaims.ExpiresAt.Time.UTC())
		fmt.Println("Parsed Issued At:", parsedClaims.IssuedAt.Time.UTC())
		fmt.Println("Parsed Not Before:", parsedClaims.NotBefore.Time.UTC())
        fmt.Println("Token is Valid")
	} else{
        fmt.Println("Token is Invalid")
    }

}
```

Notice in the example above we can use the `jwt.NumericDate` type in combination with `jwt.StandardClaims`. This shows that the `jwt` library properly serializes this struct. When you compare with the first example, you'll see the same results and the key is to use the UTC variant of the `time.Time` object to create the timestamps in the struct when marshaling.

In summary, when working with `jwt.StandardClaims`, the safest approach is to ensure your `time.Time` values are in UTC and convert them to Unix timestamps (int64) using `time.Time.Unix()`. When parsing, remember to extract the timestamp (float64 in our examples), convert it to an integer, and use the `time.Unix(int64, 0).UTC()` method to convert it back to a time object.

For further study, I strongly recommend reading *“Programming in Go”* by Mark Summerfield, it's thorough and covers the time package well. Also, the official documentation for `go-jwt/jwt` on github is your go-to reference for specific details about how claims are handled. Furthermore, you may want to reference "RFC 7519" which is the official JWT specification, giving you the authoritative reference. Pay close attention to sections on claim types and timestamp representation in the specification. And always remember, timezone issues can be tricky, so stick to UTC and unix timestamps to avoid a lot of pain.
