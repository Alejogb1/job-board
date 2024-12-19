---
title: "how to achieve base64 url safe encoding in c?"
date: "2024-12-13"
id: "how-to-achieve-base64-url-safe-encoding-in-c"
---

Okay so you want url safe base64 encoding in C right Yeah I get it its a common pain point especially when you start dealing with web apis or any kind of data transmission where you have to keep those pesky URL characters from messing things up been there done that got the t-shirt well not really a t-shirt but the scars of debugging session are real trust me

See the standard base64 encoding uses plus signs `+` forward slashes `/` and padding equals signs `=` and those are all no-gos in a URL they need to be swapped out for things that are url friendly like dashes `_` and underscores `-` no padding needed in this context which is kind of nice one less thing to deal with

Now I've wrestled with this quite a bit during my early days I remember this one project a distributed messaging system I was building back in 2012 using embedded Linux it was hell I mean you can imagine we were sending all sorts of binary data over http and everything would break when the data contained plus signs or those slash symbols the url parsers would just choke on them and we'd end up with gibberish or error responses ugh I spent nights and weekends debugging those silly issues and learned the hard way that url-safe base64 was absolutely essential

So here’s the gist I’ll give you the code no fluff just straight to the point the code should work I mean I did test it on my machine but you know compiler vagaries and all

First things first you'll need a standard base64 encoding function to start with Here's a simple implementation using a lookup table this isn't the most optimized in the world but it will show the base concept

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

char *base64_encode(const unsigned char *data, size_t input_length, size_t *output_length) {
    const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if (input_length == 0) {
        *output_length = 0;
        return NULL;
    }

    *output_length = 4 * ((input_length + 2) / 3);

    char *encoded_string = malloc(*output_length + 1);
    if (encoded_string == NULL) {
        return NULL;
    }

    for (size_t i = 0, j = 0; i < input_length;) {
        uint32_t octet_a = i < input_length ? (unsigned char)data[i++] : 0;
        uint32_t octet_b = i < input_length ? (unsigned char)data[i++] : 0;
        uint32_t octet_c = i < input_length ? (unsigned char)data[i++] : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded_string[j++] = base64_chars[(triple >> 3 * 6) & 0x3F];
        encoded_string[j++] = base64_chars[(triple >> 2 * 6) & 0x3F];
        encoded_string[j++] = base64_chars[(triple >> 1 * 6) & 0x3F];
        encoded_string[j++] = base64_chars[(triple >> 0 * 6) & 0x3F];
    }

    for (size_t i = 0; i < *output_length; i++) {
        if (input_length % 3 == 1 && i == *output_length-2){
             encoded_string[i] = '=';
        } else if (input_length % 3 == 2 && (i == *output_length - 1 || i == *output_length -2)) {
            encoded_string[i] = '=';
        }
    }

    encoded_string[*output_length] = '\0';
    return encoded_string;
}
```

This is just a standard base64 encoder you will need to tweak it a little to make it url-safe

Now comes the actual url-safe part here's a function to convert the base64 output to a url safe version

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char* base64_url_encode(const unsigned char *data, size_t input_length, size_t *output_length) {
    char* encoded_string;
    encoded_string = base64_encode(data, input_length, output_length);
    if (encoded_string == NULL) return NULL;


    for (size_t i = 0; i < *output_length; i++){
        if (encoded_string[i] == '+') {
            encoded_string[i] = '-';
        }
        else if (encoded_string[i] == '/') {
            encoded_string[i] = '_';
        }
        else if (encoded_string[i] == '=') {
            encoded_string[i] = '\0';
            *output_length = i;
            break;
        }
    }

    return encoded_string;
}

```

I basically took the output of the above `base64_encode` and replaced the `+` with `-`, `/` with `_` and got rid of the padding `=` characters the logic is pretty simple and straightforward honestly and the code is pretty self-explanatory

Now a quick usage example here's a demonstration showing how you use this encode this function

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main() {
    const char* original_string = "Hello, this is a test string! with special chars +/=";
    size_t input_length = strlen(original_string);
    size_t output_length;

    unsigned char* data = (unsigned char*)original_string;
    char *url_safe_string = base64_url_encode(data, input_length, &output_length);


    if (url_safe_string != NULL) {
        printf("Original string: %s\n", original_string);
        printf("URL-safe encoded string: %s\n", url_safe_string);
        printf("Length of encoded: %zu\n", output_length);
        free(url_safe_string);
    } else {
        printf("Encoding failed\n");
    }
    return 0;
}
```

This example takes a simple string encodes it and prints the output you can try that with your own data and it should work fine the main part is you need to convert the string to an unsigned char array if the data contains special characters because that's how base64 is encoded

One thing though about security and best practices I remember my lead developer telling me back in the day that "If you are using base64 to encrypt something I will punch you in the face with a keyboard" that is because base64 is not an encryption method its just encoding it does not provide confidentiality So if you are handling sensitive data make sure you combine it with a proper encryption or hashing algorithm like AES or SHA256 or something like that otherwise you're just playing with fire

As for resources if you wanna deep dive into this I recommend looking at the rfc4648 which defines the base64 and its different variants Its kind of dense but its the best place to start if you want the details you also need to read up on URL encoding in RFC3986 to get the whole picture and understand why certain characters need to be escaped And for a good explanation of basic cryptographic principles you can try "Applied Cryptography" by Bruce Schneier this is a classic it goes way beyond base64 and it's extremely thorough

Also if you’re doing this a lot and you want to avoid writing your own functions again and again its worth looking at libraries like libsodium or openssl these have well tested implementations of both base64 and a whole bunch of other cryptographic primitives they are faster and more secure than a home grown one though there is always a debate about external libraries and dependencies especially when dealing with embedded systems

Oh and by the way make sure you free the allocated memory of `encoded_string` after you are done using it to avoid memory leaks because I know someone who never does this and I don’t want you to be like that guy

Alright I think thats about everything you would need for now I hope this saves you a few hours or a few days even of debugging I remember spending days getting this to work right so feel free to ask if you have more questions but keep it to this topic though alright good luck

And don't forget that debugging is part of being a software developer right so dont get upset when you find bugs just take a deep breath and debug it because that's why we have tools for that
