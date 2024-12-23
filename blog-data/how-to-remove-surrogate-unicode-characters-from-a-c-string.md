---
title: "How to remove surrogate Unicode characters from a C++ string?"
date: "2024-12-23"
id: "how-to-remove-surrogate-unicode-characters-from-a-c-string"
---

Alright, let's tackle the rather thorny issue of removing surrogate unicode characters from c++ strings. I've run into this quite a few times over the years, particularly when dealing with data originating from less-than-perfect encoding situations. It's one of those problems that seems trivial on the surface, but can quickly become a frustrating detour if you don't have a solid plan.

The fundamental problem, as you probably know, lies in the nature of surrogate pairs in utf-16 encoding. Unicode, to represent the vast array of characters across various languages and symbols, utilizes a system of codepoints. While most commonly used characters can fit into a single 16-bit unit, some require two, forming a surrogate pair. A lone surrogate, without its partner, is invalid unicode. It's essentially a broken piece of the puzzle, and it manifests as those often-unwanted symbols when displayed or processed incorrectly.

The goal here isn’t just to delete the surrogate, but to do it without introducing other errors, like inadvertently cutting off valid multi-byte characters or causing encoding issues downstream. The key is to process the string, recognizing those individual surrogate code units, and to handle them appropriately by removing them only when they stand alone.

The standard c++ library offers a suite of tools that, in combination, can achieve this, but it’s not a single-step process. We’ll be mainly leveraging the `<string>` and `<codecvt>` headers, plus some logical checks for identifying surrogate codepoints. Let's look at an example.

```c++
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <algorithm>

std::string removeSurrogates(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wideInput;
    try {
      wideInput = converter.from_bytes(input);
    } catch(const std::range_error& e) {
       // Input string contains invalid utf-8 character.
       // Return input string as-is if we cannot interpret.
       return input;
    }


    std::wstring result;
    for (size_t i = 0; i < wideInput.size(); ++i) {
        wchar_t ch = wideInput[i];

        if ((ch >= 0xd800 && ch <= 0xdbff) || (ch >= 0xdc00 && ch <= 0xdfff))
        {

            bool is_leading_surrogate = (ch >= 0xd800 && ch <= 0xdbff);
            bool is_trailing_surrogate = (ch >= 0xdc00 && ch <= 0xdfff);


            if (is_leading_surrogate) {
                  //lookahead for a trailing surrogate.
                  if (i + 1 < wideInput.size() && wideInput[i + 1] >= 0xdc00 && wideInput[i+1] <= 0xdfff )
                  {
                    //Valid pair, skip.
                    result.push_back(ch);
                    result.push_back(wideInput[i+1]);
                    i++;
                    continue;

                  }
                  //Lone leading, discard
                  continue;

                }
                // Lone Trailing, Discard
                if(is_trailing_surrogate) {
                   continue;
                }



        }
        else {
            result.push_back(ch);
        }

    }
    try {
         return converter.to_bytes(result);
    }
     catch(const std::range_error& e)
    {
        //Conversion to utf8 has failed, probably due to invalid char.
        return input;
    }
}

int main() {
    std::string testString = "Hello \ud800 world\ud800\udc00 test \udfff!";
    std::string cleanString = removeSurrogates(testString);
    std::cout << "Original: " << testString << std::endl;
    std::cout << "Cleaned: " << cleanString << std::endl;
    return 0;
}
```

This first example processes the string using a wide string intermediate representation.  It's a common strategy when dealing with variable-width character encodings.  The `std::wstring_convert` allows us to convert from utf-8 (represented by a `std::string`) to wide characters (represented by `std::wstring`). The code then iterates through the wide string, checks if a character falls within the surrogate range (0xd800-0xdfff), checks if a leading surrogate is followed by a trailing surrogate and, discards any lone surrogate characters. Valid pairs are retained as they represent valid code points.

This approach has served me well across many projects. It’s critical to understand the encoding that your input string uses; I usually start by inspecting the data source or API documentation very carefully.

Now, let's explore a slightly different strategy, one that operates directly on the byte stream of a `std::string` without converting to `wstring`. This approach can avoid potential overhead if you already know the encoding to be UTF-8, and it's a good example of how to optimize when processing a high volume of data. This method requires us to do a little more work to interpret the character bytes directly, based on utf-8 character encoding rules.

```c++
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

std::string removeSurrogatesDirect(const std::string& input) {
    std::vector<unsigned char> result;
    for (size_t i = 0; i < input.size();) {
        unsigned char c = input[i];
        if (c <= 0x7f) {  // 1-byte utf-8
            result.push_back(c);
            ++i;
        } else if ((c & 0xe0) == 0xc0) { // 2-byte utf-8
            if(i + 1 < input.size()){
            unsigned char c2 = input[i + 1];
            result.push_back(c);
            result.push_back(c2);
            i+=2;
            }else{
                i++;
            }


        } else if ((c & 0xf0) == 0xe0) { // 3-byte utf-8
             if(i + 2 < input.size()){
                unsigned char c2 = input[i + 1];
                unsigned char c3 = input[i + 2];
            //Check for surrogate in utf-8 encoding
            int codepoint = ((c & 0x0f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f);

             if ((codepoint >= 0xd800 && codepoint <= 0xdbff) ) {
                  if(i+3 < input.size() && ((input[i+3] & 0xf0) == 0xe0)){
                    int codepoint2 = ((input[i+3] & 0x0f) << 12) | ((input[i+4] & 0x3f) << 6) | (input[i+5] & 0x3f);
                    if(codepoint2 >= 0xdc00 && codepoint2 <= 0xdfff){
                        //Valid pair
                         result.push_back(c);
                         result.push_back(c2);
                         result.push_back(c3);
                        i+=3;
                        continue;
                      } else {
                        i+=3;
                        continue;
                    }
                   } else {
                    i +=3;
                    continue;
                  }
                }
                  if ((codepoint >= 0xdc00 && codepoint <= 0xdfff) ) {
                    i+=3;
                    continue;
                  }


             result.push_back(c);
             result.push_back(c2);
             result.push_back(c3);
                i += 3;

              }
              else {
                  i++;
              }


        } else if ((c & 0xf8) == 0xf0) { // 4-byte utf-8
               if (i+3 < input.size()){

                result.push_back(input[i]);
                result.push_back(input[i+1]);
                result.push_back(input[i+2]);
                result.push_back(input[i+3]);
                i+=4;

                }
               else{
                    i++;
                }
        } else { // Invalid byte sequence, skip.
            ++i;
        }
    }
    return std::string(result.begin(), result.end());
}

int main() {
    std::string testString = "Hello \ud800 world\ud800\udc00 test \udfff!";
    std::string cleanString = removeSurrogatesDirect(testString);
    std::cout << "Original: " << testString << std::endl;
    std::cout << "Cleaned: " << cleanString << std::endl;
    return 0;
}
```

This second approach processes utf-8 characters byte-by-byte, using the leading bits of each byte to determine how many bytes to consume to form a valid character. This can be faster as it does not require a conversion between string types. Notice how the code checks for the surrogate ranges after decoding an utf-8 character and then it discards them as needed.

Finally, if you are working with extremely large strings, or if performance is critical, an approach based on iterators and `std::remove_if` could also be considered, along with a custom predicate to identify and remove the invalid surrogates. It may offer more optimization potential if your use case is very specific.

```c++
#include <iostream>
#include <string>
#include <algorithm>
#include <codecvt>
#include <locale>
#include <vector>

std::string removeSurrogatesIterative(const std::string& input) {

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::wstring wideInput;
    try {
      wideInput = converter.from_bytes(input);
    } catch(const std::range_error& e) {
       return input;
    }
    std::vector<wchar_t> result(wideInput.begin(), wideInput.end());


   auto it = std::remove_if(result.begin(), result.end(),
    [&result](wchar_t ch)
     {
        if ((ch >= 0xd800 && ch <= 0xdbff) ) {
                auto pos = std::find(result.begin(), result.end(), ch);
                if (pos != result.end() )
                {
                     auto next = pos +1;
                    if(next != result.end()){
                          if(*next >= 0xdc00 && *next <= 0xdfff) {
                           return false;

                          } else {
                             return true;
                          }
                    }
                   else {
                       return true;
                    }


                }
                return true;
        }


        if (ch >= 0xdc00 && ch <= 0xdfff)
        {
            return true;
        }
        return false;
    });

   result.erase(it, result.end());

    std::wstring cleanedWideString(result.begin(), result.end());
     try {
         return converter.to_bytes(cleanedWideString);
    }
     catch(const std::range_error& e)
    {
        return input;
    }


}

int main() {
    std::string testString = "Hello \ud800 world\ud800\udc00 test \udfff!";
    std::string cleanString = removeSurrogatesIterative(testString);
    std::cout << "Original: " << testString << std::endl;
    std::cout << "Cleaned: " << cleanString << std::endl;
    return 0;
}
```

This iterator-based method uses a `std::remove_if` and a custom lambda expression to inspect and remove the lone surrogate characters from the wide string representation.  It relies on a predicate that is more involved, and it might be beneficial when dealing with complex processing logic, because lambda expressions can encapsulate complex algorithms more succinctly.

For further study, the *Unicode Standard* published by the Unicode Consortium is the definitive source, particularly regarding encoding and surrogate pairs. Also, "Programming with Unicode" by Victor Stribjov is also an excellent reference, providing thorough coverage of the intricacies of character encodings and handling them programmatically.  Lastly, exploring the details of `std::codecvt`, `std::wstring`, and related functions on cppreference.com can solidify understanding the tools that C++ provides.

In my experience, the best approach often depends on context. If it’s an isolated incident, a simple method is fine. But if performance is a bottleneck or complex encoding rules are involved, the time spent on crafting a more precise and optimized solution pays back dividends.
