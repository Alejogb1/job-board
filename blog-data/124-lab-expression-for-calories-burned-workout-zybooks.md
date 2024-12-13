---
title: "1.24 lab expression for calories burned workout zybooks?"
date: "2024-12-13"
id: "124-lab-expression-for-calories-burned-workout-zybooks"
---

Okay so you're hitting the 1.24 zybooks lab problem on calorie calculation during a workout huh been there done that let's break it down it's not rocket science but getting the expression right takes a little bit of thinking and debugging that i'm very familiar with

First off let’s recap the core of the problem because it sounds like we're talking about the zybooks calorie calculation lab specifically the one usually tagged with "1.24" i've seen different variations of these but the basic idea remains consistent we’re given some user input representing workout metrics and need to compute estimated calories burned

We need to use some data of the exercise type and the duration and some user info (like weight maybe) and it needs a calculation that looks like this

`Calories = 0.0175 * MET * WeightInKilograms * TimeInMinutes`

MET is a metabolic equivalent that is kind of a constant depending on the activity and this problem usually requires an if-else statement or switch to select which MET constant to use

Now i've been around the block with these kinds of assignments and trust me you're not alone in scratching your head at first When i was a fresh student this type of exercise threw me off initially because I was overthinking it and that's a common mistake I have seen I distinctly remember one all-nighter fueled by instant coffee trying to get this exact formula working it felt like i was in a matrix movie and the matrix was written in c++ I learned the hard way to break down the problem piece by piece now let’s get the solution right

So here's how i would tackle this starting with the most common form of implementation this assumes that you're getting weight and workout time from the user input

```cpp
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    double weight_kg;
    double time_minutes;
    string activity_type;

    cout << "Enter your weight in kilograms: ";
    cin >> weight_kg;
    cout << "Enter the activity type (Running/Basketball/Cycling): ";
    cin >> activity_type;
    cout << "Enter the workout time in minutes: ";
    cin >> time_minutes;

    double met;
    if (activity_type == "Running") {
        met = 10.0;
    } else if (activity_type == "Basketball") {
        met = 8.0;
    } else if (activity_type == "Cycling") {
         met = 6.0;
    } else {
      cout << "Invalid activity type." << endl;
      return 1;
   }
  double calories = 0.0175 * met * weight_kg * time_minutes;
  cout << fixed << setprecision(2) << calories << endl;
  return 0;
}
```
This snippet provides a straightforward example that you can compile and try it also has some basic error handling the important part is how the met variable is set using if-else and the final calorie calculation using the given formula.

This might not be the perfect code yet let's improve it more by using a switch statement and it's very good practice to learn it and using it now instead of using if-else chains

```cpp
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>

using namespace std;

int main() {
    double weight_kg;
    double time_minutes;
    string activity_type;

    cout << "Enter your weight in kilograms: ";
    cin >> weight_kg;
    cout << "Enter the activity type (Running, Basketball, Cycling): ";
    cin >> activity_type;
    cout << "Enter the workout time in minutes: ";
    cin >> time_minutes;
    
    for(char &c : activity_type) c = tolower(c); // standardize input to lowercase

    double met;
    switch (activity_type[0]) {
        case 'r': // Running
            met = 10.0;
            break;
        case 'b': // Basketball
            met = 8.0;
            break;
        case 'c': // Cycling
            met = 6.0;
            break;
        default:
           cout << "Invalid activity type." << endl;
            return 1;
    }
    double calories = 0.0175 * met * weight_kg * time_minutes;
    cout << fixed << setprecision(2) << calories << endl;
    return 0;
}
```

Here we use a switch statement which is great for selecting a case based on a variable instead of if-else chains and we also use tolower to convert the user input to lower case to prevent errors with case sensitivity and there's some basic input standardization and checking included now we are going to make a final example

And now we go for the more robust and modular solution which is better because it isolates and it can be reuse for other similar tasks and we are going to use a function instead of one main function
```cpp
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>
#include <unordered_map>

using namespace std;


double calculateCalories(double weight_kg, double time_minutes, const string& activity_type) {
    unordered_map<string, double> metValues = {
        {"running", 10.0},
        {"basketball", 8.0},
        {"cycling", 6.0}
    };
   string lower_activity = activity_type;
   for(char &c : lower_activity) c = tolower(c);
    
    if (metValues.find(lower_activity) == metValues.end()) {
        return -1.0; // Indicate error with invalid activity
    }

    double met = metValues[lower_activity];
    return 0.0175 * met * weight_kg * time_minutes;
}

int main() {
    double weight_kg;
    double time_minutes;
    string activity_type;

    cout << "Enter your weight in kilograms: ";
    cin >> weight_kg;
    cout << "Enter the activity type (Running, Basketball, Cycling): ";
    cin >> activity_type;
    cout << "Enter the workout time in minutes: ";
    cin >> time_minutes;

    double calories = calculateCalories(weight_kg, time_minutes, activity_type);

    if (calories < 0) {
       cout << "Invalid activity type." << endl;
       return 1;
    }
    cout << fixed << setprecision(2) << calories << endl;

    return 0;
}
```
This final example uses a function to calculate the calories burned it stores the MET values in an unordered map and it provides clearer error handling and improves modularity it also converts the user input activity type to lower case before using it to prevent case related errors

Some quick notes you should consider this problem may seem deceptively simple but pay attention to:

*   **Input formats**:  Make sure that the input should be as expected check your zybooks requirements for example if they give input like 10 30 running then you may need to adjust the code to handle this properly
*  **Error Handling**: Check edge cases and user input you need to handle them gracefully.
*  **Floating-point precision**: always print with a fixed precision using `fixed` and `setprecision`

Now when it comes to resources for improving your code it's more about understanding the underlying logic and coding practice than just copy and pasting some code here are my recommendations

*   **"The C++ Programming Language" by Bjarne Stroustrup**:  It is heavy it's the C++ bible really if you want to really master C++ this is the way to go there is no shortcut
*   **"Effective Modern C++" by Scott Meyers**: This is for when you already know C++ but you want to learn how to write modern better C++ it is a must read after you have some experience with C++
*   **Your zybooks materials**:  Really look into your zybooks materials many times they will tell you what you need to know you may even find a similar example there

And finally never ever underestimate the power of stepping away for a few minutes and then coming back with a fresh mind it's also a good way to avoid burnout trust me I know this very well the solution might just magically pop up in your head after a small break.

(Oh and speaking of stepping away I once spent two hours debugging a typo that was literally just a missing semicolon like seriously a semicolon that's how far my programming life has come)

Good luck on your lab and remember debugging is part of the learning process embrace it.
