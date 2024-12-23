---
title: "unit testing mfc ui applications?"
date: "2024-12-13"
id: "unit-testing-mfc-ui-applications"
---

so unit testing MFC UI yeah been there done that got the t-shirt And probably a few scars too Let's dive in because it's not exactly sunshine and rainbows is it

Look MFC I gotta say it's old school It's like coding in the Jurassic era Sometimes I think dinosaurs left better documentation Seriously though this whole UI intertwined with logic thing it’s a classic recipe for testing hell I remember my first big MFC project back in oh gosh 2008-ish it was a monster of a thing it had more dialogs than a therapist's office and trying to untangle the business logic from the UI spaghetti was like trying to solve a Rubik's Cube blindfolded I learned the hard way that directly testing UI elements is a nightmare You're basically playing whack-a-mole with Windows messages and window handles trust me it's not a good time

So first thing first you’ve gotta embrace the model-view-controller or a variation of it I know MFC doesn't scream MVC but hey we gotta work with what we have My go-to approach is to extract the core logic out of the dialogs and place it in separate classes These classes are like the engine under the hood they don't care about Windows messages they just do the math the manipulation the processing whatever it is they need to do This separation lets you test that logic in isolation without all the UI baggage

Here's a basic example let's say you’ve got a dialog that does some complex calculations based on user input your MFC dialog code might look something like this

```cpp
// MFC Dialog Class CMyDialog.cpp
void CMyDialog::OnCalculateButtonClicked()
{
    CString inputStr;
    GetDlgItemText(IDC_INPUT_EDIT, inputStr);
    int inputVal = _ttoi(inputStr);

    int result = (inputVal * 2) + 10;

    CString resultStr;
    resultStr.Format(_T("%d"), result);
    SetDlgItemText(IDC_OUTPUT_EDIT, resultStr);
}

```

 you see the problem right That calculation is buried in the UI class This makes it a pain to test Now here’s how you extract it

```cpp
// Calculator.h
class Calculator
{
public:
    int Calculate(int input);
};

// Calculator.cpp
int Calculator::Calculate(int input)
{
    return (input * 2) + 10;
}

```

And your modified dialog code looks like this

```cpp
// MFC Dialog Class CMyDialog.cpp Modified
#include "Calculator.h"
void CMyDialog::OnCalculateButtonClicked()
{
    CString inputStr;
    GetDlgItemText(IDC_INPUT_EDIT, inputStr);
    int inputVal = _ttoi(inputStr);

    Calculator calculator;
    int result = calculator.Calculate(inputVal);

    CString resultStr;
    resultStr.Format(_T("%d"), result);
    SetDlgItemText(IDC_OUTPUT_EDIT, resultStr);
}

```

Much better right Now that `Calculator` class is totally independent and you can test it easily with any testing framework you like I tend to lean towards googletest for C++ It’s solid and gets the job done Look a simple test looks like this

```cpp
// CalculatorTests.cpp
#include "gtest/gtest.h"
#include "Calculator.h"

TEST(CalculatorTest, BasicCalculation) {
    Calculator calculator;
    int result = calculator.Calculate(5);
    EXPECT_EQ(result, 20);
}

TEST(CalculatorTest, AnotherCalculation) {
  Calculator calculator;
  int result = calculator.Calculate(10);
  EXPECT_EQ(result, 30);
}
```

This approach lets you test your logic thoroughly without getting bogged down in MFC minutia You focus on the core functionality that matters

Look this pattern is about isolating what you want to test and ensuring you aren't trying to unit test the whole application as a monolithic block If it makes you feel any better I’ve had my fair share of late nights debugging why the value I thought was in an edit box was actually something completely different It turns out it was some kind of Windows message event queue issue but I digress The key here is isolating the logic from the UI

Sometimes you might have to deal with functions that make direct calls to MFC like `AfxGetApp` for whatever reason Now you cant easily instantiate that in your tests so you end up with a bit of mocking In C++ we don't have fancy frameworks like in Java for that but you can use some abstraction techniques to avoid tight coupling in such scenarios

For example lets say your class needs to get a settings path using `AfxGetApp()->GetProfileString` instead of calling that directly create an interface like this

```cpp
// AppSettingsProvider.h
class AppSettingsProvider
{
public:
    virtual ~AppSettingsProvider() {}
    virtual CString GetSettingsPath() = 0;
};


```

and a concrete implementation for your application like this

```cpp
// DefaultAppSettingsProvider.h
#include "AppSettingsProvider.h"
class DefaultAppSettingsProvider : public AppSettingsProvider
{
public:
    CString GetSettingsPath() override;
};
// DefaultAppSettingsProvider.cpp
#include "stdafx.h"
#include "DefaultAppSettingsProvider.h"

CString DefaultAppSettingsProvider::GetSettingsPath(){
  return AfxGetApp()->GetProfileString(_T("Settings"), _T("Path"), _T(""));
}

```

Now your class uses this interface instead of directly calling MFC like so

```cpp
// MyClass.h
#include "AppSettingsProvider.h"

class MyClass {
 public:
  MyClass(AppSettingsProvider* settingsProvider);
  void DoSomething();
 private:
  AppSettingsProvider* m_settingsProvider;
};


// MyClass.cpp

#include "MyClass.h"
#include <iostream>

MyClass::MyClass(AppSettingsProvider* settingsProvider)
: m_settingsProvider(settingsProvider){}

void MyClass::DoSomething() {
  CString path = m_settingsProvider->GetSettingsPath();
  std::cout << "Settings path: " << path << std::endl;
}
```
Then for your tests you can create a mock implementation like this

```cpp
// MockAppSettingsProvider.h

#include "AppSettingsProvider.h"
class MockAppSettingsProvider : public AppSettingsProvider
{
public:
    CString GetSettingsPath() override;
    void setPath(CString p) {path = p;}
private:
    CString path;
};

// MockAppSettingsProvider.cpp
#include "MockAppSettingsProvider.h"
CString MockAppSettingsProvider::GetSettingsPath()
{
  return path;
}


```

Now in your test you inject the `MockAppSettingsProvider` and you are completely detached from MFC You have full control

```cpp
//MyClassTests.cpp

#include "gtest/gtest.h"
#include "MyClass.h"
#include "MockAppSettingsProvider.h"

TEST(MyClassTest, TestSettingsPath) {
  MockAppSettingsProvider mockProvider;
  mockProvider.setPath(_T("c:\\testsettings"));
  MyClass myClass(&mockProvider);
  myClass.DoSomething();

  //you can check the logs or do your own thing

  //Here we are not really testing the method directly since we are not using the return of the method but instead we can focus on the methods
  //doing a different test focusing on the method itself will need a return type but as an example this works ok
}
```

This approach is more about isolating units and focusing on the logical correctness of each piece rather than wrestling with the UI elements directly I know it feels like a lot of extra work but trust me it’s worth it in the long run Because I've had nightmares of having 1000 lines of MFC dialog classes not well tested and those memories are not pleasant

A word on resources because we don’t do random internet links here Look for books on clean code design and unit testing in C++ I recommend "Working Effectively with Legacy Code" by Michael Feathers it's not MFC specific but it will give you strategies to approach existing codebases and make them testable. "Test Driven Development" by Kent Beck is also a great read although it has Java examples the concepts are applicable here. These resources provide more detailed guidance and approaches than I could possibly cover here and I’ve found myself going back to them time and time again over my career

Oh and one joke real quick Why was the MFC app always tired Because it had too many windows open.

that's the gist of it Unit testing MFC isn't a walk in the park but it is doable with a bit of planning and some architectural changes Is it going to be easy No but is it worth it Definitely because otherwise you're signing up for constant debugging hell trust me I know.
