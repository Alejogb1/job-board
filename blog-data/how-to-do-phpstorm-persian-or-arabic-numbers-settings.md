---
title: "How to do PhpStorm Persian or Arabic numbers settings?"
date: "2024-12-15"
id: "how-to-do-phpstorm-persian-or-arabic-numbers-settings"
---

hey there, i saw your question about getting phpstorm to play nice with persian or arabic numerals, and i totally get where you're coming from. it's one of those seemingly small things that can be incredibly annoying when it doesn't work the way you expect it to, especially when you're dealing with different locales. i've been there, believe me. spent a frustrating week back in 2015 when a client asked me to build a simple web form with arabic inputs. i thought i had it all nailed down only to see that numbers were rendered as gibberish. it was a mess.

so, let's break this down. the issue isn't necessarily phpstorm itself being "bad" at handling these scripts. it's more about the underlying settings related to font rendering and encoding. phpstorm, like most text editors, relies on the operating system's configuration to correctly display characters. when things go wonky, it's often a case of the editor not picking the correct font, not understanding what encoding is going on, or a combination of both.

first things first, let's talk about fonts. make sure your chosen font supports arabic and/or persian characters. many default fonts in operating systems don't have glyphs for these. if that’s the case it won't display them correctly. monospaced fonts like 'dejavu sans mono' or 'firacode' are generally good candidates, though 'arial unicode ms' that comes with windows is a common choice too. if i were you, i'd try several options until one displays the characters properly. you usually change this in phpstorm preferences, 'editor' -> 'font', it should be pretty obvious there.

the next piece of the puzzle is encoding. you want your code files to be saved with the correct character encoding. in the vast majority of cases, this would be 'utf-8'. it's the de-facto standard for text files nowadays, and it's what phpstorm uses as the default. you can usually see or change your encoding via the file menu, usually, the encoding is specified in a sub menu like 'file encoding'. in the status bar at the bottom right, there’s also a quick way to view it or change it on the fly. check your encoding because if files are saved with wrong encoding it's highly likely they will get corrupted or badly displayed.

now, the tricky part. sometimes, despite setting up the font and encoding correctly, phpstorm still might render persian/arabic digits as latin digits or reverse their direction. this happens because of something called 'contextual rendering', which is what makes sure the glyphs are positioned and shaped correctly. that is not a bug, just how that particular system is programmed. here's where we often see that old chestnut, the dreaded "european numerals" problem.

we will work around this contextual rendering by manipulating the number strings that we use. that might seem counterintuitive, but this has to be done if you wish to display the numerals correctly.

here's a function that will convert standard western numerals to persian numerals:

```php
<?php
function convert_to_persian_numerals($number) {
  $persian_digits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  return str_replace(range(0, 9), $persian_digits, $number);
}

// Example usage:
$my_number = 12345;
$persian_number = convert_to_persian_numerals($my_number);
echo $persian_number; // Output: ۱۲۳۴۵
?>
```

and here's an example to convert to arabic numerals:

```php
<?php
function convert_to_arabic_numerals($number) {
  $arabic_digits = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'];
  return str_replace(range(0, 9), $arabic_digits, $number);
}

// Example usage:
$my_number = 67890;
$arabic_number = convert_to_arabic_numerals($my_number);
echo $arabic_number; // Output: ٦٧٨٩٠
?>
```
you will see it doesn't seem hard to convert the numerals to the correct format with these simple code snippets, we just have to replace each latin digit with the corresponding persian/arabic digit character.

and here's the code in javascript. remember though you need to implement this on the client side if you want this conversion on the front end of your project.

```javascript
function convertToArabicNumerals(number) {
  const arabicDigits = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'];
  return String(number).replace(/[0-9]/g, (w) => arabicDigits[parseInt(w)]);
}

// Example usage:
let myNumber = 98765;
let arabicNumber = convertToArabicNumerals(myNumber);
console.log(arabicNumber); // Output: ٩٨٧٦٥
```

as a little joke... i think dealing with these font issues is like trying to explain pointers to a new developer, sometimes it makes perfect sense on paper but the actual implementation makes you want to bang your head against the wall.

remember, phpstorm is only interpreting the text it receives. the core problem often lies within how the operating system renders the glyphs and how web browsers render the text when you send output from your php application. if you want to dive deeper into how text rendering works, i highly suggest you take a look at the "unicode standard" documentation which goes deep into how each character is defined, how different scripts are encoded, and how text direction works. it's a fairly technical read but will give you the foundational knowledge to understand these tricky issues. also, "programming with unicode" by victor stribi will provide you very good insight into the common issues in dealing with unicode character sets.

also, if you’re serving html to the browser, ensure that the correct 'content-type' header with 'charset=utf-8' is specified. otherwise, the browser might guess the encoding and get it wrong, and it would result in gibberish or incorrect characters in the web page displayed. it is very common to see this happening. this should be set in your web server config or you can explicitly set it in your php files using the function 'header()'. here’s an example:

```php
<?php
header('content-type: text/html; charset=utf-8');
// rest of your code
?>
```
that 'header()' call should be called before any html output in your page. this way the browser knows exactly what charset it is receiving.

to recap, make sure you set the right font in phpstorm’s settings to a font that includes the glyphs you want, make sure your files are saved as 'utf-8', and you convert the numbers before displaying them. if you're dealing with displaying this in a browser, ensure that you provide the correct charset in the response header. this should solve the vast majority of issues in rendering persian/arabic numbers in your phpstorm environment. i've spent countless hours troubleshooting issues with text and encoding, so these tips should set you on the right path. it's not always a pretty process, but once you understand how the system works it becomes a little bit easier. if you run into more specifics let me know, i’m happy to assist.
