---
title: "discretionary data from magnetic strip credit card how to parse?"
date: "2024-12-13"
id: "discretionary-data-from-magnetic-strip-credit-card-how-to-parse"
---

 so magnetic stripe data parsing yeah I've been there done that probably more times than I care to admit Let's just say early days of embedded systems and point of sale terminals were a wild west of encoding standards and lack of proper documentation

First things first you gotta understand what you're actually dealing with A magnetic stripe card usually has three tracks Track 1 Track 2 and Track 3 but most of the time you're only interested in Track 1 and Track 2 Track 3 is less common and usually for internal use of financial institutions

Track 1 is typically longer and contains alphanumeric data it follows an encoding called ISO/IEC 7813 This track usually holds data like the cardholder's name the account number and the expiry date but not always It can also have what are called discretionary data which is what you're asking about This discretionary data is highly variable and can contain anything the issuing bank or organization decides to include This is where things get messy and you'll often find a mix of characters and non-standard fields

Track 2 is shorter and contains numeric data mostly following ISO/IEC 7813 it typically holds the Primary Account Number PAN the expiry date and discretionary data much like track 1 It also follows the ISO standard but has its own set of requirements.

Now the fun part parsing this beast So you've got the raw data in a string or byte array format it's usually a mix of alphanumeric and sometimes some special control characters Now the critical part is to understand the field separators and how to differentiate actual data from the discretionary field and delimiters

For Track 1 the general format is like this:

```
%B[PAN]^[NAME]/^[EXPIRY DATE]^[SERVICE CODE]^[DISCRETIONARY DATA]?
```

For Track 2 it's more like:

```
;[PAN]=[EXPIRY DATE][SERVICE CODE][DISCRETIONARY DATA]?
```

The `B` in the track 1 prefix marks it as a track 1 reading which is followed by the PAN then `^` is a field separator that you'll find a lot of in track 1. `;` marks track 2 starting point and the `=` separator. Notice that discretionary data is after everything else

The question mark means that discretionary data might not be there which is very common

Now the most confusing part is discretionary data It’s not standardized so it really depends on the issuer. You might find things like card sequence number CVV or CVC data extra security information or even just random padding to fill space. You need to figure out the format of that part based on how the data was encoded in the first place. And yes sometimes you need to reverse engineer this from the data that you have which is what I have done in my past. It’s pretty fun but pretty tiring too

 let me show some python code this is not a complete solution just an outline

```python
def parse_track1(track1_data):
    try:
        if not track1_data.startswith('%B'):
            return None # or raise an exception

        parts = track1_data[2:].split('^')
        pan = parts[0]
        name = parts[1]
        expiry_date = parts[2]
        service_code = parts[3]
        discretionary_data = parts[4] if len(parts) > 4 else None

        return {
            "pan": pan,
            "name": name,
            "expiry_date": expiry_date,
            "service_code": service_code,
            "discretionary_data": discretionary_data,
        }
    except IndexError:
        return None  # Malformed data
```

And the Track 2 parser:

```python
def parse_track2(track2_data):
    try:
        if not track2_data.startswith(';'):
            return None

        parts = track2_data[1:].split('=')
        pan = parts[0]
        remaining = parts[1]
        expiry_date = remaining[:4]
        service_code = remaining[4:7]
        discretionary_data = remaining[7:] if len(remaining) > 7 else None


        return {
            "pan": pan,
            "expiry_date": expiry_date,
            "service_code": service_code,
            "discretionary_data": discretionary_data,
        }

    except IndexError:
        return None  # Malformed data
```

Keep in mind that expiry date is usually YYMM format service code is three digits and both PAN can be variable in length you really need to check your requirements there

The hard part is decoding the discretionary data which I cannot provide a code example for since I do not know the format This is highly specific to the issuer and how they encoded that extra field and you will need to figure it out based on the available documentation or reverse engineering.

So how did I learn this the hard way? Well back in the day I was working on a project for an alternative payment system that used custom cards We had access to the ISO standard documents (ISO/IEC 7813) which gives you an overview of the tracks structure but the discretionary data format was never really clearly defined by the card issuer It was a classic case of “here is the card data do what you want with it”. We actually ended up reverse engineering it by comparing a lot of card samples and their responses. It was a fun time especially when we were debugging in a hurry during the QA phase at night. I can tell you that the most common discretionary data was often just zero-padded fields or a simple sequence counter and a card verification value which is unique for each card

And now the joke that was requested what do you call a software developer on a cruise ship? A code explorer! yeah I know it’s terrible but its what I got for you

So yeah that’s pretty much it for magnetic stripe card data parsing. It’s not super difficult once you know the basic structure. The discretionary data part is the real challenge and that part is unique to each issuer so there's no universal solution. You’ll always need a decoder for that part or reverse engineer it.

Some resources you should check out if you really want to get deep into this:

*   **ISO/IEC 7813**: This is the bible for magnetic stripe cards It defines the physical characteristics and data encoding This is a must read. You can find this document at a standards organization or library. You might have to pay for access but it’s worth it if you work with it often
*   **Various EMV Specifications**: Although these specs focus on chip cards they also touch on data elements and processing related to track data. EMVCo website has all that info
*   **"The Smart Card Handbook" by Wolfgang Rankl and Wolfgang Effing**: This is a great book for understanding not just magnetic stripe cards but also chip cards and related technologies It has a good section on magnetic stripe technology if you can find it at a library
*   **Online forums**: There are a few forums dedicated to smart card and payment technologies where you might find someone who has tackled a specific issuer’s magnetic stripe implementation if you are really lucky

The general rule is that discretionary data is always a "figure it out yourself" sort of problem. If you have more questions or details about the discretionary data I might be able to provide more guidance. Good luck and happy parsing!
