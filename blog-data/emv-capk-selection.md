---
title: "emv capk selection?"
date: "2024-12-13"
id: "emv-capk-selection"
---

Alright so you're asking about EMV CAPK selection huh Been there done that Got the t-shirt probably have a few actually along with a couple of battle scars from debugging those payment flows

Ok first off let me lay down some real talk about EMV its not a walk in the park its more like a carefully choreographed dance and CAPK selection is a crucial step in that dance Its what lets the terminal figure out which public key to use for verifying the card's signature and trust me if this goes sideways you're going to have a bad time

From my past life working on embedded payment terminals I've had the joy of wrestling with this stuff I remember a particularly nasty bug back in 2015 where a subtle error in our CAPK table parsing would result in completely random transaction failures Talk about a fun debugging session spent countless hours staring at hex dumps and card logs eventually tracked it down to an off by one error in the table index It was so incredibly stupid I nearly quit programming But hey lessons learned am I right

Now about your question let's break it down into understandable pieces EMV uses a public key infrastructure (PKI) Each payment card has a certificate signed by a private key associated with a Certificate Authority (CA) This certificate contains a card's public key and the terminal must verify this certificate to confirm that the card is legit This where CAPKs come in CAPKs are the CA's public keys preloaded on the terminal Each card has an AID (Application Identifier) that tells the terminal which application it is using Each AID has an associated RID (Registered application provider Identifier) which is tied to a CA This is the way we know what CAPK to use

So the core of CAPK selection comes down to these steps:

1 The terminal reads the card's AID It gets this from the card using the SELECT command during the initial transaction setup
2 The terminal checks the RID part of the AID to find the associated CA
3 The terminal searches its internal CAPK table for a matching RID and a key that is valid for the card
4 If a match is found then the terminal uses the public key to verify the card's certificate

If the terminal does not find any CAPK to use or if a problem exists with the certificates then well transaction denied you get the point this must work well for things to work at all

Let me give you a taste of what this kind of code looks like with a few examples using pseudo-code which you can easily adapt to other languages:

Example 1 Basic CAPK lookup by RID

```c
typedef struct {
    byte rid[5];
    byte key[256];
    // add any other attributes like key index etc
} CAPKEntry;

CAPKEntry capkTable[MAX_CAPK_ENTRIES];

CAPKEntry* findCAPKByRID(byte* rid) {
    for (int i = 0; i < MAX_CAPK_ENTRIES; i++) {
        if (memcmp(capkTable[i].rid, rid, 5) == 0) {
            return &capkTable[i];
        }
    }
    return NULL;
}

// Example usage
byte cardRID[5] = {0xA0, 0x00, 0x00, 0x00, 0x04}; // Visa
CAPKEntry *foundCapk = findCAPKByRID(cardRID);
if (foundCapk != NULL) {
    // use the found capk key for the transaction
}
```

This example shows the most simple case of looking up CAPK using a RID which is part of the AID

Example 2 CAPK lookup with key index and expiration handling

```c
typedef struct {
    byte rid[5];
    byte key[256];
    byte keyIndex;
    time_t expirationDate;
    byte certHash[20];
} CAPKEntry;

CAPKEntry capkTable[MAX_CAPK_ENTRIES];

CAPKEntry* findValidCAPK(byte* rid byte keyIndex, time_t currentDate) {
    for (int i = 0; i < MAX_CAPK_ENTRIES; i++) {
        if (memcmp(capkTable[i].rid, rid, 5) == 0 &&
            capkTable[i].keyIndex == keyIndex &&
            capkTable[i].expirationDate > currentDate) {
            return &capkTable[i];
        }
    }
    return NULL;
}

// Example usage
byte cardRID[5] = {0xA0, 0x00, 0x00, 0x00, 0x04};
byte cardKeyIndex = 0x01;
time_t currentTime = time(NULL);
CAPKEntry *foundCapk = findValidCAPK(cardRID,cardKeyIndex, currentTime);
if (foundCapk != NULL) {
    // use the found capk key for the transaction
}

```

This is more realistic showing a CAPK with key index and some form of expiration this is important for managing the keys that are updated in time

Example 3 Reading the AID and getting the RID

```cpp
#include <iostream>
#include <vector>
#include <cstring>

std::vector<unsigned char> readCardAID() {
    // This is a simulation of reading card AID from the smartcard reader
    std::vector<unsigned char> aid = {0xA0, 0x00, 0x00, 0x00, 0x04, 0x10, 0x10};
    // In real world this is read using a specific smartcard library and command such as SELECT
    return aid;
}

std::vector<unsigned char> getRIDFromAID(const std::vector<unsigned char>& aid) {
    if(aid.size() < 5){
      return {};
    }
    std::vector<unsigned char> rid(aid.begin(), aid.begin() + 5);
    return rid;
}

int main() {
    std::vector<unsigned char> cardAID = readCardAID();
    std::vector<unsigned char> cardRID = getRIDFromAID(cardAID);
    for(unsigned char c : cardRID){
      std::cout << std::hex << (int)c << " ";
    }
    std::cout << std::endl;
    // You'll use cardRID to look up the capk table now using the example above
    return 0;
}
```

This example shows the reading of the AID and the extracting of the RID This is a simplified simulation for illustrative purposes in a real smart card system this requires more complex low-level communication with the card reader

Now a few important things to keep in mind during your journey with EMV CAPK selection

1 **CAPK Table Management**: You will need a way to load and update the CAPK table because things change keys expire and new ones show up I swear the CAPK updates always came at the worst possible time mostly late Friday night but thats just part of the magic of this stuff right
2 **Key Expiration**: As shown in the examples your logic needs to handle key expiration and switch to new ones before expiry You don't want your transactions failing because of outdated keys the horror
3 **Error Handling**: Lots of things can go wrong Make sure you have proper error logging and reporting so you can easily track down issues that are bound to occur at one point or another

If you are serious about diving deeper into this rabbit hole I recommend checking out the EMV specifications directly they are the definitive guide even if dense and sometimes hard to follow The EMV book 4.3 specifications are a good place to start and the ISO 7816 standard also has some good information on the cards themselves There are also few open source libraries for the EMV implementations that you could look into for implementation details

I know sometimes this stuff seems like the absolute worst especially when you get stuck on a subtle error but stick with it it’s totally doable with proper debugging and patience

And here is my one joke of the post I can't believe I'm going to actually write this what did the developer say when his code was working for the first time perfectly? "It's not a bug it's a feature that I haven't released yet" pretty bad huh

Anyway good luck hope this helps and may your CAPK selection always work on the first try
