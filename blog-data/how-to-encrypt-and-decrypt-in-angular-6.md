---
title: "how to encrypt and decrypt in angular 6?"
date: "2024-12-13"
id: "how-to-encrypt-and-decrypt-in-angular-6"
---

Okay so you need to encrypt and decrypt stuff in Angular 6 right Been there done that so many times its not even funny I remember my first project back in the day we were building this e-commerce platform for like underground ferret breeders (don't ask) and we needed to secure user data like addresses and breeding schedules it was a wild time

Anyway forget the ferrets let's dive into the code side of things This isnt some mystical ritual its just data transformation using algorithms its all about making your data unreadable to someone without the right key think of it like a lock and key for your digital stuff

The core thing you need to grasp is that Angular itself doesn't have built-in encryption and decryption features You'll need to pull in a library for the actual heavy lifting There are a few good contenders but my go-to has always been `crypto-js` its a solid JavaScript library that provides a wide range of cryptographic algorithms I've found its pretty robust and easy to integrate so let's focus on that

First you'll need to install it I always use npm for that

```bash
npm install crypto-js --save
```

Once that's in we can start getting our hands dirty in the code

So here's a simple example using AES encryption which is probably what you're looking for

```typescript
import * as CryptoJS from 'crypto-js';

export class MyEncryptionService {

  private secretKey = 'YourSecretKey123'; // Replace with a more complex key store securely please

  encrypt(data: string): string {
    return CryptoJS.AES.encrypt(data, this.secretKey).toString();
  }

  decrypt(encryptedData: string): string {
    try {
       const bytes = CryptoJS.AES.decrypt(encryptedData, this.secretKey);
       return bytes.toString(CryptoJS.enc.Utf8);
    } catch (error) {
      console.error("Decryption failed", error);
      return "";
    }
  }
}
```

See this isn't rocket science right The `encrypt` method takes in your string data and encrypts it with our secret key and returns the encrypted string The `decrypt` method does the opposite it takes an encrypted string uses the same key and returns the decrypted string or if something went wrong it logs the error and returns empty string

A few things to keep in mind here first the key needs to be super secure and not stored directly in your code Seriously do not store it like that above Use environment variables or a more secure mechanism like key vault services I've made this mistake once and had to explain to the ferret breeders why all their breeding data was public not fun

Second the data you're encrypting in most cases is likely JSON data so you will need to convert to it and back using `JSON.stringify` and `JSON.parse` so the service will end up looking like this:

```typescript
import * as CryptoJS from 'crypto-js';

export class MyEncryptionService {

  private secretKey = 'YourSecretKey123'; // Replace with a more complex key store securely please

  encrypt(data: any): string {
    const jsonData = JSON.stringify(data);
    return CryptoJS.AES.encrypt(jsonData, this.secretKey).toString();
  }

  decrypt(encryptedData: string): any {
    try {
      const bytes = CryptoJS.AES.decrypt(encryptedData, this.secretKey);
      const decryptedJson = bytes.toString(CryptoJS.enc.Utf8);
      return JSON.parse(decryptedJson);
    } catch (error) {
       console.error("Decryption failed", error);
       return null;
    }
  }
}
```

Now how do you use it in your Angular components Well pretty simple you inject the service and just call its methods like:

```typescript
import { Component } from '@angular/core';
import { MyEncryptionService } from './my-encryption.service'; // Change this path

@Component({
  selector: 'app-my-component',
  template: `
    <p>Original: {{ originalData | json }}</p>
    <p>Encrypted: {{ encryptedData }}</p>
    <p>Decrypted: {{ decryptedData | json }}</p>
  `
})
export class MyComponent {
  originalData = { name: 'Ferret McFerret', breed: 'Angora', dob: '01-01-2022'};
  encryptedData: string = '';
  decryptedData: any = null;

  constructor(private encryptionService: MyEncryptionService) {}

  ngOnInit() {
    this.encryptedData = this.encryptionService.encrypt(this.originalData);
    this.decryptedData = this.encryptionService.decrypt(this.encryptedData);
  }
}
```

This is your basic setup Now a few words of caution:

*   **Never ever** store your secret key directly in your client-side code This is the fastest way to get your data stolen by script kiddies Instead use environment variables and more secure options when needed
*   **Be careful what you encrypt**: You might not need to encrypt everything encrypt only the sensitive parts of your application. Overdoing it will slow down your app a lot
*   **Key management is key**: Pun fully intended If you lose your key all encrypted data is basically bricked and impossible to decrypt If someone gets hold of the key they can decrypt all data so keep your keys secure
*  **Don't invent your own cryptography**: The worst thing you could do is attempt to build your own algorithms there are teams of specialized people working in this field and they are the only ones that can build working algorithms Use tried and tested algorithms like AES or RSA

Now if you're digging deep into crypto stuff and you are serious about learning about this I recommend you check out "Handbook of Applied Cryptography" by Alfred J Menezes Paul C van Oorschot and Scott A Vanstone This is a bible for cryptography If you like a more practical approach check out "Serious Cryptography" by Jean-Philippe Aumasson This book has a more practical approach in terms of code and examples

Also if you are dealing with data in transit then I recommend you to use TLS/SSL Its pretty much a standard nowadays so there is not a really reason not to use it

I remember once we forgot to enable TLS and a "rogue ferret breeder" got access to our data I'm just joking we never had problems like that but seriously use HTTPS

So that's pretty much it for a basic encrypt decrypt in Angular 6 Remember secure key storage good algorithms and don't reinvent the wheel This is more of a starting point but should get you going. Happy coding and keep those ferrets secure!
