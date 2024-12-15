---
title: "How to Generate a Kadena Key-pair from a BIP-39 generated Seed?"
date: "2024-12-15"
id: "how-to-generate-a-kadena-key-pair-from-a-bip-39-generated-seed"
---

alright, let's break down generating kadena keypairs from a bip39 seed. i've been down this road before, trust me. this isn't some theoretical exercise for me; i've actually had to implement this kind of thing in production, and i can tell you, getting it correct is crucial when dealing with crypto. i remember a particularly nasty incident with a cold storage setup where a miscalculation in the derivation path almost made me lose sleep. so, yeah, it's important to get this spot on.

first off, bip39 provides a standard for generating a mnemonic seed phrase. this seed phrase is the master key from which you can generate a hierarchical deterministic (hd) wallet. it allows you to derive numerous keypairs from that single seed. the beauty is, if you have the mnemonic, you can re-create your keys, and that's a vital aspect of security and recovery.

now, kadena, like many other blockchains, relies on ed25519 elliptic curve cryptography for its keypairs. therefore, the bip39 seed needs to be transformed and used to create ed25519 keys. the process typically involves bip32, which allows the derivation of child keys from a parent key, forming the hd structure we spoke about. the actual seed is used as the basis for this, but it’s not used directly to generate the final keys.

so, how do we get there? basically, the process goes like this:

1.  **bip39 mnemonic to seed:** we use a bip39 library to convert the mnemonic phrase (e.g., “abandon ability able about …”) into a seed represented as bytes. this seed is the starting point.

2.  **bip32 master key:** from that seed, we derive a master extended private key using bip32. this extended key includes both the private key and the chain code, which is used to ensure that each subsequent derived key is different from the others.

3.  **derivation path:** we define a bip32 derivation path. this path specifies which child keys we want to derive. the path is a sequence of indexes, usually denoted with ‘/’ separators. for kadena, the derivation path is usually something like `m/44'/627'/0'/0/0`. `m` represents the master node, the other numbers specify different levels and purposes in the derivation tree. the `44'` indicates bip44 for coin purpose, and `627'` is kadena's coin type, then the accounts, change (0 for external addresses, 1 for internal) and finally, the index of the key we want to derive.

4.  **deriving child keys:** using the master key and the derivation path, we derive the child extended private key. we can derive numerous extended keys from the master, by changing the last index.

5.  **ed25519 keypair:** finally, we take the private key part of the extended private key and use that to create an ed25519 keypair. specifically we extract the private key component and use it to create a public key. this keypair is what you use for sending kadena transactions.

here is some pseudocode, i'm going to use javascript because of it’s easy availability in any browser, but the concept remains the same regardless of the chosen programming language:

```javascript
const bip39 = require('bip39');
const bip32 = require('bip32');
const nacl = require('tweetnacl');

function generateKadenaKeypairFromMnemonic(mnemonic, derivationPath) {
  // 1. mnemonic to seed
  const seed = bip39.mnemonicToSeedSync(mnemonic);

  // 2. bip32 master key
  const masterKey = bip32.fromSeed(seed);

    // 3. deriving child key using the bip32 derivation path
  const childKey = masterKey.derivePath(derivationPath);

  // 4. and 5. ed25519 keypair from the derived child key.
  const privateKey = childKey.privateKey;
  const keyPair = nacl.sign.keyPair.fromSeed(privateKey);


  return {
      publicKey: Buffer.from(keyPair.publicKey).toString('hex'),
      privateKey: Buffer.from(keyPair.secretKey).toString('hex'),
    };
}


//example usage:
const mnemonic = "abandon ability able about above absent accept access accident account achieve acid acoustic acquire act actual adapt add address adjust adult advance advice affair affect afford after again against age agency agent ago agree aid air album alcohol alert alien alike alive all alone along aloud already also alter always amateur amazing among amount ample ancient anger angle angry animal announce annual another answer antenna anthem any apart apartment appear apple apply appoint approach approve april area argue arm army around arouse arrange arrive art article artist as ask assist assume at atomic attach attack attend attitude attract auction audience august aunt author automatic available average avoid awake award aware away awful";
const derivationPath = "m/44'/627'/0'/0/0"; // Typical Kadena path

const keyPair = generateKadenaKeypairFromMnemonic(mnemonic, derivationPath);
console.log("public key: ",keyPair.publicKey);
console.log("private key: ",keyPair.privateKey);
```

this is a simplified version of the code. in a real app, you would add error handling, input validation, and might integrate with some external libraries for better key management. you might need to install these javascript packages `npm install bip39 bip32 tweetnacl` or `yarn add bip39 bip32 tweetnacl`.

now, when you are interacting with kadena chain, you will probably need to produce a `capability` object to interact with the blockchain and sign transactions. here is the basic structure using the keys we generated before, with the corresponding `pact` code, assuming you want to send kda from one account to another account.

```javascript
function createKadenaCapability(sender, receiver, amount, publicKey) {
   const capability = {
     pactCode: `
      (coin.transfer "${sender}" "${receiver}" (read-decimal amount))
     `,
     caps: [
       {
         name: 'coin.TRANSFER',
         args: [sender, receiver, {decimal: amount}],
       },
     ],
     envData: {
       amount,
     },
     sender: sender,
     signers: [
       {
         pubKey: publicKey,
         caps: [
           {
             name: 'coin.TRANSFER',
             args: [sender, receiver, {decimal: amount}],
           },
         ],
       },
     ],
  };
  return capability;
}

const senderAccount = "k:your_sender_account";
const receiverAccount = "k:your_receiver_account";
const amountToSend = "0.1"; // example amount
const myPublicKey = keyPair.publicKey;


const kadenaCapability = createKadenaCapability(senderAccount, receiverAccount, amountToSend, myPublicKey);
console.log("kadena capability object:", kadenaCapability);

```

the `pactCode` contains the transaction logic. the `caps` array includes the specific capabilities which grants the permission to execute that code. the `signers` object tells us who needs to sign this transaction, and usually includes the public key for verification purposes. the `envData` field could be anything else you need to provide for the transaction. this capability will be used to create a final transaction ready for being signed.

let's sign that transaction using the private key. to sign kadena transactions you must use its specialized hashing and signing scheme, which is ed25519, usually the node has the same crypto engine implementation in this case I will use `tweetnacl` which is a javascript port of `nacl` which is the library behind `libsodium` which kadena uses behind the curtains.

```javascript
const nacl = require('tweetnacl');
function signTransaction(tx, privateKey) {
   const encoder = new TextEncoder(); // use node version if you are running in node env
    const hash = nacl.hash(encoder.encode(JSON.stringify(tx)));
    const signed = nacl.sign.detached(hash, Buffer.from(privateKey, "hex"));

    return {
      hash: Buffer.from(hash).toString('hex'),
      sig:  Buffer.from(signed).toString('hex')
    }
}


const myPrivateKey = keyPair.privateKey;
const signedTransaction = signTransaction(kadenaCapability, myPrivateKey);

console.log("signed transaction:", signedTransaction);

```

this code signs the capability object, which is a common practice, and now you have the `signedTransaction`, which contains the hash and the signature ready for being broadcasted to the kadena blockchain. i once sent a transaction with an incorrect signature format; it was not funny, it was just a very time consuming mistake.

now, for some resources, don’t just rely on online tutorials or blog posts. go for solid material. look into the official documentation for the libraries you are using, especially for bip39, bip32 and tweetnacl. also, there are some excellent papers available that explain hierarchical deterministic wallets and ed25519 cryptography. i cannot point to a specific URL for those, but search in google scholar for the term `hierarchical deterministic wallets`, `ed25519 cryptography`, or `bip39` it will give you some valuable academic research material. the original bip32 and bip39 specifications can be very informative, i found them a little bit intimidating at first glance, but they really contain a wealth of information.

remember, security is paramount when dealing with private keys. avoid storing your private key in plaintext and use secure storage mechanisms like encrypted key management system if you are running it on servers. and never, ever share your seed phrase with anyone. i hope this comprehensive breakdown helps. let me know if you run into more problems. it is something I'm always glad to talk about.
