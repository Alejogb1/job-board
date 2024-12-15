---
title: "Is azure verifiable identity something, and can we write our own custom smart contract?"
date: "2024-12-15"
id: "is-azure-verifiable-identity-something-and-can-we-write-our-own-custom-smart-contract"
---

alright, so you're asking about azure verifiable credentials and if we can roll our own smart contracts with them, huh? i've been down this rabbit hole a few times, and it’s a pretty interesting area, so let me break down what i’ve learned and how i’d approach it.

first, let’s tackle the ‘is it something?’ part. yeah, azure verifiable credentials (vc) are definitely a thing. they're microsoft's take on decentralized identity, leveraging standards like w3c’s verifiable credentials and did (decentralized identifiers). the core idea is to give people more control over their identity data. instead of relying on centralized authorities, you get credentials issued to a digital wallet, you present them to relying parties, and they verify the claims through cryptographic means. it's like having a digital passport where only the necessary information is shared, not everything.

now, the crucial part of this that makes it useful is the underlying architecture. vc’s are built on top of a specific set of specifications that define the standard way to issue and verify credentials. this means we're not locked into microsoft’s system if we don’t want to be. we can issue credentials using microsoft’s identity platform and verify those credentials on our own infrastructure, or vice versa. this interoperability is huge.

i remember back in 2019, when i was working on a supply chain project, we needed a way to track certificates of authenticity for various components. we initially explored centralized ledgers, but the transparency and data control issues were a roadblock. then we stumbled upon the early days of verifiable credentials, and it clicked. it wasn’t fully polished back then but the core concepts were there: we could issue verifiable credentials for each component and then verify them at various points in the supply chain without any centralized point of control over the data. that first implementation wasn't using azure per se, it was a custom setup but the concepts were the same.

it helped me to conceptualize it all to think of the process as having three main steps: issuance, presentation, and verification.

*   **issuance**: this is where you, as an issuer, generate a vc and digitally sign it using your private key.
*   **presentation**: this is the wallet user sharing with the relying party a selected list of claims from the verifiable credentials stored in their wallet.
*   **verification**: this is where a relying party checks the signature of the presented verifiable credential against the public key of the issuer to verify if the document is valid.

so, yeah, azure verifiable credentials are not some abstract concept. they are a practical implementation of a very well thought of specification. it's about digital trust and self-sovereign identity.

moving onto your next question: can we write our own custom smart contracts? this is where it gets a bit nuanced. azure vc themselves *don’t directly involve smart contracts in the typical sense*. the core credential operations are handled through the verifiable credentials service and associated libraries which follow the defined standards. this means that *you don’t interact directly with a blockchain or write smart contracts that handle vc issuance or verification itself*.

however, *and this is important*, you *can* certainly *integrate* verifiable credentials into systems that *do* use smart contracts. for instance, you could use the verification process of the vc as a gatekeeper in a smart contract function. imagine this scenario:

1.  a user has a vc stating they are over 18 years old.
2.  they present that vc to a decentralized application (dapp).
3.  the dapp verifies the vc, using a library that is capable of reading a signed jws from a json document.
4.  *then*, and *only then*, the dapp’s smart contract allows the user to execute a certain transaction.

that verification step, that check that a certain claim is present and valid could be used in a custom written smart contract to perform certain actions. you are not building *the* vc process in your smart contract, but you are using the vc results to gatekeep some actions inside the smart contract.

so, no, you don't write a smart contract to issue or verify vc’s. microsoft's service and libraries do that for you. *but* you absolutely *can* use a successfully verified vc to trigger logic within your own smart contracts. this is where you get to combine the power of decentralized identity with the smart contracts.

let me give you a practical example using solidity, a language commonly used for ethereum smart contracts, and some made-up helper functions. assume i have a helper javascript class that handles the verification of a jws. let’s say i built it after reading w3c specifications.

```javascript
// simplified verification logic using a hypothetical javascript class.
class vcverifier {
    async verify(jws, issuerPublicKey, requiredClaims) {
        // this logic would perform the actual verification following w3c standards.
        const isValid = await this.performSignatureVerification(jws, issuerPublicKey);
        if(!isValid) return false;

        const claims = this.extractClaims(jws);
        for (const claim of requiredClaims) {
            if(!claims[claim]) return false;
        }
        return true
    }

    async performSignatureVerification(jws, issuerPublicKey){
          // actual signature verification logic would be here
          // (e.g. using jose library in javascript)
          return true // returns true after successful verification
    }
    extractClaims(jws) {
        //actual logic to decode the claims object
        return { "ageOver18": "true" }
    }
}

```

now a smart contract could use this to allow actions only if the claims are verified:

```solidity
pragma solidity ^0.8.0;

contract verifiableContract {
    address public owner;
    bytes32 public issuerPublicKey;


    constructor() {
        owner = msg.sender;
        // placeholder - real-world, you'd load this from a more secure location
        // this is not a realistic scenario to embed a public key
        issuerPublicKey = "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function.");
        _;
    }


    function verifyVcAndPerformAction(string memory jws) public {
        // first, i’d offload jws verification to an oracle or a server using a secure offchain api call.
        // but for example we will simulate that verification process here.
        // this is for simulation only!
        bool verified = offchainVerify(jws);
        require(verified == true, "vc verification failed");
       // if verification is successful, then perform an action
        _performAction();

    }


    function offchainVerify(string memory jws) private returns(bool){
        //simplified example of an offchain call
        //in a real project this will be done with an oracle call
         // we are calling a javascript simulation class
         vcverifier verifier = new vcverifier();
         bool isValid  = verifier.verify(jws, issuerPublicKey, ["ageOver18"]);
         return isValid;

    }


    function _performAction() private {
          //only if the vc is verified this will run
         // here is the smart contract logic after the vc is verified.
    }

    function changeOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }
}

```

in the example above,  `verifyVcAndPerformAction`  would receive the jws as an argument. in a more realistic scenario, this value would not be a simple string but more like an id that will trigger an external call to a server which will contain the jws. the `offchainVerify` function simulates a server call to a javascript class using a hypothetical `vcverifier` javascript class that does the validation, returning `true` only if the credential is valid. then if the vc is verified the `_performAction` function will execute. this means the smart contract is protected by the vc verification.

now, here’s another use case. let's say we have a token system and we want to allow certain actions only if the user is from a specific country. we issue a vc that states where the user is from. again, using a hypothetical javascript class:

```javascript

class vcverifier {
    async verify(jws, issuerPublicKey, requiredClaims) {
        // this logic would perform the actual verification following w3c standards.
         const isValid = await this.performSignatureVerification(jws, issuerPublicKey);
        if(!isValid) return false;
        const claims = this.extractClaims(jws);
        for (const claim of requiredClaims) {
            if(!claims[claim]) return false;
        }
         return true
    }
    async performSignatureVerification(jws, issuerPublicKey){
          // actual signature verification logic would be here
          // (e.g. using jose library in javascript)
           return true // returns true after successful verification
    }
    extractClaims(jws) {
         //actual logic to decode the claims object
         return { "country": "us" }
    }
}

```

and now, a smart contract which will only allow users with the “us” tag to receive tokens:

```solidity
pragma solidity ^0.8.0;

contract tokenDistribution {
    address public owner;
    bytes32 public issuerPublicKey;
    mapping(address => uint256) public tokenBalances;

    constructor() {
        owner = msg.sender;
       // placeholder - real-world, you'd load this from a more secure location
        issuerPublicKey = "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function.");
        _;
    }

    function distributeTokens(string memory jws, uint256 amount) public {
        bool verified = offchainVerify(jws);
        require(verified == true, "vc verification failed");

        //now we distribute the tokens to the user if verified
        tokenBalances[msg.sender] += amount;

    }
    function offchainVerify(string memory jws) private returns (bool) {
         //simplified example of an offchain call
         //in a real project this will be done with an oracle call
         vcverifier verifier = new vcverifier();
         bool isValid  = verifier.verify(jws, issuerPublicKey, ["country"]);
         return isValid;

    }
    function changeOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }
}
```

here, `distributeTokens` function will only award tokens if the user's vc contains the "us" country claim, after verification.

these examples are simplified, of course. in reality, you'd have to deal with things like decentralized storage for public keys, more secure mechanisms for oracle calls and proper error handling. but these should give a practical picture of how verifiable credentials are used in the context of a smart contract. it’s not about replacing the smart contract itself. it’s about augmenting the capabilities by adding identity and trust logic.

so, in summary, azure verifiable credentials are definitely 'something'. they offer a powerful and standards-based approach to digital identity. and while you don't write smart contracts that *directly* handle vc issuance or verification, you can absolutely integrate verified vc data into your own smart contracts in very useful and flexible ways. it’s a powerful combination if you know how to leverage it. it is not like having a chicken that lays gold eggs, you still need to put in the hard work.

if you're interested in delving deeper, i'd recommend looking into the w3c's verifiable credentials data model specification. understanding the core concepts of the w3c standards is key. also research the jose standards, which deal with how to sign and encode json web tokens (jwt) as that’s the main underlying standard for verifiable credentials. there is also the ‘mastering ethereum’ book, if you want to learn more about smart contracts.

hope this clears things up. let me know if you have more questions!
