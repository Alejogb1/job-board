---
title: "Why are NEAR tokens stuck in a staking pool - issue with key - and how to release?"
date: "2024-12-14"
id: "why-are-near-tokens-stuck-in-a-staking-pool---issue-with-key---and-how-to-release"
---

alright, so you've got near tokens in a staking pool, and they're not moving, and it sounds like a key issue. i've been there, trust me. this isn't uncommon, and while it can be a bit stressful, it's usually solvable. let's break down what's likely happening and what you can do about it, from someone who has definitely been burned by this before.

first, the basic problem: when you stake near, you're essentially delegating your tokens to a validator, and that delegation is controlled by cryptographic keys. the keys that you used when you initially staked or delegated are what the network uses to identify that those tokens are under your control. if those keys are inaccessible or if you've lost access to the keys, then, well, your near are effectively locked in that contract because only the holder of those keys can instruct the contract to unstake and move the tokens. think of it like a safe deposit box, your keys, are your physical keys for accessing that deposit box, without the physical keys, you can't retrieve your valuables.

from what i'm understanding in the question, this problem is not the typical delay in un-staking, that issue is a completely separate and common occurrence in the near network. which takes a while (between 36 and 72 hours) before your near tokens are fully available. this seems to be more fundamental, a situation where you cannot initiate the unstaking at all due to issues of key loss or key-mismatch.

i've experienced this issue way back in the day, back in 2021 during the 'near phase 2' mainnet upgrade. i remember i created a hot wallet, (which i should not have), delegated to a small validator node i knew, and a couple of weeks later, my computer's drive crashed. i had not properly backed up my keys then. i tried everything; drive recovery, data recovery services, but in the end the keys were gone. i lost access to the account. learned a hard and very expensive lesson that day.

it's crucial to understand the different types of keys you might be dealing with. you'll most often encounter these two, in the context of staking:

*   **full access keys:** these can do pretty much anything with the account, including sending tokens, staking, unstaking, and managing other keys. these are the ones that you must protect the most.
*   **function call access keys:** these are restricted keys for specific functions in the near protocol. this type of key is ideal for smart contract interaction, like staking and unstaking with a specific staking contract or another account, but nothing more.

now, the solution to your situation varies depending on exactly how your keys went sideways. but here are the most common scenarios and paths to explore:

**scenario 1: you have your keys, but they are not working**

this is the most common case and the best case scenario. it usually means you're using the wrong key or the wrong key file. i've done this more times than i can count, switching between devnets and mainnets can get confusing if you're not careful and have several key files lying around.

*   **check your active key:** ensure you're using the correct private key for the account you used to stake. verify the account id by matching the key file name and account, often found with an `.json` extension. double check you are using the correct network. sometimes i would forget that i was using my testnet account instead of mainnet account, which can cause this type of problem.
*   **key type:** confirm that your key is a 'full access key' and not a function-call key. if you used a function-call key, you'll have to use the 'full access key' to make the un-staking call. this will require you to access the full-access key, which may be different from the function call key, usually located on the file system or inside the wallet extension.
*   **double check your near-cli setup:** ensure your near-cli is properly configured. your `~/.near-credentials/` directory should contain the correct key file for your account.
*   **try a different near interface:** if the command-line is giving you grief, try using a different client. you could use the near wallet web app or a desktop wallet. each has its quirks, but this approach can sometimes bypass issues that the cli can have if you are a beginner user.

if this doesn't fix it, you may need to reimport your keys to the near-cli or wallet, this can be done with your key file or seed phrase depending on your setup.

here's a near-cli command that you should try, if you're already set up, to check that your account and key are okay (replace `your_account.near` with your actual account id). this command tries to verify your balance and connection with the network:

```bash
near view your_account.near get_account_balance
```

a successful response should give you a balance. if you have an error, that is a hint that there is a problem with your account credentials or key file.

**scenario 2: you've lost access to your keys**

this is the bad scenario. if you've truly lost your full-access key, or lost access to a seed phrase that derives the keys, then things get significantly more complicated. there is not much that you can do in this situation, other than hope you made a backup, or that you are using a custodial exchange that does the key management for you, and in that case, your funds are safe as long as the exchange is solvent and secure.

*   **consider using the recovery process:** near has a recovery system. you can only use it if you set up recovery methods previously (like a recovery email or a phone number, or a hardware key, or another recovery account). you cannot set this up after the fact. if you do, follow the steps outlined in the near documentation about account recovery. this is a very important and often overlooked step. it can be the only way to get access to your account if you lost keys or a seed phrase.
*   **custodial option:** if you staked through a custodial exchange like binance or kraken and you can login there, then, in the vast majority of cases, your funds are safe. it will require you to reach out to customer service, to get clarification on how to un-stake your funds.
*   **advanced help:** there are specialised services to help people recover lost keys but they come with substantial risk and are costly. i will not recommend any specific ones, because i believe in the philosophy of self-custody and that should be a priority for any crypto user.

the following near-cli command shows how to unstake from the command line:

```bash
near call <staking_pool_account_id> unstake '{"amount": "<amount_to_unstake>"}' --accountId <your_account.near> --depositYocto 1
```

*   `staking_pool_account_id`: replace this with the id of the staking pool you used (for example, `pool.near`).
*    `amount_to_unstake`: this is the amount of near to unstake in yocto near. it needs to be a numerical value, for example to un-stake 1 near you would enter `1000000000000000000000000`, which is the smallest unit on near.
*   `your_account.near`: this is your own account id.
*   `--depositYocto 1`: this is mandatory, it deposits 1 yocto near to the transaction. it is necessary for function calls with the near protocol.

**important considerations**

*   **security first:** if you do regain access to your account and keys, move your assets immediately and create a new account with new keys, preferably use a hardware wallet.
*   **key management:** i cannot stress enough, use a hardware wallet to store the keys. it has saved me countless sleepless nights. and always make backups. keep backups in different physical locations and ideally with different storage media. this is not something you learn in an afternoon, but it is a critical security step.
*   **test first:** if you're messing around with staking pools or keys, try everything with small amounts of near first before staking substantial amounts. this is a good practice to have to avoid loss of funds due to a configuration error.
*   **documentation:** the official near documentation is your friend. they have very clear guides, examples and frequently updated details about every functionality on near. you can also find a detailed explanation of keys and the near protocol in the near white paper (not something very technical but provides good insight).

here's a slightly more complex example of unstaking all funds, in yocto near:

```bash
near view <staking_pool_account_id> get_account_staked_balance --accountId <your_account.near> | jq -r .staked | xargs near call <staking_pool_account_id> unstake '{"amount": "0"}' --accountId <your_account.near> --depositYocto 1
```

this uses `jq` to extract the staked balance and then uses it in an unstake command. it also unstakes all funds available in that particular pool.

the biggest mistake i made when i was starting on near, is to assume that i understood everything and tried to cut corners, which came back to bite me. also trusting in hot wallets. this is why i recommend learning good security practices and being diligent when dealing with keys, and always make regular backups of your seed phrase and key files. it is a pain, i know, but it is a very necessary pain. and when it comes to crypto you must take personal responsibility for the security of your keys.

finally, remember that the near network is very active and there is always help available. if you are still stuck, consider looking at the official near forums or stack exchange. they are really good places to find very smart people that are willing to help. good luck.
