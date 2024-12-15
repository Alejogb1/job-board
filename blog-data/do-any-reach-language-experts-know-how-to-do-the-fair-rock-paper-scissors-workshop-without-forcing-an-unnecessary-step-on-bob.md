---
title: "Do any Reach language experts know how to do the 'Fair Rock-Paper-Scissors' workshop without forcing an unnecessary step on 'Bob'?"
date: "2024-12-15"
id: "do-any-reach-language-experts-know-how-to-do-the-fair-rock-paper-scissors-workshop-without-forcing-an-unnecessary-step-on-bob"
---

hello,

so, you're tackling the fair rock-paper-scissors problem in reach, and trying to avoid making bob do an extra unnecessary step, huh? i've been there. many times. i’ve lost countless hours to similar issues, let me tell you. it’s those seemingly simple cases that can really make you question your sanity. i even remember one time, back when reach was still in its early days (well, relatively early; feels like eons ago now), i spent a whole weekend trying to get a multi-party contract to work smoothly. it involved a similar "fairness" problem, and the initial solution was… well, let's just say it involved way too many steps for everyone. the gas fees alone were enough to make my wallet cry. i ended up having to tear it all down and start over. anyway, enough nostalgia.

here's the thing with reach, especially when trying to build a fair multi-party app, is that you really need to think through the flow. forcing bob to take an extra step, as you pointed out, is a common trap. the core issue here is likely about how you’re handling commitments and reveals, particularly regarding when to expose bob's chosen move. the challenge, if i understood correctly, is to allow alice and bob to choose their actions simultaneously while preserving the fairness of the game.

one of the first things that i always do is carefully examine the state machine in the code. what i mean is, how is the information flow designed between actors?. let’s break down why you might be encountering the extra step. most often this extra unnecessary step comes when you have an intermediary stage where one actor (in your case bob) has to reveal a committed choice, while the other party (alice) could have already proceeded to the next stage and already exposed the result. this often results in an unnecessary blocking stage where we are waiting for bob to reveal when actually we could skip that stage in our workflow and keep it simultaneous.

the key is to use reach’s commitment features properly and to make sure that the game logic doesn’t proceed until the necessary information is available but without unnecessary steps. here’s what i typically recommend:

1. **commitment and reveal pattern:** you should already know this. alice and bob choose their move secretly (a commitment) and later reveal it. make sure that both are done correctly. the goal is that both participants should provide this commitment on the same step. the reveal step must be done as efficiently as possible, without requiring an extra turn.

2. **using `parallelReduce`:** this reach function is your friend. instead of forcing bob to take an extra step, use parallelReduce to trigger both commitment and reveals at the same time.

3. **state variables:** keep track of committed moves in state variables of the contract. you will need them for comparing and generating the result.

let's look at an example. consider the following snippet. this one uses a single round (simplest case).

```reach
  'reach 0.1';

  const Player = {
    commit: Fun([Bytes(32)], Null),
    reveal: Fun([Bytes(32), UInt], Null),
  };

  export const main = Reach.App(() => {
    const Alice = Participant('Alice', {
      ...Player,
      wantsPlay: Fun([], Null),
    });
    const Bob = Participant('Bob', {
      ...Player,
      wantsPlay: Fun([], Null),
    });

    init();

    Alice.interact.wantsPlay();
    Bob.interact.wantsPlay();

    const [aliceCommit, bobCommit] = parallelReduce([
      Alice,
      Bob
    ], ([aliceCommit, bobCommit], p) => {
        const commitment = p.commit(digest);
        return [commitment.value,commitment.value]
    });
    
    const [aliceMove, bobMove] = parallelReduce([
      Alice,
      Bob
    ], ([aliceMove, bobMove], p) => {
       const revealed = p.reveal(digest,move);
      return [revealed.value,revealed.value];
    });

    const outcome = (aliceMove == bobMove ? 0 :
                     (aliceMove == 0 && bobMove == 1 ? -1 :
                     (aliceMove == 1 && bobMove == 2 ? -1 :
                     (aliceMove == 2 && bobMove == 0 ? -1 : 1))));

      commit();

    if (outcome == 0) {
      each([Alice, Bob], () => {
        interact.reportTie();
      });
    } else if (outcome > 0){
        Alice.only(() => {
          interact.reportWin();
      });
        Bob.only(() => {
           interact.reportLose();
      });
    } else {
      Bob.only(() => {
        interact.reportWin();
      });
        Alice.only(() => {
          interact.reportLose();
      });
    }
  });
```

in the above example, the commitments and reveals are done simultaneously, thanks to `parallelReduce`. this completely avoids the extra step for bob. each participant commits their move, then reveals it, all without unnecessary back-and-forth. i've made it so the winner is not calculated in any actor part, and only in the shared state part of the contract. in reach, we need to be careful to not move any variable assignment to each actor since the variable will be only assigned in the perspective of each actor and they could differ.

now, let’s say you want to expand this to a multi-round setup. here is a snippet showing that:

```reach
  'reach 0.1';

  const Player = {
    commit: Fun([Bytes(32)], Null),
    reveal: Fun([Bytes(32), UInt], Null),
  };

  export const main = Reach.App(() => {
    const Alice = Participant('Alice', {
      ...Player,
      wantsPlay: Fun([], Null),
    });
    const Bob = Participant('Bob', {
      ...Player,
      wantsPlay: Fun([], Null),
    });

    init();

   const rounds = 3;
   for (let i = 0; i < rounds; i++) {

    Alice.interact.wantsPlay();
    Bob.interact.wantsPlay();

      const [aliceCommit, bobCommit] = parallelReduce([
      Alice,
      Bob
    ], ([aliceCommit, bobCommit], p) => {
        const commitment = p.commit(digest);
        return [commitment.value,commitment.value]
    });
    
    const [aliceMove, bobMove] = parallelReduce([
      Alice,
      Bob
    ], ([aliceMove, bobMove], p) => {
       const revealed = p.reveal(digest,move);
      return [revealed.value,revealed.value];
    });

    const outcome = (aliceMove == bobMove ? 0 :
                    (aliceMove == 0 && bobMove == 1 ? -1 :
                    (aliceMove == 1 && bobMove == 2 ? -1 :
                    (aliceMove == 2 && bobMove == 0 ? -1 : 1))));

    commit();

      if (outcome == 0) {
        each([Alice, Bob], () => {
          interact.reportTie();
        });
      } else if (outcome > 0){
          Alice.only(() => {
            interact.reportWin();
        });
          Bob.only(() => {
            interact.reportLose();
        });
      } else {
        Bob.only(() => {
          interact.reportWin();
        });
          Alice.only(() => {
            interact.reportLose();
        });
      }
    }
  });
```

this version includes a `for` loop to create multiple rounds. note how we keep the `parallelReduce` structure. we apply it in each round so there is no extra unnecessary stage for Bob. you can play with this code if you want and modify the rounds variable.

also, keep in mind that security is an important factor. commitment schemes are notoriously difficult. an incorrectly implemented commitment can allow a malicious actor to cheat. so, ensure that you pick a good secure commitment schema. if you plan to make this production-ready, do not roll your own commitment scheme. use a well-tested one from a reliable cryptography library.

and if you want to dive even deeper, i recommend checking out the reach documentation on parallel reduce and state variables. it's really essential reading when dealing with complex multi-party workflows in reach. also, "building secure and reliable distributed applications: a formal approach" by andrew p. blackwell could be interesting to see. it has a very formal take on similar problems. or check "hands-on smart contract development with solidity and ethereum" by kevin solorio, although it uses solidity it does cover many underlying ideas about security and smart contract programming. finally, "a pragmatic programmer" by andrew hunt is not focused on smart contracts but on good software development principles that i apply when building any application.

one funny thing i remember, a colleague of mine spent weeks trying to debug a bug that was caused by him using variable names with capital letters, and in reach, you have to keep the naming rules. oh, the memories!

anyway, i hope this helps you to refine your reach app. happy coding!
