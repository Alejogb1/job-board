---
title: "What programming language is used in .arl files for Tezos?"
date: "2024-12-23"
id: "what-programming-language-is-used-in-arl-files-for-tezos"
---

Alright, let's dive into the specifics of .arl files within the Tezos ecosystem. I recall a rather challenging project involving optimizing smart contract deployments where I had to get intimately familiar with these files. The answer isn't a straightforward one, as .arl files aren’t associated with a specific, conventional programming language in the way that, say, python files are. Instead, they represent an intermediate format, typically stemming from *compilation outputs*. More precisely, they hold the *Micheline* representation of a Tezos smart contract's code.

Think of it like this: you write a smart contract using a high-level language (like SmartPy, Ligo, or Archetype). This code is then processed through a compiler—essentially, a translator—that transforms it into Michelson, the low-level instruction set that the Tezos virtual machine directly understands. This Michelson representation, however, isn’t always stored directly as raw text. Instead, it is often expressed in a structured, JSON-like format called Micheline. And that’s exactly what a .arl file houses—the Micheline representation of the compiled Michelson code.

So, to answer the question explicitly, there isn't *a* programming language used directly in .arl files. The content is derived from the process of compiling contract code written in languages such as SmartPy, Ligo, or Archetype. .arl is essentially the intermediate format and it stores the result of this translation in Micheline form, which is not technically a programming language on its own. To get a feel for it, it might be helpful to consider it an assembly language representation of contract code for Tezos. It's the form that sits between the high-level code you write and the low-level bytecode that executes. This design approach is not unique to Tezos; it mirrors common patterns in compiler design where you often have intermediate representation (IR) before machine code.

Now, let’s get into some code to illustrate these points. Suppose we have a very simple contract written using the SmartPy language, a Python-based DSL for Tezos. Here’s a snippet:

```python
import smartpy as sp

class SimpleCounter(sp.Contract):
  def __init__(self):
      self.init(counter = 0)

  @sp.entry_point
  def increment(self):
      self.data.counter += 1

  @sp.entry_point
  def getCounter(self):
      sp.result(self.data.counter)

if "main" == __name__:
    sp.add_compilation_target("simpleCounter", SimpleCounter())

```
This straightforward example defines a simple counter contract with the ability to increment it. Now, when we compile this code using SmartPy's compiler, the output will generate multiple files; among them is the .arl file that will contain the Micheline representation of the compiled contract. The content of this .arl would be a text file in the Micheline format, something similar to this:

```json
[
   { "prim": "parameter",
     "args":[ { "prim":"or",
               "args":[
                        { "prim":"unit", "annots":["%increment"] },
                        { "prim":"unit", "annots":["%getCounter"]}
                      ] }
              ]
  },
  {
   "prim":"storage",
   "args":[ { "prim":"int" } ]
  },
  {
   "prim":"code",
   "args":[
    [
      {"prim":"DUP"},
      {"prim":"CAR"},
      {"prim":"UNPAIR"},
      {"prim":"IF_LEFT",
         "args":[
            [
             {"prim":"PUSH", "args":[{"prim":"int"}, 1]},
             {"prim":"ADD"},
             {"prim":"NIL", "args":[{"prim":"operation"}]},
             {"prim":"PAIR"}
            ],
            [
              {"prim":"DUP"},
              {"prim":"CAR"},
              {"prim":"SWAP"},
              {"prim":"DROP"},
              {"prim":"NIL", "args":[{"prim":"operation"}]},
              {"prim":"PAIR"}
            ]
          ]
       }
     ]
    ]
  }
]
```

This example shows a simplified and slightly formatted version for readability. The actual .arl will contain this structure, although potentially more condensed without spaces or line breaks. The JSON-like notation you see here defines the Michelson code which was produced by the SmartPy compiler from the high-level Python code. Note the "prim" keywords: these are the primitive Michelson instructions and constructions like `parameter`, `storage`, `code`, `dup`, `car`, `pair` etc.. The first segment defines the parameter types for contract entrypoints (`increment` and `getCounter`). The second part defines the storage field (`int` for the counter), and the final section contains the actual instructions for the contract's logic.

This format is *not* something you'd write directly. You'd instead use a high-level language (SmartPy in this instance) and then the compiler does the translation process, generating this .arl file. Now let's consider a contrasting example. If you were using Ligo, a different smart contract language also for Tezos (that's closer to OCaml), you would write the contract in Ligo language and then use the Ligo compiler which produces a similar structure in the .arl file after compilation; the Michelson instructions in Micheline form would have a slightly different layout than the one created by SmartPy, reflecting the Ligo contract's syntax and semantics.

Here’s a very simple Ligo smart contract:

```ligo
type storage = int;

type parameter = Increment of unit | GetCounter of unit;

type return = list(operation) * storage;

let main (action,store:parameter * storage) : return =
   match action with
   | Increment(_unit) ->
     let new_store:storage = store + 1;
     (([] : list (operation)), new_store)
   | GetCounter(_unit) ->
      (([] : list (operation)), store)
```
When you compile this with the Ligo compiler, you'll also have a .arl file containing the Michelson code representation in Micheline form:
```json
[
  { "prim": "parameter",
    "args": [
      { "prim":"or",
        "args": [
          { "prim": "unit", "annots": ["%Increment"] },
          { "prim": "unit", "annots": ["%GetCounter"] }
         ]
        }
      ]
    },
    {
      "prim": "storage",
      "args": [ { "prim": "int" } ]
    },
    {
      "prim": "code",
      "args": [
        [
          {"prim":"DUP"},
          {"prim":"CAR"},
          {"prim":"UNPAIR"},
          {"prim":"IF_LEFT",
          "args":[
            [
             {"prim":"PUSH", "args":[{"prim":"int"}, 1]},
             {"prim":"ADD"},
              {"prim":"NIL", "args":[{"prim":"operation"}]},
              {"prim":"PAIR"}
            ],
             [
               {"prim":"DUP"},
               {"prim":"CAR"},
               {"prim":"SWAP"},
              {"prim":"DROP"},
              {"prim":"NIL", "args":[{"prim":"operation"}]},
              {"prim":"PAIR"}
            ]
          ]
        }
      ]
    }
]
```

This output looks structurally similar to the SmartPy output. Both are in Micheline, but the source code and even the specifics of the intermediate Michelson representation are very close, reflecting the functional similarity between the two contracts in different languages. This demonstrates how .arl files act as a bridge – they allow you to inspect the low-level Michelson code after compilation, but the real interaction is with the high-level languages like Ligo or SmartPy.

Let's add another example, this time using Archetype, a domain-specific language for Tezos that’s also different in design and approach:

```archetype
  entry increment () {
    s.counter := s.counter + 1
  }
  entry getCounter () {
    return s.counter
  }
  variable counter : int = 0
```

The resulting .arl file after compilation would again be in the same Micheline structure, though the arrangement of Michelson operations could differ again based on the specific optimization and compilation strategies of the Archetype compiler. It's still representing the Michelson code, but it might be structured or arranged slightly differently from SmartPy and Ligo outputs.

As a practical resource, I recommend exploring the official Michelson documentation, along with the documentation for the various high-level languages and their compilers (SmartPy, Ligo, Archetype) that target the Tezos platform. You can find them on their respective websites or project repositories; start with the official Tezos documentation, which goes deep into how the virtual machine processes Michelson. There is also the book "Real World OCaml" which provides a deep understanding of the foundational principles of functional programming, useful especially for using Ligo. It would also be worthwhile to examine formal papers on compiler design and intermediate representations (IRs), which shed light on why languages often have such steps like generating an IR prior to bytecode, as it allows for modular compilation process, optimizations, and better support for different target virtual machines. In essence, the .arl file is not the end game; it's just a checkpoint in the journey from high-level abstraction to executable contract. Understanding this helps when troubleshooting or optimizing contracts for the Tezos blockchain.
