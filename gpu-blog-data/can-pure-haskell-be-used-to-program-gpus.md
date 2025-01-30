---
title: "Can pure Haskell be used to program GPUs?"
date: "2025-01-30"
id: "can-pure-haskell-be-used-to-program-gpus"
---
Direct memory manipulation, fundamental to many GPU programming paradigms, is inherently at odds with Haskell's purity. The question of using pure Haskell for GPU programming, therefore, requires a careful look at how we bridge this gap, not through direct manipulation but through abstraction and code generation. I've spent considerable time exploring functional approaches to parallel computing, encountering both the elegance and the limitations, and I'll outline the core challenges and potential solutions.

The primary hurdle lies in the nature of functional programming and its avoidance of side effects. GPUs, in their essence, are parallel processors executing imperative instructions operating directly on memory locations. Pure Haskell, by design, prevents such direct mutations. This doesn't inherently preclude using Haskell for GPU programming but necessitates a shift in thinking. We cannot compile pure Haskell directly to GPU assembly. Instead, we need a system to take Haskell code and translate it into a form the GPU can understand: typically, a language such as CUDA C/C++ or OpenCL kernels.

Essentially, the process involves building a domain-specific language (DSL) embedded within Haskell. This DSL captures the essence of parallel computation, expressed in a functional way, and it's the DSL, not the raw Haskell, that eventually gets translated. The process can be broadly broken into three stages: **1) DSL definition**: We establish data types and functions within Haskell that represent GPU computations; **2) Code Generation**: We write a function within Haskell that takes our DSL representation and produces GPU-compatible code in C-like languages; **3) Execution**: We use external libraries to load the generated kernel onto the GPU and execute it.

Let's consider a simple example: performing an element-wise addition of two vectors. In a purely functional approach, we would avoid in-place modification of vector elements. The following will outline how one might build such a system.

```haskell
data GPUVector a = GPUVector Int (Array Int a)

-- Represents an operation in our DSL
data GPUOperation a = Add (GPUVector a) (GPUVector a)
                   | Return (GPUVector a)

-- A function that interprets our operations and does not evaluate
interpretOperation :: Num a => GPUOperation a -> GPUVector a
interpretOperation (Add (GPUVector size1 arr1) (GPUVector size2 arr2))
    | size1 /= size2 = error "Vectors are not equal length"
    | otherwise = GPUVector size1 (listArray (0, size1 - 1) $ zipWith (+) (elems arr1) (elems arr2) )
interpretOperation (Return vec) = vec

-- This function builds an example operation
buildAddOperation :: Int -> (Int -> a) -> (Int -> a) -> GPUOperation a
buildAddOperation size initFunc1 initFunc2 =
  let vec1 = GPUVector size (listArray (0, size - 1) (map initFunc1 [0..size - 1]))
      vec2 = GPUVector size (listArray (0, size - 1) (map initFunc2 [0..size - 1]))
  in Add vec1 vec2
```

This code defines the *types* necessary to represent GPU operations as data structures in Haskell. `GPUVector` represents a vector in GPU memory and `GPUOperation` is a DSL of available operations. The crucial point here is that `interpretOperation` is an *interpreter* for our DSL. It *evaluates* operations, but only for a local *CPU* based vector representation. The `buildAddOperation` function allows creation of new operations with specific initializations. This implementation is functionally correct but will not execute on the GPU. It is, in fact, a normal Haskell program.

Now, let's extend this to include the translation step to OpenCL.

```haskell
-- Function to translate the DSL to OpenCL
translateToOpenCL :: Num a => GPUOperation a -> String
translateToOpenCL (Add (GPUVector size1 _) (GPUVector size2 _))
  | size1 /= size2 = error "Vectors are not equal length"
  | otherwise = unlines [
      "__kernel void vector_add(__global float* a, __global float* b, __global float* c) {",
      "  int i = get_global_id(0);",
      "  if (i < " ++ show size1 ++ ") {",
      "    c[i] = a[i] + b[i];",
      "  }",
      "}"
    ]
translateToOpenCL (Return _) = error "Return cannot be translated to OpenCL"

-- A simplified runner for OpenCL translation and compilation
runOpenCLOperation :: Num a => GPUOperation a -> IO String
runOpenCLOperation op = do
  let openCLCode = translateToOpenCL op
  -- Simplified OpenCL kernel compilation and execution
  -- Assuming a library named OpenCLWrapper
  -- Would include things such as setting up the environment,
  -- compiling the OpenCL code,
  -- allocating GPU memory, and copying data back
  result <- compileAndRunOpenCL openCLCode
  return result
```

This is a simplified example but encapsulates the core idea. The `translateToOpenCL` function takes our DSL `GPUOperation` and returns a string containing the corresponding OpenCL code. This is where we go from a functional abstraction to imperative GPU code. The `runOpenCLOperation` demonstrates the final stage of using an external `OpenCLWrapper` library (which we are not implementing) to handle compilation and execution of the generated kernel on the GPU. The output of `runOpenCLOperation` would then be the result of the computation in a String representation.

Finally, let's demonstrate a more complex computation: a parallel map operation.

```haskell
-- DSL for parallel map
data GPUOperationMap a b = Map (GPUVector a) (a -> b)
                            | ReturnMap (GPUVector b)

-- Interpreter for parallel map
interpretMap :: (a -> b) -> GPUOperationMap a b -> GPUVector b
interpretMap mapper (Map (GPUVector size arr) _) =
     GPUVector size (listArray (0, size - 1) $ map mapper (elems arr))
interpretMap _ (ReturnMap vec) = vec

translateMapOpenCL :: (Show a, Show b) => GPUOperationMap a b -> String
translateMapOpenCL (Map (GPUVector size _) func) =
  unlines [
      "__kernel void map_kernel(__global " ++ typeName (undefined :: a) ++ "* input, __global " ++ typeName (undefined :: b) ++ "* output) {",
      "    int i = get_global_id(0);",
      "    if (i < " ++ show size ++ ") {",
      "      output[i] = " ++ generateMapperCall func ++ "(input[i]);",
      "    }",
      "}"
    ]
translateMapOpenCL (ReturnMap _) = error "ReturnMap cannot be translated"

-- Helper for creating type names
typeName :: a -> String
typeName _ = case typeOf (undefined :: a) of
  (ConType con) | con == typeOf (undefined :: Int) -> "int"
                | con == typeOf (undefined :: Float) -> "float"
                | otherwise -> error "Unsupported type"
  otherwise -> error "Unsupported type"

-- Helper to generate code for mapper functions. For brevity, we only allow simple functions
-- Not fully implemented, but demonstrates the concept
generateMapperCall :: (a -> b) -> String
generateMapperCall func = case func of
    (a -> b) -> "(a => a + 1)"  -- Replace with appropriate C-like representation


buildMapOperation :: Int -> (Int -> a) -> (a->b) -> GPUOperationMap a b
buildMapOperation size initFunc mapper =
    let vec1 = GPUVector size (listArray (0, size - 1) (map initFunc [0..size - 1]))
    in Map vec1 mapper


runOpenCLMap :: (Show a, Show b) => GPUOperationMap a b -> IO String
runOpenCLMap op = do
    let openCLCode = translateMapOpenCL op
    -- Simplified OpenCL execution
    result <- compileAndRunOpenCL openCLCode
    return result
```

This example introduces a `GPUOperationMap` type and a corresponding `translateMapOpenCL` function. It aims to map an arbitrary Haskell function (constrained here to simple cases) to GPU execution. The example shows the idea of converting Haskell's type system to C's. The `generateMapperCall` function exemplifies the challenge of code generation—converting arbitrary Haskell expressions into C-like code is generally difficult. These function would require additional checks to prevent malicious code.

In summary, pure Haskell can be used to *describe* GPU computations, but it cannot execute them directly. This requires us to build DSLs, code generators, and bridge the gap using libraries like OpenCL or CUDA bindings. This is the approach used by libraries such as Accelerate or Obsidian. These libraries use Haskell to define computations that can be compiled down to GPU instructions.

For further study, several resources are available. For a theoretical basis for functional programming applied to parallel computing, look for material on *Data Parallel Haskell*. Regarding specific libraries, consider reading the documentation for the *Accelerate* package, which provides a functional DSL for GPUs. For a more general overview of GPU programming paradigms from a practical point of view, focus on resources concerning *CUDA* and *OpenCL*, along with specific hardware manuals from GPU manufacturers. These will provide the context and background necessary to better understand the complexities involved. Studying the implementation details of these resources would allow for the creation of similar tools.
