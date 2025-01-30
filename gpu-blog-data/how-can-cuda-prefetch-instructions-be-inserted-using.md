---
title: "How can CUDA prefetch instructions be inserted using LLVM IR Builder?"
date: "2025-01-30"
id: "how-can-cuda-prefetch-instructions-be-inserted-using"
---
Achieving optimal memory access patterns in CUDA kernels often necessitates explicit control over data movement between global and shared memory. Prefetching, in particular, can significantly mitigate stalls caused by long latency loads from global memory. While CUDA exposes prefetching mechanisms through intrinsics within its programming model, implementing these directly using LLVM Intermediate Representation (IR) provides fine-grained control and enables specialized optimizations not otherwise attainable through higher-level abstractions. I’ve spent considerable time optimizing high-throughput compute kernels on NVIDIA GPUs, and manipulating LLVM IR is a technique I've frequently employed to squeeze every ounce of performance.

The core issue when inserting CUDA prefetch instructions within LLVM IR using the `IRBuilder` is that these instructions are not directly represented by standard LLVM opcodes. Instead, CUDA's prefetching is typically accomplished using specific intrinsics, such as `llvm.nvvm.prefetch.global.to.shared` (or its variants, depending on the specific prefetch target and address space). These intrinsics map to the underlying machine instruction sequences that perform the data movement. The `IRBuilder` class, part of the LLVM API, is the conduit through which we generate these instructions into an LLVM Module. Essentially, we must craft an appropriate call to the intrinsic function, with the correct operand types and address spaces, and then insert this call at the desired location in the IR.

To begin, we must obtain the `llvm::Module` instance we wish to modify. We then need to retrieve the function, or create it if necessary, in which we will insert the prefetch instructions. The primary components when building this instruction are the target address in global memory, the address in shared memory to store the prefetched data (if a store operation is required), the size of the data to prefetch (usually the size of a single vector, or a small multiple of it), and the address space designators for all relevant pointers. Address spaces are of paramount importance here as they direct the hardware to interpret the pointers correctly, specifically whether the pointer is to global, shared, or other memory.

Below are three code examples, demonstrating various prefetch scenarios. Each example assumes a basic LLVM setup where an `llvm::IRBuilder<>` has been created and is ready to insert instructions. The examples focus on clarity, not production-ready code. You would need to appropriately include LLVM headers and link against the LLVM libraries to compile this.

**Example 1: Simple Prefetch of a Single Element**

```c++
// Assume we have llvm::IRBuilder<> Builder, llvm::Module *M, llvm::Function *F.
llvm::Type *int32PtrType = llvm::PointerType::get(llvm::Type::getInt32Ty(M->getContext()), 1); // Address space 1 = Global
llvm::Type *int32Type = llvm::Type::getInt32Ty(M->getContext());
llvm::Type *int64Type = llvm::Type::getInt64Ty(M->getContext());

// Assume 'globalPtr' is a llvm::Value* holding a pointer in global memory (address space 1).
// Assume 'sharedPtr' is a llvm::Value* holding a pointer in shared memory (address space 3).
llvm::Value* globalPtr; // Assume allocated and populated.
llvm::Value* sharedPtr; // Assume allocated and populated.

// 1. Locate the llvm.nvvm.prefetch.global.to.shared intrinsic.
llvm::Function *prefetchFunc = M->getFunction("llvm.nvvm.prefetch.global.to.shared.i32.i32");
if(!prefetchFunc)
{
    // Create if it doesn't exist.
    llvm::FunctionType *funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), {int32PtrType, llvm::Type::getInt32Ty(M->getContext()), llvm::Type::getInt64Ty(M->getContext()) }, false);
    prefetchFunc = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage, "llvm.nvvm.prefetch.global.to.shared.i32.i32", M);
}

// 2. Create the arguments to the intrinsic
llvm::Value *sizeBytes = llvm::ConstantInt::get(int64Type, 4); //Size of 4 bytes

// 3. Insert the call to the prefetch intrinsic.
std::vector<llvm::Value*> args = {globalPtr, llvm::ConstantInt::get(int32Type,0) , sizeBytes }; // second argument is offset, zero for now
Builder.CreateCall(prefetchFunc, args);
```

This example illustrates the fundamental procedure. It first identifies or creates the necessary intrinsic function (`llvm.nvvm.prefetch.global.to.shared.i32.i32`). Then, it prepares the arguments required by the intrinsic: the pointer to the global memory address, an offset (set to zero in this example), and the size of the memory to prefetch (four bytes for an integer). Finally, the code uses `Builder.CreateCall` to insert the function call into the IR. This generates a prefetch instruction that brings a single 32-bit integer from the address pointed to by `globalPtr` into shared memory (although no shared memory address is actually passed directly to the instruction in this form of prefetch).

**Example 2: Prefetching with an Offset and Different Type**

```c++
// Assume we have llvm::IRBuilder<> Builder, llvm::Module *M, llvm::Function *F
llvm::Type *floatPtrType = llvm::PointerType::get(llvm::Type::getFloatTy(M->getContext()), 1);
llvm::Type *int32Type = llvm::Type::getInt32Ty(M->getContext());
llvm::Type *int64Type = llvm::Type::getInt64Ty(M->getContext());

// Assume 'globalPtr' is a float pointer.
llvm::Value* globalPtr; // Assume allocated and populated, float pointer
llvm::Value *offset = llvm::ConstantInt::get(int32Type, 8); // Offset of 8 bytes

// 1. Locate the appropriate prefetch intrinsic.
llvm::Function *prefetchFunc = M->getFunction("llvm.nvvm.prefetch.global.to.shared.f32.i32");
if (!prefetchFunc)
{
    llvm::FunctionType* funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), { floatPtrType, llvm::Type::getInt32Ty(M->getContext()), llvm::Type::getInt64Ty(M->getContext()) }, false);
    prefetchFunc = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage, "llvm.nvvm.prefetch.global.to.shared.f32.i32", M);
}
// 2. Prepare arguments: Size = size of float, offset added.
llvm::Value *sizeBytes = llvm::ConstantInt::get(int64Type, 4);

// 3. Generate the call instruction with the modified arguments.
std::vector<llvm::Value*> args = { globalPtr, offset, sizeBytes };
Builder.CreateCall(prefetchFunc, args);

```

This example builds upon the first by introducing an offset into the global memory pointer (`offset`) and changing the intrinsic to handle float data types (`llvm.nvvm.prefetch.global.to.shared.f32.i32`). This demonstrates the need to select the correct variant of the prefetch intrinsic based on the underlying data type and memory access patterns. The offset is passed as the second argument to the intrinsic. This is especially valuable when working with arrays.

**Example 3: Explicit Shared Memory Write using Store Instruction (Simplified)**

```c++
// Assume we have llvm::IRBuilder<> Builder, llvm::Module *M, llvm::Function *F
llvm::Type *int32PtrType = llvm::PointerType::get(llvm::Type::getInt32Ty(M->getContext()), 1);
llvm::Type *sharedInt32PtrType = llvm::PointerType::get(llvm::Type::getInt32Ty(M->getContext()), 3); // Address space 3 = Shared.
llvm::Type *int32Type = llvm::Type::getInt32Ty(M->getContext());
llvm::Type *int64Type = llvm::Type::getInt64Ty(M->getContext());

// Assume 'globalPtr' and 'sharedPtr' are available, as are offsets.
llvm::Value* globalPtr; //global pointer
llvm::Value* sharedPtr; //shared memory pointer
llvm::Value *offsetGlobal = llvm::ConstantInt::get(int32Type, 0);
llvm::Value *offsetShared = llvm::ConstantInt::get(int32Type, 0);

// 1. Locate the intrinsic function, prefetch followed by shared write.
llvm::Function *prefetchFunc = M->getFunction("llvm.nvvm.prefetch.global.to.shared.i32.i32");
if(!prefetchFunc)
{
    llvm::FunctionType *funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), {int32PtrType, llvm::Type::getInt32Ty(M->getContext()), llvm::Type::getInt64Ty(M->getContext()) }, false);
    prefetchFunc = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage, "llvm.nvvm.prefetch.global.to.shared.i32.i32", M);
}

llvm::Function *barrierFunc = M->getFunction("llvm.nvvm.barrier.all");
if(!barrierFunc)
{
    llvm::FunctionType* funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), false);
    barrierFunc = llvm::Function::Create(funcType, llvm::GlobalValue::ExternalLinkage, "llvm.nvvm.barrier.all", M);
}

// 2. Calculate size of prefetch, and arguments for call.
llvm::Value *sizeBytes = llvm::ConstantInt::get(int64Type, 4);
std::vector<llvm::Value*> args = {globalPtr, offsetGlobal, sizeBytes };

// 3. Insert the call.
Builder.CreateCall(prefetchFunc, args);
Builder.CreateCall(barrierFunc); //Ensure prefetch is complete

//4. Generate a load from a pointer in global memory, now (hypothetically) available locally
llvm::Value* loadedValue = Builder.CreateLoad(llvm::Type::getInt32Ty(M->getContext()), globalPtr);

//5. Store to a location in shared memory
llvm::Value* sharedPtrOffset = Builder.CreateGEP(llvm::Type::getInt32Ty(M->getContext()), sharedPtr, offsetShared);
Builder.CreateStore(loadedValue, sharedPtrOffset);
```
This more complex example demonstrates the more typical process of a prefetch followed by an explicit store into shared memory. Note the use of `llvm.nvvm.barrier.all` to ensure the prefetch operation has completed and the loaded data is available. A `CreateLoad` instruction is used to read data from global memory now that it has (ideally) been moved into shared memory. This value is then explicitly written into shared memory using `CreateStore` at the shared pointer location, offset by `offsetShared` using a GetElementPtr (GEP) instruction.

**Resource Recommendations**

The primary resource is the LLVM documentation, specifically the API reference for `llvm::IRBuilder` and the general documentation on LLVM IR. NVIDIA's documentation on CUDA intrinsics is also essential, providing a list of available intrinsics and their associated functionalities. While not directly about LLVM, familiarity with the CUDA programming model and memory hierarchy (global, shared, local) is fundamental for proper prefetching. Further study of GPU architecture is beneficial, particularly understanding memory coalescing, bank conflicts in shared memory, and the various latency characteristics of different types of memory. Examining existing LLVM compiler passes for CUDA may also offer insights into common code generation techniques, though that is a much more advanced topic. The LLVM source code itself provides a practical reference to examine how intrinsics are handled within the compiler infrastructure.
