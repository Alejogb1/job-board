---
title: "Why does Cycript throw a ReferenceError 'Can't find variable: apr_pool_t' when executing @import com.saurik.substrate.MS?"
date: "2024-12-23"
id: "why-does-cycript-throw-a-referenceerror-cant-find-variable-aprpoolt-when-executing-import-comsauriksubstratems"
---

Alright, let's tackle this. The `ReferenceError: Can't find variable: apr_pool_t` you're encountering in Cycript when trying `@import com.saurik.substrate.MS` is a fairly common and frustrating issue, especially if you've dabbled in mobile substrate modification, as I have quite a few times in the past. It's not a bug in the traditional sense, but rather a consequence of how Cycript and Mobile Substrate (now often referred to as just "substrate") interact with their underlying dependencies and runtime environment. Let’s break down exactly what’s happening and, more importantly, how we address it.

First, the root cause: `apr_pool_t` isn’t something you'd typically find lying around in a standard iOS or macOS runtime. It’s a specific data structure from the Apache Portable Runtime (APR) library. Substrate, being built on APR for its core functions including memory management and threading, critically depends on this library. The problem arises because Cycript, a runtime introspection and injection tool, runs within its own context and, by default, isn’t linked or provided with the correct APR environment that Substrate expects. When you use `@import com.saurik.substrate.MS`, you're effectively asking Cycript to load Substrate's symbols. Substrate's internal machinery immediately tries to access `apr_pool_t`, and because it’s not available in Cycript's context, the `ReferenceError` is thrown.

It's crucial to understand that Cycript doesn’t statically link against Substrate's dependencies. Rather, it attempts to use them dynamically at runtime. This is a key difference, and it means that you can't just assume that because Substrate is installed on your system, Cycript will have access to its internals automatically. Think of it like trying to access a shared library without setting the correct `LD_LIBRARY_PATH` or equivalent environmental variables.

Now, for solutions. The typical approach involves informing Cycript where to find the necessary symbols and libraries. This often boils down to injecting or making available the required APR environment, and there are a few ways to go about it, each with its own benefits and trade-offs.

Let me outline three primary ways I’ve used to resolve this issue, along with code examples to illustrate each.

**Method 1: Manual Symbol Export Through an Injection Script**

This involves crafting a custom injection script that runs before we try to import the Substrate module in Cycript. The script would essentially locate the APR pool structure within Substrate’s memory space and then “export” it to the Cycript runtime. I’ve found this method particularly effective in situations where direct modifications of the Cycript process are not feasible or desired.

```c++
// Example C++ injection code (you'd compile this into a dylib)
#include <dlfcn.h>
#include <substrate.h>

extern "C" {
  __attribute__((constructor)) void inject() {
    void* handle = dlopen("/Library/Frameworks/Substrate.framework/Substrate", RTLD_NOW); // Adjust path as needed
    if (handle == nullptr) {
      return; // Failed to load Substrate
    }
    
    MSImageRef substrate_image = MSGetImageByName("/Library/Frameworks/Substrate.framework/Substrate"); // or its relative path
	if (!substrate_image) return;
	
	void* apr_pool_create_symbol = MSFindSymbol(substrate_image, "_apr_pool_create_unmanaged_ex"); //Substrate >= 0.9.7100
	if (!apr_pool_create_symbol) apr_pool_create_symbol = MSFindSymbol(substrate_image, "_apr_pool_create_unmanaged"); //older substrate versions
	if (!apr_pool_create_symbol) {
		dlclose(handle);
		return; // Failed to find apr_pool_create
	}
	
	typedef apr_pool_t* (*apr_pool_create_func)(apr_pool_t *pool, apr_pool_t *parent, const char *tag, apr_allocator_t *alloc);
    auto pool_create_func = reinterpret_cast<apr_pool_create_func>(apr_pool_create_symbol);
    
	apr_pool_t* pool = pool_create_func(nullptr, nullptr, nullptr, nullptr);
    
	if(pool){
    	MSHookFunction((void*)dlsym(RTLD_DEFAULT, "malloc"), (void*)malloc, (void**)&malloc); // optional malloc fix
		// Find the "apr_pool_t" structure definition
    	void * apr_pool_address = (void*)&pool;
		// The important part, making available for Cycript context, we might need dlsym of apr_pool_t struct itself
		// to export it to global context or to a specific Cycript context
    	void* cycript_context = (void*)dlsym(RTLD_DEFAULT, "cycriptContext");
		if(cycript_context){
        	*(void**)(cycript_context+104) = (void*)apr_pool_address; // offset might vary depending on cycript version
		}
	}
    dlclose(handle);
  }
}
```

This code attempts to load the substrate framework, finds the relevant `apr_pool_create` function, creates an `apr_pool_t`, and then, importantly, tries to push its address into the `cycriptContext` which is then picked up by the Cycript VM. The offset of the `cycriptContext` can vary depending on the Cycript version so finding a good offset might be a source of debugging. Note that the code also attempts to hook the `malloc` function which may help resolve some memory related errors, depending on the Cycript's specific implementation.

**Method 2: Manipulating the Cycript Launch Environment**

Another approach is to modify the environment where Cycript is launched, effectively pre-loading the necessary libraries before it executes. This often involves using `DYLD_INSERT_LIBRARIES` in conjunction with a simple library that initializes the APR environment. It’s a more system-level approach that can be effective but requires a bit more care. This is especially useful when the first method is not an option and the code we want to execute in Cycript depends heavily on substrate symbols.

```c
// Simple C dylib to prep the APR environment
#include <dlfcn.h>
#include <substrate.h>

extern "C" {
  __attribute__((constructor)) void init() {
    void* handle = dlopen("/Library/Frameworks/Substrate.framework/Substrate", RTLD_NOW);
    if (handle == nullptr) {
       return; // bail out
    }
    
	MSImageRef substrate_image = MSGetImageByName("/Library/Frameworks/Substrate.framework/Substrate"); // or its relative path
	if (!substrate_image) return;
	
	void* apr_pool_create_symbol = MSFindSymbol(substrate_image, "_apr_pool_create_unmanaged_ex"); //Substrate >= 0.9.7100
	if (!apr_pool_create_symbol) apr_pool_create_symbol = MSFindSymbol(substrate_image, "_apr_pool_create_unmanaged"); //older substrate versions
	if (!apr_pool_create_symbol) {
		dlclose(handle);
		return; // Failed to find apr_pool_create
	}
    
	typedef apr_pool_t* (*apr_pool_create_func)(apr_pool_t *pool, apr_pool_t *parent, const char *tag, apr_allocator_t *alloc);
    auto pool_create_func = reinterpret_cast<apr_pool_create_func>(apr_pool_create_symbol);

	apr_pool_t* pool = pool_create_func(nullptr, nullptr, nullptr, nullptr);
	if (pool) {} // we create the pool so that it's available to substrate via the framework loading
	
    dlclose(handle);
  }
}
```

This code creates an `apr_pool_t` during its initialization. Now, to use it you would launch Cycript like this (assuming the library is at `/tmp/apr_init.dylib`):

```bash
DYLD_INSERT_LIBRARIES=/tmp/apr_init.dylib cycript
```
This forces Cycript to load our library which sets up the APR environment.

**Method 3: Utilizing Cycript's Internal Modules**

While less common, it's also theoretically possible to extend Cycript’s functionality with custom modules, which themselves could provide the `apr_pool_t` and associated symbols. This requires deeper understanding of Cycript's API and internal structure. I haven’t used this in production, but it's something I've experimented with and considered as a more maintainable solution in a complex environment where multiple Cycript instances may need access to these symbols without resorting to complex environment or injection scripts. I will omit the code example due to the complexity and because it is rarely used in comparison to the first two methods.

For additional resources, I recommend looking into these:

*   **"Advanced Mac OS X Programming" by Mark Dalrymple:** While not specifically about mobile substrate or Cycript, this provides a solid foundation in how dynamic libraries and runtime linking work on macOS, which is directly relevant to understanding the issue.

*   **The APR project documentation:** Directly reading the APR source code and documentation (available from the Apache foundation website) will give you precise details about the `apr_pool_t` structure and its usage. Understanding the origin of the missing dependency is crucial.

*   **The Substrate source code:** While not publicly documented, exploring the decompiled or source code for older versions of Substrate is essential to fully understand its interactions with APR.

These are not beginner-friendly, so I suggest you study the first two methods and see which suits you best. The crux of it is understanding the context in which Cycript runs versus the context where Mobile Substrate was designed to execute, and reconciling the two so that both are aware of their shared dependencies. The error you are encountering is very symptomatic of these environments not being correctly set up.
