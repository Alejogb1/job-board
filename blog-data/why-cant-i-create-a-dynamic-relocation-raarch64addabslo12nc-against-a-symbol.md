---
title: "Why can't I create a dynamic relocation R_AARCH64_ADD_ABS_LO12_NC against a symbol?"
date: "2024-12-23"
id: "why-cant-i-create-a-dynamic-relocation-raarch64addabslo12nc-against-a-symbol"
---

, let's delve into this. It's a situation I've personally bumped into a few times, specifically when we were optimizing some bootloaders a while back. The issue with creating a dynamic relocation of type `r_aarch64_add_abs_lo12_nc` against a symbol isn't some arbitrary quirk; it's deeply rooted in how relocations and addressing modes work on aarch64, and specifically in the limitations of this particular relocation type.

The `r_aarch64_add_abs_lo12_nc` relocation, as its name somewhat implies, operates on the lower 12 bits of an absolute address. It's designed for situations where the instruction itself already contains some address information, and this relocation is meant to *add* a small offset—a 12-bit value—to it. Crucially, it's *not* designed to resolve a full address directly from a symbol. This is where the core problem lies. Symbols, in the context of linking and relocation, represent addresses that are often not known until the final linking stage. They could be located anywhere in memory.

Let me break it down a bit more with why `r_aarch64_add_abs_lo12_nc` is unsuitable: The 'nc' part at the end signifies 'no carry,' meaning that it doesn’t handle any carry from adding the low 12-bit offset into the instruction's pre-existing address. If the result of the addition overflows the lower 12 bits, the upper bits are simply discarded, resulting in incorrect addressing.

The limitation here is that `r_aarch64_add_abs_lo12_nc` expects to modify an instruction which already contains most of the target address, and where that offset will be a small modification. If you attempt to use this type of relocation against a symbol, especially one far away in memory space, the 12-bit offset provided by the relocation will almost certainly be insufficient to fully resolve the symbol’s address. A symbol address could potentially have any value across the available address space, not only a small offset from what the base instruction encodes. It is just too limited to handle the general case of an unknown, full-address symbol.

Consider an instruction something like:

```assembly
    ldr w0, [x1, #0]  ; Assume x1 contains some partial base address
```

If we have a symbol `my_data` located at some address completely separate to where this instruction's base address resides, a simple 12-bit offset isn’t going to get us there. The `r_aarch64_add_abs_lo12_nc` relocation would take that 12 bit offset and try to add it to the instruction. Even if you knew the actual value of `my_data`, that would not suffice, and it's far from what the hardware will execute.

Instead, what is needed is a relocation that allows for calculation of a full address, and not just modify the low 12 bits.

Now, let’s look at what *will* work, with examples.

**Example 1: Using ADRP and ADD**

Typically, you resolve a full 64-bit address on aarch64 by using `adrp` (address page) and `add` instructions in combination. These are specifically designed to handle the full address range. First, you use `adrp` to get the base address of the page containing the symbol. Then use the `add` instruction, which, in conjunction with a suitable relocation type, calculates the offset within the page.

Here's some pseudo-assembly to illustrate this concept:

```assembly
    adrp x0, my_data@PAGE   ;Load the page address of my_data
    add  x0, x0, my_data@PAGEOFF; Add the offset within the page
    ; x0 now holds the full address of my_data
    ldr  w1, [x0] ; load from address in x0
```

Here the `@PAGE` and `@PAGEOFF` are placeholders for relocation types that will be used to calculate the complete address of `my_data`. The equivalent relocations for this on AArch64 would typically be `R_AARCH64_ADR_PAGE21` and `R_AARCH64_ADD_ABS_LO12_NC`. It is a common pattern, but also illustrates the point that we don't directly use `R_AARCH64_ADD_ABS_LO12_NC` against the symbol directly for the complete address calculation.

**Example 2: Using LDR (PC-Relative)**

If your symbol is relatively near the current code location (within a 4 megabyte range), another common method is to use `ldr` to load the address from the code itself. For example:

```assembly
    ldr x0, address_of_data
    ; x0 now holds the full address of my_data

address_of_data:
    .quad my_data   ; Address of my_data will be placed here
    ; then follow instructions to read or write from address in x0
    ldr w1, [x0]
```

Here, the linker would take the symbol 'my_data' and resolve the instruction, substituting it with the appropriate 64-bit address. The relocation used here would typically be `R_AARCH64_ABS64` against the `.quad my_data` instruction which is placed inline in the code.

**Example 3: GOT (Global Offset Table)**

For scenarios where position independent code (PIC) is necessary, the global offset table (GOT) is commonly used. This allows all references to data, symbols, etc. to go through a single table, which can then be fixed up by the dynamic linker.

```assembly
    ldr x0, [x2, my_data@GOT] ; x2 contains the base address of the GOT
    ; x0 now contains the address of my_data
    ldr w1, [x0]

```
Here `my_data@GOT` will use `R_AARCH64_GOT_PAGE_PCREL` to calculate an offset from the address of the instruction to a location in the GOT which in turn has the address of my_data. The key point here is that while the GOT entry itself might use a form of a 12-bit offset for its own internal resolution within the GOT page, the *initial instruction* is not directly applying `R_AARCH64_ADD_ABS_LO12_NC` to the symbol. The relocation type `R_AARCH64_GOT_PAGE_PCREL` in conjunction with the addressing through the GOT table avoids this issue entirely.

The key take away is: if the address is not a local offset, then `R_AARCH64_ADD_ABS_LO12_NC` is not the appropriate relocation type.

It’s important to check the aarch64 ABI documentation for specific details of how these relocations work. You can typically find good information in the *Procedure Call Standard for the Arm 64-bit Architecture* document published by Arm. Also, exploring the GNU Binutils documentation for `as` and `ld` will provide a much deeper understanding of how symbols, relocations, and code generation interact. It will detail the specific nuances and details. Similarly, a strong understanding of ELF object file formats, especially the relocation section, can also be highly beneficial. These are not simple reads, and require time and attention, but are core to working at this level of detail.

In short, the problem you're experiencing arises from trying to fit a square peg (a full symbol address) into a round hole (a 12-bit offset relocation). You need to use the specific addressing mechanisms—`adrp` and `add`, pc relative addressing, GOT access—that are designed to handle full symbol address resolution on aarch64 and use the correct relocation types designed for each addressing mode.
