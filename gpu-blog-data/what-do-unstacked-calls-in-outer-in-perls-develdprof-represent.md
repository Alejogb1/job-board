---
title: "What do 'unstacked calls in outer' in Perl's Devel::DProf represent?"
date: "2025-01-26"
id: "what-do-unstacked-calls-in-outer-in-perls-develdprof-represent"
---

Unstacked calls in the output of Perl's `Devel::DProf` profiler indicate subroutines whose execution time is not accounted for within the call stack of other profiled subroutines. They represent time spent in either top-level code (outside of any explicit subroutine calls) or in subroutines whose entry or exit was missed by the profiler. This situation usually points to profiling limitations or specific code structures that `Devel::DProf` may not handle perfectly, rather than an inherent programming error. I've seen this behavior in several production systems, and it typically requires further investigation to pinpoint its cause and impact.

The core principle behind `Devel::DProf` is that it samples the Perl interpreter's state at regular intervals, recording the current subroutine being executed. These samples form the basis for the profiler's reports. When the profiler encounters a sample not associated with a known stack frame (i.e., a subroutine call established through a `push` on the call stack and not yet popped with a `return`), it categorizes that time as 'unstacked'. Understanding why and where these "unstacked" times occur is critical for accurate performance analysis. It signifies time spent that is not directly attributable to any defined routine within the profiled execution context.

The "outer" part of "unstacked calls in outer" indicates these calls are happening outside of any sampled subroutine. This typically includes time spent in the top-level script or within the Perl interpreter's internal operations that are not part of user-defined subroutines. Sometimes, this may also include time spent in subroutines that were invoked using a mechanism that bypasses the standard Perl call stack management. For example, `eval` with a string argument, calls through XS modules (particularly in cases where they have not correctly incorporated stack walking information), or time consumed by signal handlers are potential candidates.

Let's examine three specific scenarios where unstacked calls might arise, with corresponding code examples:

**Example 1: Time Spent in Top-Level Execution**

```perl
#!/usr/bin/perl
use strict;
use warnings;

my $start = time();
for (my $i = 0; $i < 1000000; $i++) {
    my $sum = 0;
    $sum += $i * $i;
    # Perform some arbitrary calculation
    my $result = sqrt($sum);
}
my $end = time();
print "Time elapsed in top-level code: ", $end - $start, "\n";

sub my_sub {
    my $x = shift;
    return $x * 2;
}

my_sub(10); # Call a subroutine
```

In this scenario, the majority of the execution time is consumed by the for loop directly in the top-level scope. When `Devel::DProf` reports the results, this time won't be attributed to any specific subroutine. It will appear as 'unstacked calls in outer'. The call to `my_sub` does have stack frames and will be properly measured, however the vast majority of execution time is outside the scope of this call. `Devel::DProf` will correctly attribute the very brief execution of `my_sub` and, consequently, the overall time spent will largely be reported as unstacked.

**Example 2: Unprofiled XS Modules**

```perl
#!/usr/bin/perl
use strict;
use warnings;
use Time::HiRes qw(gettimeofday);

my $start = gettimeofday;
my $result = Time::HiRes::gettimeofday(); # Direct call to XS
my $end = gettimeofday;
print "Time elapsed using Time::HiRes::gettimeofday(): ", ($end-$start), "\n";

sub my_sub_2 {
  my $x = shift;
  return $x * 3;
}
my_sub_2(20);
```

`Time::HiRes` is often implemented using XS. If the XS module does not properly integrate with Perl's call stack management for profiling purposes, the execution time spent within `Time::HiRes::gettimeofday` will also appear as unstacked. This arises because the profiler is unaware of the XS module's entry into and exit from execution, so `Devel::DProf` doesn't register the necessary context information. In other words, while the XS calls themselves might be perfectly valid, their impact on the call stack is invisible to `Devel::DProf` unless specific support is provided by the module itself. Here again, `my_sub_2` would be properly profiled but time spent in the XS call would be unstacked.

**Example 3: Code within `eval`**

```perl
#!/usr/bin/perl
use strict;
use warnings;

my $code = 'my $sum = 0; for (my $i = 0; $i < 100000; $i++) {$sum += $i;}; return $sum;';

my $start = time();
my $result = eval($code);
my $end = time();

print "Result of eval: $result\n";
print "Time elapsed in eval: ", $end - $start, "\n";

sub my_sub_3 {
    my $x = shift;
    return $x * 4;
}
my_sub_3(30);
```

In this scenario, the code inside the `eval` string is executed without creating proper stack frames that `Devel::DProf` understands. While the call to `eval` itself may have a stack frame, the code executed within that string doesn't provide the necessary signals for the profiler to track the time spent inside the string. The time spent in eval will be reported as unstacked. As before `my_sub_3` would be properly profiled if it consumed any reasonable amount of time.

In all of these scenarios, the key issue is that the profiler is unable to attribute the execution time to a well-defined subroutine within the sampled context. The time isn’t “lost”; it is simply unaccounted for within the standard profiling framework. It is important to note that "unstacked calls in outer" *do* represent time that the Perl process is using, but *Devel::DProf* is unable to attribute that time to any function that has been sampled.

To properly debug and profile these cases, I often perform several steps, based on past experiences:

First, I’ll closely examine the profiler output to identify which sections of the code have disproportionately large unstacked times.  This initial inspection provides some clues about where to focus. If the unstacked time is significant, it may be necessary to introduce more subroutines to move more of the logic into a profiled context.

Second, when suspicion is focused on XS modules, I will try to confirm if profiling is supported. Consulting the module's documentation, if available, or examining the XS code itself can provide valuable insight.  If necessary, I might look at workarounds or alternative approaches to avoid depending upon that module directly. When working with complex code which incorporates a lot of XS, I will often try replacing the XS calls with a no-op.  If doing so reduces the unstacked execution time, it indicates that the module may be the cause.

Finally, when `eval` is suspected, a common solution is to try converting the code inside eval to a named subroutine. This allows `Devel::DProf` to properly register calls, entry, and exit times and avoids the unstacked categorization. Often, this is an exercise in reducing code complexity and it helps improve code structure.

For further study, I recommend focusing on literature concerning dynamic program analysis.  Specifically, examine work pertaining to call stack inspection and sampling methodologies. Examining documentation for the Perl interpreter itself is sometimes valuable. Research on topics like XS development, especially in the context of creating stack frames, will also prove helpful.  A strong understanding of the Perl runtime internals is useful. Finally, a deeper dive into the source code for `Devel::DProf` can provide further insight into its implementation and limitations. While specific documentation may not be available at a granular level, these resources will allow you to build a thorough understanding of this problem.
