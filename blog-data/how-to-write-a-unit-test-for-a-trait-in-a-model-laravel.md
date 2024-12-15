---
title: "How to Write a unit test for a trait in a model laravel?"
date: "2024-12-15"
id: "how-to-write-a-unit-test-for-a-trait-in-a-model-laravel"
---

alright, so you're looking to unit test a trait within your laravel model, huh? i've been there, battled that particular beast more times than i care to remember. it’s one of those things that seems simple on the surface but can get tricky pretty quickly if you don't have a solid plan. let me break down how i've tackled this sort of thing in the past, using my own scars as the basis of this advice.

the core issue with unit testing traits is that they're not standalone entities. they're designed to be mixed into classes, particularly your eloquent models. so you can’t test a trait in isolation, you’re always testing it within the context of something else.

my go-to method involves creating a mock model specifically for testing the trait. it's a throwaway class, it exists solely for the purpose of testing, and i don't keep it around longer than i need to. think of it as a temporary lab coat for your trait. it's a little boilerplate but it gives you a solid foundation for your tests. you're essentially building a specific testing environment.

let's say you have a trait that handles timestamps. yeah i know it’s already provided by eloquent. but let’s say you made one that does something custom with that. called `customtimestamps`. it might look like this:

```php
// app/traits/CustomTimestamps.php
namespace app\traits;

use Illuminate\Support\Carbon;

trait CustomTimestamps
{
    public function getCustomCreatedAt(): string
    {
        return Carbon::parse($this->created_at)->format('Y-m-d');
    }

    public function getCustomUpdatedAt(): string
    {
        return Carbon::parse($this->updated_at)->format('Y-m-d');
    }
}
```

now, to test this, i wouldn't try to add it to an existing model. i would create a dedicated test model, it usually resides in my `tests/` directory, somewhere near the test. like so:

```php
// tests/Support/TestModel.php
namespace tests\Support;

use Illuminate\Database\Eloquent\Model;
use app\traits\CustomTimestamps;

class TestModel extends Model
{
    use CustomTimestamps;

    protected $table = 'test_models';
    protected $fillable = ['created_at', 'updated_at'];
    public $timestamps = true; // i forgot to add this one time, spent 3 days looking at why was not updating
}

```

notice that `$timestamps = true`? always remember that! because if you don't have that, your tests will never work and it will look like the trait is the problem when in reality is the lack of timestamps on the model. it's a pain when that happens.

this `testmodel` doesn't need to represent a real database table in your application. it is just a container for the trait. it will be easier for our tests. you'll need to create a migration, just for testing purposes, something like this (i use sqlite for testing so its simple)

```php
// database/migrations/xxxx_create_test_models_table.php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up()
    {
        Schema::create('test_models', function (Blueprint $table) {
            $table->id();
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('test_models');
    }
};
```

run your migrations and get ready to write tests.

then, here’s how a typical test case using phpunit would look:

```php
// tests/Unit/CustomTimestampsTest.php
namespace tests\Unit;

use tests\TestCase;
use tests\Support\TestModel;
use Illuminate\Support\Carbon;

class CustomTimestampsTest extends TestCase
{
    public function test_get_custom_created_at_format()
    {
      $now = Carbon::now();
        $model = TestModel::create(['created_at' => $now]);
        $this->assertEquals($now->format('Y-m-d'), $model->getCustomCreatedAt());
    }

    public function test_get_custom_updated_at_format()
    {
      $now = Carbon::now();
        $model = TestModel::create(['updated_at' => $now]);
        $this->assertEquals($now->format('Y-m-d'), $model->getCustomUpdatedAt());
    }
}
```

in these tests, we’re creating instances of our `testmodel`, setting the timestamps, and then asserting that the trait’s methods work as expected. notice how i'm using `carbon::now()`? that’s intentional. you want consistency and certainty in your tests, using fixed values might introduce unintended problems later on, so i prefer it when the tests automatically adjust themselves when i run them. that way, you know the problem is with your code and not your test data. i’ve spent hours debugging issues where the problem was not the code but a hardcoded date that went past the limit or that had a particular format that was being mistaken by a previous bug i fixed.

now, there’s a bit of a dance with setting up your testing environment, i get it, but it’s a worthwhile investment. it lets you test your traits in isolation without messing with your real models. it’s like having a private playground for experimentation. one time, i got a bug report that said my custom format for timestamps was showing the wrong date by one day. i was scratching my head till i found out it was a timezone issue with my own server’s setup and not the code itself. a properly setup testing environment would have saved me hours of confusion.

for more in-depth information on testing laravel applications, i would suggest the following resources. first, the laravel documentation itself is great. check the section about testing. after that, you could look at "refactoring" by martin fowler, it is not laravel specific but its a book that changed my entire career on software development, mainly the first two chapters. also, "working effectively with legacy code" by michael feathers is good, even if your code is not legacy it has many useful patterns to work with.

remember, when testing, focus on the behavior of your trait, not how it's implemented. avoid coupling your tests to internal implementation details. this makes your tests less brittle. for example, if your trait uses a specific internal format, and you test for that internal format, if you decide to change that internal format, you would need to change your tests as well. your tests should verify the "what" not the "how" of your code.

testing is an art, and it's something you refine over time. one time, i spent hours refactoring some tests that were very tightly coupled to the implementation details and it was like removing a heavy rock from my shoulders when i decoupled the test from the details, it just felt so much cleaner. it also made them very fast because now they do the minimum for each test.

one thing i always do when i encounter a new problem is to check the tests that other developers write. it’s not always useful but sometimes it shows you patterns you might have missed. you can learn much from others' mistakes. for example, i saw this guy that wrote a massive test case for everything and he spent ages making sure everything worked perfectly. then, one little change, broke everything. that is why small specific tests for individual units are important and will save your sanity. (i might have been him that time, not gonna lie)

so, in short, my strategy: create a dedicated mock model, test the trait’s behavior through it, focus on behavior not implementation details, and remember to use consistent testing environments and data. and always double-check for the timestamps being set! it could be a head-scratching experience if you forget it. happy testing! and if you ever get stuck, remember to check the docs.
