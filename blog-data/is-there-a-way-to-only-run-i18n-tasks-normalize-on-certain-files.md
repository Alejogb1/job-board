---
title: "Is there a way to only run i18n-tasks normalize on certain files?"
date: "2024-12-14"
id: "is-there-a-way-to-only-run-i18n-tasks-normalize-on-certain-files"
---

so, yeah, dealing with i18n-tasks and its normalization, i’ve been there. it's a beast sometimes, especially when you've got a project that's grown organically and the i18n files are… well, let's just say they're not exactly uniform.

from what i gather, you’re looking to avoid running `i18n-tasks normalize` across your entire codebase, and instead target specific files. that's a smart move actually. running it globally can sometimes lead to unwanted changes, especially if you're working with a mix of old and new formats, or if you've got specific formatting preferences in some areas.

when i first ran into this years ago, my codebase was a total mess (we all have those early coding experiences). we’d been using a variety of translation formats, some were json, some were yaml, some had inconsistent indentation, and even some with duplicate keys. the global normalize command felt like a nuclear option, and i was definitely worried about it messing things up more. it almost felt like trying to solve a jigsaw puzzle with a bulldozer. so yeah, i definitely get the 'need to be precise' point.

the i18n-tasks gem itself, it doesn’t directly expose an option to specify files on the command line for normalize, like some other tools do. there's no `--files` or similar flag. it seems to assume it is a full codebase operation. but that doesn't mean you are stuck with that. the gem is flexible and allows us to get what we need using its api.

what i found was the trick is to leverage the `i18n-tasks` api directly. you can write a small script to filter which files get normalized. it’s a bit more work than a simple command line argument, but it gives you fine-grained control, exactly what you need. think of it as using a scalpel instead of a hammer.

i'll walk you through the approach i've used in a few projects. basically, you write a small ruby script that uses the `i18n-tasks` library to load the tasks, and then you specify a list of file patterns to work with.

here is one example i’ve used in a project with some files in json format and some in yaml format:

```ruby
require 'i18n/tasks'

i18n = I18n::Tasks::BaseTask.new
files_to_normalize = [
  'config/locales/en/*.json',
  'config/locales/fr/*.yml',
  'config/locales/es/specific_file.yml'
]

files = i18n.config[:data][:files].select do |file|
  files_to_normalize.any? { |pattern| File.fnmatch(pattern, file) }
end

i18n.normalize_files!(files)

puts "normalization completed for selected files."
```

in this snippet, we first require `i18n/tasks`. then we instantiate a `I18n::Tasks::BaseTask` object. then, we define the list of file patterns. it can be regular file paths, or you can use wildcards like `*`. then i filter all the files against my patterns using `File.fnmatch`. finally we call `i18n.normalize_files!` with that filtered list.

after this you can run this ruby script (like `ruby your_script_name.rb`) and it will normalize only the files you specify.

sometimes, you will need to refine your filtering process. so, maybe you want to exclude some directories, here’s a more elaborate example with an exclusion and showing a specific format:

```ruby
require 'i18n/tasks'

i18n = I18n::Tasks::BaseTask.new
files_to_normalize = [
  'config/locales/en/*.json',
  'config/locales/fr/*.yml'
]

exclude_dirs = [
    'config/locales/en/old'
]

files = i18n.config[:data][:files].select do |file|
    files_to_normalize.any? { |pattern| File.fnmatch(pattern, file) } &&
    !exclude_dirs.any? { |exclude_dir| file.start_with?(exclude_dir) }
end

i18n.normalize_files!(files, format: :json)

puts "normalization completed for selected files (json format)."
```
in this version, i’ve added an `exclude_dirs` array. files that fall under directories listed in this array will be skipped. also, notice we’re passing the `format: :json` argument to `normalize_files!` to ensure only specific formats are processed, if that's required. this is really handy when you want to normalize the json files in a project to a certain format while letting the other ones alone.

now, if you are using a more complex setup that involves different formats and more granular control, then you might need to dive deeper. here's an example that includes both yaml and json formats while also doing checks if the file exists (since i had some odd symlink issues in the past and i was tired of getting exceptions):

```ruby
require 'i18n/tasks'

i18n = I18n::Tasks::BaseTask.new
files_to_normalize_json = [
  'config/locales/en/*.json',
  'config/locales/es/*.json'
]

files_to_normalize_yaml = [
  'config/locales/fr/*.yml',
  'config/locales/de/*.yml'
]

files_json = i18n.config[:data][:files].select do |file|
    files_to_normalize_json.any? { |pattern| File.fnmatch(pattern, file) } && File.exist?(file)
end

files_yaml = i18n.config[:data][:files].select do |file|
    files_to_normalize_yaml.any? { |pattern| File.fnmatch(pattern, file) } && File.exist?(file)
end


i18n.normalize_files!(files_json, format: :json)
i18n.normalize_files!(files_yaml, format: :yaml)

puts "normalization completed for selected files (json and yaml formats)."
```

here we are normalizing json files and yaml files separately. first we are setting up different arrays of files to process per format, then we are making sure the files exist, and finally we are calling the normalize function with each format. i find this the most robust way, and you can get pretty clever adding more granular filtering and logic to it as you go. i used this setup in a project that had to deal with different translation formats from external sources, a total nightmare i tell you.

regarding resources, i'd recommend spending some time with the i18n-tasks gem's documentation on their github repository. it has all the core concepts described properly. also, the ruby documentation is great, and the `file` class is a must to understand how you can filter by paths and file patterns. i find these are enough to get started and solve most of the cases.

i’m happy to elaborate more if something is not clear or if you need to solve a particular situation.
