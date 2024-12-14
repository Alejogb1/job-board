---
title: "How to do Custom fields for Powermail in TYPO3?"
date: "2024-12-14"
id: "how-to-do-custom-fields-for-powermail-in-typo3"
---

alright, so custom fields in powermail, yeah, been there, done that, got the t-shirt – probably a few actually. it’s one of those things that seems simple on the surface but can get a little intricate pretty fast. i’ve definitely spent my share of late nights battling with typoscript and fluid templates to get things just perfect. it’s not exactly rocket science, but it requires some precision.

the thing with powermail is, its strength is also its challenge: it’s extremely flexible. this means you’re not just stuck with the standard text input boxes or dropdowns. you can pretty much make it do anything. you want a custom date picker? a fancy image upload? multiple selections with checkboxes? it’s all doable. but here’s where it gets interesting, we need to roll our sleeves up.

let’s talk about the core problem: extending the default powermail form fields with your custom needs. you are not going to modify the core. doing that is like putting ketchup on a rare steak, just don’t. we will use powermail's extension capabilities. in my early days, i remember i tried hacking the core directly, well, let's just say that it was not a fun experience when upgrading to the next version. learned my lesson the hard way and now i try my best to use proper ways to extend, even if it might take a few extra minutes, it saves me hours later. the principle is simple: let the extension be extendable and don't touch the core.

to add a custom field we have to touch three main things: typoscript, fluid template, and the extension configuration in the backend of typo3 (if necessary). let’s tackle each part.

**typoscript setup:**

typoscript is the configuration language for typo3. we're going to use it to tell powermail about our custom field type. this involves creating a custom "marker" that powermail will understand. it’s like adding a new word to powermail’s vocabulary.

here’s the basic typoscript you’ll need to get started:

```typoscript
plugin.tx_powermail {
  settings {
    misc {
        customFieldMarker {
            my_custom_field = TEXT
            my_custom_field {
                value = custom_field_marker
            }
        }
    }
  }
}
```

what this piece of code is doing is simple. we are adding a marker named `my_custom_field`. this `my_custom_field` has a `value` called `custom_field_marker`. now in fluid we can use that marker as a reference to our custom field. `custom_field_marker` will be replaced with the custom html of our field.

now, where exactly do you place this code? well, that depends on your setup. but the most straightforward approach is to add this to your main typoscript template, in the `setup` section, usually, something like `EXT:your_site_package/Configuration/TypoScript/setup.typoscript`. if you don't have that, well, start using it. that's where all your site-specific typoscript should be. you should not be adding typoscript snippets to the `template` module of typo3. try not to do that, it is a bad practice.

**fluid template modification:**

now that we have set up our typoscript, it's time to bring our custom field to life in fluid. we have a reference marker now, we have to make use of it in the powermail fluid template.

powermail's fluid templates are where the html form actually resides. the templates you'll want to be looking at are in `EXT:powermail/Resources/Private/Templates/Form`. if you never touched any of these templates, you are going to need to copy that entire folder to your own extension, something like `EXT:your_site_package/Resources/Private/Templates/Form` and then modify from that folder. if you don't copy and modify, you risk losing those changes with a powermail update. remember that golden rule, never touch the core.

we are going to modify the `Form.html` template, find the section in the file that renders a normal field ` <f:render partial="Form/Field" arguments="{field:field}" />`. now you have to add your custom condition to that field, check if the field's `marker` is equal to the `value` that we set in typoscript before, if it is, render your custom field, if not, just render the normal field. here’s how that might look:

```html
<f:if condition="{field.marker} == '{settings.misc.customFieldMarker.my_custom_field}'">
   <div class="powermail_fieldwrap powermail_fieldwrap_{field.type} {f:if(condition: field.mandatory, then: 'mandatory')}"  id="powermail_fieldwrap_{field.uid}">
      <label for="powermail_field_{field.uid}" class="powermail_label">{field.title} <f:if condition="{field.mandatory}">*</f:if></label>
      <div class="powermail_field">
         <input type="text" name="tx_powermail_pi1[field][{field.uid}]" id="powermail_field_{field.uid}" value="{field.value}" placeholder="{field.placeholder}" />
      </div>
      <f:if condition="{field.error}"><div class="powermail_field_error">{field.error}</div></f:if>
   </div>
<f:else>
  <f:render partial="Form/Field" arguments="{field:field}" />
</f:else>
```

let’s unpack that a little. the `<f:if condition="{field.marker} == '{settings.misc.customFieldMarker.my_custom_field}'">` is the key here. we are checking if the `marker` property of the current field is the same as the `value` we defined in typoscript. if that is true, then we are going to render our custom html, in this case, a simple text input. if not, it goes to the normal field rendering. you can add any html here, not just an input text. you can add your react components, or your custom js code.

also, don't forget to add the `<f:if condition="{field.error}"><div class="powermail_field_error">{field.error}</div></f:if>` line, that displays the error message under the field in case of validation error.

**powermail backend configuration**

now this is where you set up the custom field in the backend and associate that custom marker. you have to go to the `forms` module, then select your form, then go to the fields tab, there you can create a new field. in the new field, go to the advanced tab, and there you will find the marker input field. here you need to put the same `my_custom_field` that you set in the typoscript.

and that’s it. well, not quite, this is just a simple example.

**a little more advanced - a custom select dropdown**

let's say you wanted to add a select dropdown field instead of a simple text input field. here is how the fluid template should look.

```html
<f:if condition="{field.marker} == '{settings.misc.customFieldMarker.my_custom_field}'">
    <div class="powermail_fieldwrap powermail_fieldwrap_{field.type} {f:if(condition: field.mandatory, then: 'mandatory')}" id="powermail_fieldwrap_{field.uid}">
        <label for="powermail_field_{field.uid}" class="powermail_label">{field.title} <f:if condition="{field.mandatory}">*</f:if></label>
        <div class="powermail_field">
            <select name="tx_powermail_pi1[field][{field.uid}]" id="powermail_field_{field.uid}">
                <f:for each="{field.settings.options}" as="option">
                    <option value="{option.value}" {f:if(condition: '{option.value} == {field.value}', then: 'selected="selected"')}>{option.label}</option>
                </f:for>
            </select>
        </div>
        <f:if condition="{field.error}"><div class="powermail_field_error">{field.error}</div></f:if>
    </div>
    <f:else>
        <f:render partial="Form/Field" arguments="{field:field}" />
</f:else>
```

the difference is here we render a `<select>` tag instead of a text input. we also use the `{field.settings.options}` variable to populate the select with available options. in the typo3 backend you need to add the field, then go to the options tab, and then add your options there. each option has a value and a label.

**validation, always a joy**

we have to deal with validation too, and this can become a little tricky. powermail has some built-in validators but those are not enough in a lot of cases. powermail's validation happens in the `Classes/Domain/Validator` directory. you can create a new validator or extend the existing ones. however, this is a topic for another time. let's just say, if you need it, then you need to create your own validation class.

**resources and further learning**

if you want to dive deeper into typoscript, i recommend “typo3: the official guide” (it’s a book available online). it’s the classic reference for all things typoscript, and it helped me a lot in the past. also, read “fluid templating in typo3”. for fluid templating, it’s a great start. you can also take a look at the source code of powermail, it is a great reference of how things are done in a proper way. it is a good practice to always learn from the best. that's how i learned.

the core thing here is that we are extending the core not modifying it, that is the main takeaway from all this, and something i cannot emphasize enough, otherwise you are in for a very long debugging session when you try to update to the next typo3/powermail version.

it is something that you get used to doing after a while, but it might seem a bit verbose at the beginning. now, if you’ll excuse me, i have to go back to a bug, it’s a weird one, it seems like the browser is not loading the styles, wait, maybe i just haven’t cleared the cache… yep, the cache again, it's always the cache, the cache is the biggest enemy of us web developers, well, that, and maybe our own code at times too.
