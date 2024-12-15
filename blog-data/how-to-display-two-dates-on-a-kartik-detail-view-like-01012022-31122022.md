---
title: "How to display two dates on a kartik detail view like 01/01/2022-31/12/2022?"
date: "2024-12-15"
id: "how-to-display-two-dates-on-a-kartik-detail-view-like-01012022-31122022"
---

alright, let's talk about displaying date ranges in a kartik detail view. it sounds like you're aiming to have something like `01/01/2022-31/12/2022` rendered nicely. i've been down this path before, and it's pretty straightforward once you get the hang of it.

first off, let's be clear, kartik isn't a single entity, it's a collection of yii2 widgets. you're most likely talking about kartik's `detailview` widget when you mention detail view. i assume you have two date attributes, say `start_date` and `end_date` that you're trying to display.

from my experience, there are a couple of common pitfalls to avoid here. one is relying on default date formats, which can vary depending on server locales and can lead to a user seeing incorrect date interpretations. another one i've seen is doing the date formatting inside the model. this just isn't the correct location for presentation logic. the model should focus on data, not how it's presented.

so, how do we do it correctly? we handle the formatting within the kartik detailview configuration itself, using the format option. let's break down how i've achieved this in past projects.

here's the basic approach i normally follow, showcasing the usage of `date` format and `php` formatters:

```php
<?php
use kartik\detail\DetailView;
use yii\helpers\Html;

echo DetailView::widget([
    'model' => $model,
    'attributes' => [
        [
            'attribute' => 'start_date',
            'format' => ['date', 'php:d/m/Y'],
            'label' => 'start date',
        ],
        [
            'attribute' => 'end_date',
            'format' => ['date', 'php:d/m/Y'],
             'label' => 'end date',
        ],
        [
          'label' => 'date range',
          'value' => function($model) {
              $start_date = Yii::$app->formatter->asDate($model->start_date, 'php:d/m/Y');
              $end_date = Yii::$app->formatter->asDate($model->end_date, 'php:d/m/Y');
              return "$start_date-$end_date";
            }
        ]
    ],
]);
?>
```

in this code snippet, we are displaying the `start_date` and `end_date` using the `date` format with a `php` modifier which allows us to choose the date representation. the key part here is the custom `value` attribute, which combines these dates into the range string using php. i'm using `Yii::$app->formatter->asDate` to ensure consistent formatting, this is important because direct formatting inside the value function might miss yii's locale and timezone settings. also notice how we define our own label for each attribute, it is good practice.

this example already gives you the date range output. but say that you have many date attributes, wouldn't be cool to do a custom component, that way you do not have to copy paste this `value` function all over the place? i've had a similar situation, so i did it like this:

```php
<?php
namespace app\components;
use Yii;
class DateRangeFormatter
{
  public static function formatRange($startDate, $endDate, $format = 'php:d/m/Y')
  {
    $formattedStartDate = Yii::$app->formatter->asDate($startDate, $format);
    $formattedEndDate = Yii::$app->formatter->asDate($endDate, $format);
    return "$formattedStartDate-$formattedEndDate";
  }
}
```

now, with our custom component we can do it like so:

```php
<?php
use kartik\detail\DetailView;
use app\components\DateRangeFormatter;

echo DetailView::widget([
    'model' => $model,
    'attributes' => [
        [
            'attribute' => 'start_date',
            'format' => ['date', 'php:d/m/Y'],
            'label' => 'start date',
        ],
        [
            'attribute' => 'end_date',
            'format' => ['date', 'php:d/m/Y'],
             'label' => 'end date',
        ],
        [
          'label' => 'date range',
          'value' => function($model) {
              return DateRangeFormatter::formatRange($model->start_date, $model->end_date);
            }
        ]
    ],
]);
?>
```
this is cleaner, more maintainable, and uses the component we just created. this also centralizes the formatting logic. you can then easily change how the date range string looks in one single place.

one last thing that i've used in the past, suppose you want to use a javascript date library, like moment js, this is totally doable inside the `value` property. it is better to do formatting inside php but javascript formatting can be done like so:

```php
<?php
use kartik\detail\DetailView;
use yii\helpers\Html;

echo DetailView::widget([
    'model' => $model,
    'attributes' => [
         [
            'attribute' => 'start_date',
            'format' => ['date', 'php:d/m/Y'],
             'label' => 'start date',
        ],
        [
            'attribute' => 'end_date',
            'format' => ['date', 'php:d/m/Y'],
             'label' => 'end date',
        ],
        [
          'label' => 'date range (js format)',
          'value' => function($model) {
            $start_date = $model->start_date ? Yii::$app->formatter->asDate($model->start_date, 'php:Y-m-d') : null;
            $end_date = $model->end_date ? Yii::$app->formatter->asDate($model->end_date, 'php:Y-m-d') : null;
             if(!$start_date || !$end_date){
              return null;
             }
              $js = <<<JS
                moment("$start_date").format('DD/MM/YYYY') + '-' + moment("$end_date").format('DD/MM/YYYY')
                JS;
              return Html::tag('span', '', ['data-text' => $start_date.' '.$end_date, 'data-moment-range' => true, 'data-moment-format' => 'DD/MM/YYYY', 'data-date-range-start' => $start_date, 'data-date-range-end' => $end_date, 'data-js' => $js]);
            },
         'format' => 'raw'
        ]
    ],
]);
?>
<script>
  $(document).ready(function(){
   $('[data-moment-range="true"]').each(function(){
        let jsCode = $(this).data('js');
        let newHtml = eval(jsCode);
         $(this).text(newHtml)
    })
  })

</script>
```

here we have a very simple javascript, that will grab any html element that has the property `data-moment-range=true` and apply moment js formatting. also notice how i'm sending `Y-m-d` to the `moment()` constructor, that's because it expects a consistent format. also, i'm returning a `span` tag, with `format='raw'` in the `value` property, that way the html is not escaped.

keep in mind that these examples are tailored to fit your case. if your dates are timestamps, you may need to use `Yii::$app->formatter->asTimestamp()` before using `asDate()`. you can also change the php date format string from `php:d/m/Y` to whatever representation you need. if you have timezones in the mix, ensure to configure yii's `formatter` component accordingly.

i've learned the hard way that consistency in date formatting across the application is a blessing. using yii's formatter helps with this and avoids many unexpected date representation issues down the road. one time a client saw a date displayed differently and thought that the data was inconsistent. that was a fun friday afternoon...

regarding resources, i highly recommend reading yii's documentation on `yii\i18n\formatter`, specifically the `asDate` method. php's `date` function manual is also a must-read for understanding date formats. i also suggest reading the kartik's documentation on how attributes work in detail view, there are good examples that help you extend the detail view capabilities. moment js website and documentation can be a good resource, and it provides all the formatting options.

hope this helps you display your dates ranges properly, let me know if you run into any particular cases.
