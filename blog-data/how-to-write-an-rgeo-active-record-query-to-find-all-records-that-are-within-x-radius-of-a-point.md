---
title: "How to write an RGeo Active Record Query to find all records that are within x Radius of a point?"
date: "2024-12-15"
id: "how-to-write-an-rgeo-active-record-query-to-find-all-records-that-are-within-x-radius-of-a-point"
---

alright, so you're looking to snag records within a specific radius of a point using rgeo and activerecord. been there, done that, got the t-shirt – and a few scars from debugging those spatial queries in the wee hours. it's a fairly common scenario when you're working with location-based data.

let me walk you through how i typically tackle this, using a straightforward, hopefully helpful approach. i'm going to assume you've already got rgeo and activerecord all set up and you have a spatial column defined on your model, using a geometry type. if not, that’s a whole different topic, maybe for another day, but i recommend the documentation of both.

the core idea is to create a spatial predicate, and in this case we are using the `dwithin` function of postgis, which will find all records within a given distance of your point. we will create the predicate in activerecord and use it to retrieve records.

first things first, let's take a look at a model. i will call it `location`. let's say your location model has a column named `lonlat`, which holds a point geometry. the column may be named `geom` or any other name but the example is using the `lonlat` column.

```ruby
class Location < ApplicationRecord
  # your other model stuff here
end
```

now, the juicy part. let's say you want to find all locations within a radius of 10 kilometers from a point with longitude -73.9857 and latitude 40.7577. here's how i would craft that activerecord query:

```ruby
  def self.locations_within_radius(longitude, latitude, radius_km)
    point = RGeo::Geographic.spherical_factory.point(longitude, latitude)
    radius_meters = radius_km * 1000
    where(
      "st_dwithin(lonlat, ?, ?)",
      point,
      radius_meters
    )
  end
```

that `st_dwithin` is a postgis function, not built into rgeo itself, but provided by the spatial extension of postgress, if you are using another spatial engine like mysql or oracle the predicate `st_dwithin` may be different. this function checks if the distance between the `lonlat` column and the given `point` is within the specified `radius_meters`.

this approach assumes the `lonlat` field stores the longitude and latitude values correctly, the order is usually longitude then latitude. if it is latitude then longitude, or any other order, you need to be careful, you need to use `st_setsrid(st_makepoint(longitude,latitude),4326)` or the equivalent for the other order. also the coordinate system must match between the point and the `lonlat` column in order for the query to make sense, if they are not on the same coordinate system there are postgis functions to reproject the geometry. usually is with a 4326 srid or sometimes a projected coordinate system if the coordinates are in meters.

it's important to convert your kilometers to meters because that's the unit that postgis's `st_dwithin` operates on. it's a classic gotcha that gets some people on their first spatial queries. i learned this the hard way when i was trying to find locations in the same city and my query would return half the country, turned out i was using kilometers instead of meters.

and here’s how you would use this in practice:

```ruby
  locations = Location.locations_within_radius(-73.9857, 40.7577, 10)
  locations.each { |location| puts "found location: #{location.id}" }
```

this will print all the location ids of the locations within 10km from the coordinates.

now, if you are not working with spherical coordinates (like long/lat) but instead with planar ones you may have to adapt your rgeo factory. the code above uses `rgeo::geographic.spherical_factory` for long/lat coordinates. if you use a different planar projection system you have to adapt it accordingly. usually is `rgeo::cartesian.factory` or some other `factory` object. for example, if you have a coordinate system based on UTM zones you must use `rgeo::cartesian.factory(:srid => 32617)`. here is another example that uses a planar factory, that uses meters as measurement units, usually the coordinate system is in meters or feet:

```ruby
  def self.locations_within_radius_meters_planar(x, y, radius_meters, srid = 32617)
    point = RGeo::Cartesian.factory(:srid => srid).point(x, y)
      where(
        "st_dwithin(lonlat, ?, ?)",
        point,
        radius_meters
      )
  end
```

now, if the radius is also given in other units that are not in the same measure as the coordinate system you must convert the units into the coordinate system units, or reproject your geometry into a different coordinate system to make sense.

and here’s the usage with the example `x,y` coordinates:

```ruby
 locations = Location.locations_within_radius_meters_planar(583942, 4526264, 1000, 32617)
  locations.each { |location| puts "found location: #{location.id}" }
```

this will print all locations within 1km (1000 meters) from the x,y coordinate in the 32617 srid coordinate system.

it is worth noting that the performance of `st_dwithin` can become a bottleneck when dealing with a lot of data. postgis does not index spatial columns by default and the query performance without spatial indexes can be extremely poor. so make sure you create a spatial index to improve performance, if not your query will be slow and could hang your server. something like this would help `create index my_locations_lonlat_idx on locations using gist(lonlat)` on postgresql. the query will be way faster if you have spatial indexes, usually, it makes it 100 times faster. i’ve seen queries go from 10 seconds to 0.1 seconds with just a spatial index.

there's also another very interesting and more efficient approach that avoids the calculations of the distances and uses the bounding box technique. you can generate a bounding box using postgis functions such as `st_expand` and find all the records inside of the box using the `st_intersects` operator, it is faster than `st_dwithin` and it also can be indexed.

here is an example using that approach, the `st_expand` will create a bounding box of a given width around the point and then `st_intersects` will retrieve the records.

```ruby
 def self.locations_within_radius_bounding_box(longitude, latitude, radius_km)
    point = RGeo::Geographic.spherical_factory.point(longitude, latitude)
    radius_meters = radius_km * 1000
    bounding_box = "st_expand(?, ?)"
      where(
        "st_intersects(lonlat, #{bounding_box})",
        point,
        radius_meters
      )
  end
```

and here is an example of its usage:

```ruby
  locations = Location.locations_within_radius_bounding_box(-73.9857, 40.7577, 10)
    locations.each { |location| puts "found location: #{location.id}" }
```

this will find all records intersecting with the bounding box of the radius from the point. this method is faster and is more efficient, you can also index the bounding box queries using spatial indexes.

now, regarding some solid resources to expand your knowledge on this topic:

for a deeper dive into the theoretical aspects of spatial data and algorithms, i highly recommend "geometric algorithms: theory and applications" by mark de berg. it's a classic and will give you a solid foundation.

if you are using postgis, the documentation is really good, and you can find it on the postgis website, but be prepared for a massive documentation. a good book is “postgis in action” by Regina Obe and Leo Hsu. it is a great book for postgis. it is not updated but the core features are still the same.

for activerecord specific spatial stuff, the rgeo-activerecord gem's readme and documentation is always a good place to start and the rails documentation is also very useful for this. the spatial data processing gem documentation is also useful and covers how to parse and import data into your geometry fields.

remember to always test your queries thoroughly, especially the more complex spatial ones. add some logging and make sure you index the spatial column. that's how i usually do it. if you have any further questions feel free to ask.

one last thing, if you are into maps, be prepared to be lost in your way. just a little tech joke to lighten up the subject.
