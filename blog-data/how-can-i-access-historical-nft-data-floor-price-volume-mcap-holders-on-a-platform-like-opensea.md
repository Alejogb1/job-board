---
title: "How can I access historical NFT data (floor price, volume, MCAP, holders) on a platform like OpenSea?"
date: "2024-12-23"
id: "how-can-i-access-historical-nft-data-floor-price-volume-mcap-holders-on-a-platform-like-opensea"
---

Okay, let's tackle this. Accessing historical NFT data – floor price, volume, market cap (MCAP), holders – on a platform like OpenSea isn’t always straightforward, especially when you need granularity beyond what's readily available on the user interface. I've navigated these waters quite a few times in my past projects, and the key lies in understanding how these platforms expose (or sometimes *don't* expose) their data. We’re not talking about a single ‘magic’ API endpoint, but a collection of techniques and strategies.

Let’s start with a caveat: OpenSea, like many similar marketplaces, doesn't offer a single comprehensive public API that directly provides the exact historical data we’re seeking in a nicely packaged format. They often prioritize current, real-time data for obvious business reasons. Consequently, historical retrieval usually involves a combination of their available APIs, potentially some data scraping, and, depending on the level of precision required, relying on third-party data aggregators.

Firstly, OpenSea *does* have a publicly available API (though not always fully documented), which we can leverage for some specific information. For instance, we can use their 'events' endpoint to get a history of sales transactions, which is pivotal for reconstructing historical floor prices. Keep in mind, however, that this method has limitations: you have to page through the results, which can be numerous, and the response doesn’t give us MCAP directly; we need to calculate that.

Here's a simplified example using Python and the `requests` library to fetch sale events for a particular collection (you'll need to find the collection's slug on OpenSea):

```python
import requests
import json

def get_historical_sales(collection_slug, limit=300, next_cursor=None):
    url = f"https://api.opensea.io/api/v1/events?collection_slug={collection_slug}&event_type=successful&limit={limit}"
    if next_cursor:
        url += f"&cursor={next_cursor}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    response.raise_for_status() # raise exception for bad status codes
    return response.json()

def extract_sales_data(response_json):
    sales = []
    for event in response_json.get("asset_events", []):
        if event.get("event_type") == "successful":
            try:
                sale_price = float(event['payment_token']['eth_price'])
                sales.append({
                    "timestamp": event['created_date'],
                    "price": sale_price
                })
            except (KeyError, TypeError):
                # Handle cases where the eth_price may be missing
                continue
    return sales, response_json.get("next", None)

def fetch_all_sales(collection_slug):
  all_sales = []
  next_cursor = None
  while True:
    response = get_historical_sales(collection_slug, next_cursor=next_cursor)
    sales, next_cursor = extract_sales_data(response)
    all_sales.extend(sales)
    if not next_cursor:
      break
  return all_sales

if __name__ == "__main__":
    collection_slug = "boredapeyachtclub" # Example collection
    sales_data = fetch_all_sales(collection_slug)
    print(json.dumps(sales_data, indent=2))

```

This snippet will fetch recent sales for the "boredapeyachtclub" collection, including timestamps and sale prices. Notice the use of the 'next_cursor'. OpenSea's API, like many others, paginates responses, so we need to iterate to get all the data. The price information is given as 'eth_price' which should be converted to float type to allow for calculations. This gives us the raw material, but we have to process it further. To calculate the floor price at a specific time, you would need to sort these sales by time and then determine the minimum price at a given point. Similarly, volume can be calculated by aggregating sales within a time interval.

Calculating Market Cap (MCAP) from this requires knowing the total number of items in the collection, which we can retrieve from the same API (or, more practically from the collection metadata) and the current floor price. Note, however, that getting the historical MCAP is an approximation; it would involve combining historical sales data with floor price calculations at corresponding points in time.

Now, about retrieving holder data. OpenSea's API doesn’t expose this directly in a time-series format. You'll only retrieve the current holders via an `assets` endpoint for a specific collection, or for individual tokens. To get historical data on holders, we require a different strategy, and a much more complex one:

```python
import requests
import json
import datetime

def fetch_asset_owners(collection_slug, limit=50):
  url = f"https://api.opensea.io/api/v1/assets?collection_slug={collection_slug}&limit={limit}"
  headers = {"Accept": "application/json"}
  response = requests.get(url, headers=headers)
  response.raise_for_status()
  return response.json()

def extract_owner_data(response_json):
    owners = {}
    for asset in response_json.get("assets", []):
        owner = asset.get("owner",{})
        owner_address = owner.get("address")
        if owner_address:
            owners[owner_address] = owners.get(owner_address, 0) + 1
    return owners

def estimate_historical_holders(collection_slug, sales_data, timestamp):
    #this is a simplified model - assumes sales directly lead to a change of holder, which might not be entirely accurate in real life
    holders = set()
    current_timestamp = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    for sale in sales_data:
        sale_timestamp = datetime.datetime.fromisoformat(sale['timestamp'].replace("Z", "+00:00"))
        if sale_timestamp <= current_timestamp:
            # we use a dummy owner address here
            holders.add(sale.get("seller", "dummy_seller")) # seller is missing in open sea api response - only for demonstration
            holders.add(sale.get("buyer", "dummy_buyer")) # buyer is missing in open sea api response - only for demonstration

    return len(holders)


if __name__ == "__main__":
    collection_slug = "boredapeyachtclub"
    sales_data = fetch_all_sales(collection_slug)
    target_time = sales_data[len(sales_data)//2]['timestamp'] #lets pick a timestamp around mid sales
    estimated_holders = estimate_historical_holders(collection_slug, sales_data, target_time)
    print(f"Estimated number of holders at time {target_time}: {estimated_holders}")

    #Example current owners
    current_owners = extract_owner_data(fetch_asset_owners(collection_slug))
    print(f"Current holders number: {len(current_owners)}")

```

This snippet showcases how to get *current* holders and it provides a very rough and basic idea how one can estimate the holder number at a given point in time. Note that the `sales_data` we retrieved in the first code example does not contain the seller or buyer info, so here we use a dummy owner, purely for the demo purpose. In real world, a more reliable method might be using chain data using Web3 libraries, or third party services that do track this historical data.
It's important to emphasize this is not accurate, as secondary transactions between two existing holders will not change the number of holders, and also some sales could be with an exchange or escrow service which is not a real person. OpenSea’s API doesn't reveal historical holders, thus an approximation method or more sophisticated third party data integration are necessary.

For more reliable and granular historical NFT data, consider exploring tools and datasets provided by companies like Nansen, Dune Analytics, or Footprint Analytics. They often use on-chain data and more advanced methods to track changes in ownership, volumes, etc. As for academic resources, for an in-depth understanding of blockchain data analysis, the book "Mastering Bitcoin" by Andreas Antonopoulos is invaluable. For an understanding of time-series data analysis techniques (useful for analyzing price and holder changes), "Time Series Analysis: Forecasting and Control" by George Box, Gwilym Jenkins, and Gregory Reinsel is a standard reference. Papers discussing blockchain analysis, such as those from academic conferences like IEEE Symposium on Security and Privacy or ACM Conference on Computer and Communications Security, often delve into these types of data retrieval and analytical techniques, focusing on on-chain analysis where possible for more accuracy.

In conclusion, accessing historical NFT data on OpenSea requires a multi-pronged approach. While the OpenSea API is a starting point, you'll likely need to augment this with other techniques and resources to acquire the detailed, historical data you need. My past work has shown it's rarely a simple API call, but it can be accomplished with some processing and the proper tooling. Remember to always respect rate limits and data use policies when interacting with any platform's API.
