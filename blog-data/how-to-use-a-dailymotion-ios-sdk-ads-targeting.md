---
title: "How to use a Dailymotion IOS SDK ADS Targeting?"
date: "2024-12-15"
id: "how-to-use-a-dailymotion-ios-sdk-ads-targeting"
---

alright, so you're looking at dailymotion's ios sdk and trying to get a handle on ad targeting, eh? i've been there, man, more times than i care to remember. it’s one of those things that seems straightforward on paper but can get hairy pretty fast when you start playing around with real-world scenarios. let me walk you through what i've learned, and hopefully, it will save you some head-scratching.

first off, let's talk about the basics. dailymotion's sdk doesn't expose a direct, simple 'setTargeting' function where you can just dump in a dictionary of key-values and be done. instead, it relies on ad parameters that are usually built into the ad request itself. these parameters might be sent along the video player’s initialization, or through a different mechanism provided by dailymotion’s ad server. this means, as a developer, you are not exactly directly "targeting," but more like "influencing" the selection of ads. so how do we actually influence this process? it's about understanding the context and adding the relevant information that dailymotion's ad server will then use to try and show the most relevant ads.

when i first tackled this a few years back, i was working on this news app that had a video section. it was pretty straightforward stuff, the usual ui setup and video playing logic. but, the ads... oh, the ads. we were getting this random mix of adverts that really didn’t align with our audience. a bunch of ads for car parts on a news feed about vegan food was definitely not what we were aiming for. it became really clear that we had to find a way to provide better targeting.

what really helped me understand the flow was reading the documentation of the vast framework underlying the actual ads management. the specific documents aren't directly from dailymotion itself but are industry wide like the iab specifications, they delve deeply into how these parameters are interpreted by different ad networks. so i highly suggest going through those materials. it will give a much better idea of what’s going on behind the scenes.

anyway, let's get into some specifics with code examples. first things first, you need to configure your player. this is where a lot of this contextual data starts to get involved.

```swift
import DailymotionPlayerSDK

class MyViewController: UIViewController {

    var playerView: DMPlayerView!

    override func viewDidLoad() {
        super.viewDidLoad()

        // setup player view
        playerView = DMPlayerView(frame: self.view.bounds)
        self.view.addSubview(playerView)

        // build video parameters, this is where the magic starts
        var videoParameters: [String: Any] = [:]
        videoParameters["custom_param_genre"] = "news"
        videoParameters["custom_param_age"] = "25-34"
        videoParameters["custom_param_location"] = "us"
        videoParameters["custom_param_interest"] = "politics"

        // init player with id and parameters.
        playerView.load(videoId: "x7v405k", parameters: videoParameters)
    }
}
```

here, you can see how we are adding several custom parameters directly to our `videoParameters` dictionary. in this scenario, we are adding a `genre`, `age`, `location` and an `interest` parameter. these are not the only parameters you can send, it really depends on what dailymotion’s ad server accepts. you'll have to refer to their documentation for a complete list. or just play around and see what works...

this is where the art of "influencing" ad targeting becomes really clear. these parameters are interpreted by dailymotion’s system, and it uses them to select ads which are relevant to these parameters. the more relevant information you provide, the more relevant the ads that might be shown. notice that these are not specific dailymotion parameters, the `custom_param_` prefix means that the parameters are handled by the ad server and should ideally match some kind of internal parameter in dailymotion.

now, about those parameters. while dailymotion lets you add custom data like this, it's critical to understand what actually works and what is just noise. for example, if you send a custom age range like '15-17', it’s possible that they don’t have any specific ads targeted at that range, or that the range is processed differently. that’s where the documentation and experimentation come in.

and now for a brief interlude. i once spent a whole friday afternoon staring at a log file only to discover i had accidentally named all my parameters with the same name, talk about a facepalm. if you are using a debugger that might also help. so just check that twice before losing all your hair over it.

moving on, sometimes you might have more complex targeting needs which depend on dynamic values. this might be user preferences, content categories, or time related values. that's where updating these parameters on the fly comes in, so let's say you need to update the parameters after the player has already started:

```swift
import DailymotionPlayerSDK

class MyViewController: UIViewController {

    var playerView: DMPlayerView!

    override func viewDidLoad() {
        super.viewDidLoad()
        // setup player view and load. same code as the previous one.
        playerView = DMPlayerView(frame: self.view.bounds)
        self.view.addSubview(playerView)
        var videoParameters: [String: Any] = [:]
        videoParameters["custom_param_genre"] = "news"
        videoParameters["custom_param_age"] = "25-34"
        videoParameters["custom_param_location"] = "us"
        videoParameters["custom_param_interest"] = "politics"
        playerView.load(videoId: "x7v405k", parameters: videoParameters)
    }

    func updateAdTargeting(location: String, interest: String) {
         var newParameters = playerView.parameters // get current parameters
         newParameters["custom_param_location"] = location;
         newParameters["custom_param_interest"] = interest;
         playerView.parameters = newParameters // update parameters
         playerView.reload() // reload to reflect new ad targeting.
    }

    // sample function to simulate a change
    func buttonTapped() {
        self.updateAdTargeting(location: "uk", interest: "sports")
    }

}
```

in this snippet, we have a function called `updateAdTargeting`. it takes a new location and interest, updates the `playerView.parameters` and calls `playerView.reload()` this will trigger the system to load new ads, using the provided updated parameters. this is particularly useful for apps with changing contexts. like, if a user changes their profile information, you could potentially refresh the ad parameters, or in the example provided it is simulating a button tap.

also, while on the subject, remember to always handle ad events, the dailymotion sdk will notify you when an ad starts, finishes, or if there are any errors. this can be done with the provided delegate methods, this is important for tracking purposes and for diagnosing ad delivery problems.

finally, one thing that was very helpful for me was using a proxy tool to inspect the actual network requests and responses that the sdk was making to the dailymotion’s ad server. this gave me a much better idea of exactly what data was being sent, and how the server was responding. this way, when ads didn’t play, or where not as relevant as i thought, i was able to know why. tools like charles proxy or fiddler can be invaluable for troubleshooting and fine tuning your ad targeting.

so, to wrap up, targeting ads with dailymotion’s ios sdk is more about influencing ad selection through contextual data that you pass within your player parameters. and remember, iab documentation will provide a deep background into the mechanisms of ad delivery. you set custom parameters when you initialize the player, then, if required, update them with a player reload. and finally, always inspect your requests to ensure that everything is working as expected. you’ll get a better feel for what works by experimenting. and you are good to go.
