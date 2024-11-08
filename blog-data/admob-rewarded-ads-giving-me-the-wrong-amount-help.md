---
title: "AdMob Rewarded Ads Giving Me the Wrong Amount - Help!"
date: '2024-11-08'
id: 'admob-rewarded-ads-giving-me-the-wrong-amount-help'
---

```csharp
private int _rewardedCount;

private void Start() {
    Admob.Instance().rewardedVideoEventHandler += onRewardedVideoEvent;
}

public void ShowRewardVideo() {
    if(Admob.Instance().isRewardedVideoReady()) {
        Admob.Instance().showRewardedVideo();
    } else {
        Admob.Instance().loadRewardedVideo("ca-app-pub-5129395190259237/xxxxxxxxx");
        ShowRewardVideo();
    }
}

void onRewardedVideoEvent(string eventName, string msg) {
    Debug.Log("handler onRewardedVideoEvent---" + eventName + "   " + msg);
    if(eventName == "onRewarded") {
        _rewardedCount++;
        Debug.Log($"Rewarded video count: {_rewardedCount}");
        PlayerPrefs.SetInt("mmig", PlayerPrefs.GetInt("mmig") + 20);
        PlayerPrefs.Save();
    }
}
```
