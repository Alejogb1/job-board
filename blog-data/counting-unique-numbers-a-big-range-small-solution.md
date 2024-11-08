---
title: "Counting Unique Numbers: A Big Range, Small Solution?"
date: '2024-11-08'
id: 'counting-unique-numbers-a-big-range-small-solution'
---

```c#
public class FloatUtil
{
    public static uint ToLexicographicIndex(float value)
    {
        //transfer bits to an int variable
        int signed32 = BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
        uint unsigned32 = (uint)signed32;

        //(0x80000000 - unsigned32) returns 
        //appropriate index for negative numbers
        return (signed32 >= 0)
                   ? unsigned32
                   : 0x80000000 - unsigned32;
    }

    public static uint NumbersBetween(float value1, float value2)
    {
        if (float.IsNaN(value1) || float.IsInfinity(value1))
        {
            throw new ArgumentException("value1");
        }

        if (float.IsNaN(value2) || float.IsInfinity(value2))
        {
            throw new ArgumentException("value2");
        }

        uint li1 = ToLexicographicIndex(value1);
        uint li2 = ToLexicographicIndex(value2);

        //make sure return is positive
        return value1 >= value2 ? li1 - li2 : li2 - li1;
    }
}
```

