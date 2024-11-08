---
title: "How to Visualize Outlier Detection Performance with ROC Curves"
date: '2024-11-08'
id: 'how-to-visualize-outlier-detection-performance-with-roc-curves'
---

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<process version="6.2.000">
  <context>
    <input/>
    <output/>
    <macros/>
  </context>
  <operator activated="true" class="process" compatibility="6.2.000" expanded="true" name="Process">
    <process expanded="true">
      <operator activated="true" class="generate_data" compatibility="6.2.000" expanded="true" height="60" name="Generate Data" width="90" x="45" y="75">
        <parameter key="target_function" value="gaussian mixture clusters"/>
        <parameter key="number_examples" value="200"/>
        <parameter key="number_of_attributes" value="2"/>
      </operator>
      <operator activated="true" breakpoints="after" class="anomalydetection:Local Outlier Factor (LOF)" compatibility="2.3.000" expanded="true" height="94" name="Local Outlier Factor (LOF)" width="90" x="179" y="120">
        <parameter key="nominal_measure" value="JaccardSimilarity"/>
        <parameter key="parallelize evaluation process" value="true"/>
      </operator>
      <operator activated="true" breakpoints="after" class="anomalydetection:Generate ROC" compatibility="2.3.000" expanded="true" height="130" name="Generate ROC (2)" width="90" x="380" y="120">
        <parameter key="label value for outliers" value="outlier"/>
        <parameter key="label value for normal data" value="normal"/>
      </operator>
      <connect from_op="Generate Data" from_port="output" to_op="Local Outlier Factor (LOF)" to_port="example set"/>
      <connect from_op="Local Outlier Factor (LOF)" from_port="example set" to_op="Generate ROC (2)" to_port="example set"/>
      <connect from_op="Generate ROC (2)" from_port="roc set" to_port="result 1"/>
      <portSpacing port="source_input 1" spacing="0"/>
      <portSpacing port="sink_result 1" spacing="0"/>
      <portSpacing port="sink_result 2" spacing="0"/>
    </process>
  </operator>
</process>
```
