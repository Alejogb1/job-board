---
title: "How can Teachable Machine RTD be integrated into a React Native application?"
date: "2025-01-30"
id: "how-can-teachable-machine-rtd-be-integrated-into"
---
Integrating Teachable Machine's real-time detection (RTD) capabilities into a React Native application presents a unique set of challenges, primarily stemming from the platform's reliance on a web-based environment for model execution and React Native’s native mobile orientation. The core difficulty lies in bridging the gap between the web-based TensorFlow.js model produced by Teachable Machine and React Native’s runtime environment. This requires a carefully considered strategy that usually involves native module development or a web-view hybrid approach.

In my experience, having developed several vision-based mobile applications, the most effective approach revolves around utilizing a React Native webview component to host the TensorFlow.js model, coupled with a communication bridge to convey detection data back to the React Native application. This method avoids the complexity of directly implementing the TensorFlow.js runtime on the native side of the React Native application.

The process starts with exporting a trained model from Teachable Machine as a TensorFlow.js model, specifically the “hosted” option which generates a folder containing ‘model.json’ and ‘weights.bin’ files along with an associated index.html file, typically pre-configured to run the model with the device camera. This HTML file becomes the basis for the webview content, but will need modification for communication with the React Native environment.

To facilitate this communication, the webview’s JavaScript context is modified to dispatch custom events when a detection occurs. These events contain the prediction results, which are then intercepted by the React Native application using the webview's `onMessage` prop. This establishes a rudimentary communication bridge, enabling the React Native environment to react to real-time detections. The React Native environment uses `postMessage` to communicate with the webview.

The primary challenge lies in the limitations of the webview, especially regarding camera access on certain operating systems. React Native's built-in camera functionality offers greater control, but it is not directly compatible with the Teachable Machine generated code. Therefore, the ideal situation involves using the camera within the webview, leveraging the pre-configured camera access from the Teachable Machine export.

The approach is divided into the following steps:

1.  **Prepare the Teachable Machine Export:** Obtain the hosted TensorFlow.js model export from Teachable Machine.
2.  **Modify the `index.html`:** Add event listeners for communicating prediction data and expose a way for the webview to receive data from the React Native application.
3.  **Implement React Native Webview:** Incorporate a webview component in the React Native application, loading the modified `index.html`.
4.  **Set Up Communication Bridge:** Utilize `onMessage` and `postMessage` between React Native and the webview, relaying real-time detection information.
5.  **Render Detection Data:** Display the detected classes within the React Native app.

Here are some examples to illustrate the implementation:

**Example 1: Modified `index.html` (JavaScript inside `<script>` tag)**

```javascript
let model;
let webcam;

async function init() {
    const modelURL = "./model.json";
    const metadataURL = "./metadata.json";
    model = await tmImage.load(modelURL, metadataURL);
    const flip = true;
    webcam = new tmImage.Webcam(200, 200, flip);
    await webcam.setup();
    await webcam.play();
    window.requestAnimationFrame(loop);

    // Set up listener for React Native commands
    window.addEventListener('message', (event) => {
        if (event.data === 'start') {
             console.log('Webview started!');
        }
    });
}


async function loop() {
  webcam.update();
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  const prediction = await model.predict(webcam.canvas);
    const predictions = prediction.map(p => ({className: p.className, probability: p.probability}));
    window.ReactNativeWebView.postMessage(JSON.stringify(predictions));

}

init();
```

*This example shows the core logic of the exported Teachable Machine code modified to post predictions back to React Native using `window.ReactNativeWebView.postMessage`.* This assumes that the `index.html` has been placed inside the React Native project.

**Example 2: React Native component (using react-native-webview)**

```jsx
import React, { useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { WebView } from 'react-native-webview';

const DetectionScreen = () => {
    const [detections, setDetections] = useState([]);
    const handleMessage = (event) => {
        try {
            const parsedData = JSON.parse(event.nativeEvent.data);
             setDetections(parsedData);

        } catch(error) {
            console.log('Failed to parse data:', error);
        }
    };

    const webViewRef = React.useRef(null);

    const onWebViewLoad = () => {
        if(webViewRef.current){
            webViewRef.current.postMessage('start');
        }

    };

    return (
        <View style={styles.container}>
            <WebView
                ref={webViewRef}
                style={styles.webview}
                source={require('./assets/web/index.html')} // Adjust path as needed
                onMessage={handleMessage}
                onLoad={onWebViewLoad}
            />
              {detections.length > 0 && (
                <View style={styles.detectionDisplay}>
                  {detections.map((detection, index) => (
                    <Text key={index} style={styles.detectionText}>
                      {`${detection.className}: ${(detection.probability * 100).toFixed(2)}%`}
                    </Text>
                  ))}
                </View>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        marginTop: 20,
    },
    webview: {
        flex: 1,
    },
     detectionDisplay: {
        position: 'absolute',
        bottom: 20,
        left: 20,
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        padding: 10,
        borderRadius: 5,
    },
        detectionText: {
        fontSize: 16,
    },
});

export default DetectionScreen;
```

*This component sets up the webview to load the HTML file. It uses `onMessage` to retrieve data from the webview and updates the application state with the detection results. The `webViewRef` is used to send commands to the webview once loaded, using `postMessage`.*

**Example 3: Minimal React Native UI**

This is a very minimal example, so the `detectionDisplay` in the previous example provides a more robust UI solution.

```jsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const DetectionDataDisplay = ({detections}) => {
    return (
      <View style={styles.container}>
           {detections.map((detection, index) => (
                <Text key={index}>{`${detection.className} : ${(detection.probability* 100).toFixed(2)}%`}</Text>
            ))}
     </View>

    );
}


const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center'
    },
});

export default DetectionDataDisplay;
```

*This is a simple component to display detection data. In a real application, this would likely be styled to match your application’s UI and have more elaborate presentation of the detection information.*

This overall approach creates a reasonably effective real-time object detection setup. The webview renders the camera view through the browser and runs the TensorFlow model, while the React Native application receives and processes the prediction results, facilitating UI updates.

For further exploration, the React Native documentation provides detailed information about the WebView component. Also, researching community resources related to JavaScript-native bridges in React Native, particularly those focused on webview communication, can be beneficial. Finally, TensorFlow.js documentation provides extensive details on how the library is structured and used. Familiarity with basic web development concepts, such as HTML, CSS, and JavaScript is also advantageous. Exploring examples of applications built with the `react-native-webview` library will also prove useful. This will solidify your understanding of the patterns involved with integrating web technologies with React Native applications.
