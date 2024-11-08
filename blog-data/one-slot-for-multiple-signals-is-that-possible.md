---
title: "One Slot for Multiple Signals?  Is That Possible?"
date: '2024-11-08'
id: 'one-slot-for-multiple-signals-is-that-possible'
---

```cpp
#include <QtWidgets>

class MyWidget : public QWidget {
    Q_OBJECT
public:
    MyWidget(QWidget *parent = nullptr) : QWidget(parent) {}

public slots:
    void buttonClicked() {
        auto button = qobject_cast<QPushButton*>(sender());
        if (button) {
            // Get the attribute from the button (e.g., using button->property("myAttribute"))
            // and react based on it.
        }
    }

};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MyWidget widget;
    widget.show();

    // ... Create buttons and connect their clicked signals to widget's buttonClicked slot

    return app.exec();
}
```
