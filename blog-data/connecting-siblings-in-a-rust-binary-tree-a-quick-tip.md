---
title: "Connecting Siblings in a Rust Binary Tree: A Quick Tip"
date: '2024-11-08'
id: 'connecting-siblings-in-a-rust-binary-tree-a-quick-tip'
---

```rust
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Default)]
pub struct TreeNode {
    left: Option<Rc<TreeNode>>,
    right: Option<Rc<TreeNode>>,
    sibling: RefCell<Option<Weak<TreeNode>>>,
    v: u8,
}

impl TreeNode {
    pub fn new(v: u8) -> Rc<Self> {
        Rc::new(TreeNode {
            v,
            ..TreeNode::default()
        })
    }
    pub fn new_with(left: Option<Rc<TreeNode>>, right: Option<Rc<TreeNode>>, v: u8) -> Rc<Self> {
        Rc::new(TreeNode {
            left,
            right,
            v,
            sibling: RefCell::new(None),
        })
    }
    pub fn set_siblings(self: &Rc<Self>) {
        let Some(left) = self.left() else { return };
        let right = self.right();

        left.sibling.replace(right.map(Rc::downgrade));

        if let Some(sibling) = self.sibling() {
            right
                .unwrap()
                .sibling
                .replace(sibling.left().map(Rc::downgrade));
        }

        left.set_siblings();
        right.map(|r| r.set_siblings());
    }

    pub fn left(&self) -> Option<&Rc<Self>> {
        self.left.as_ref()
    }
    pub fn right(&self) -> Option<&Rc<Self>> {
        self.right.as_ref()
    }
    pub fn sibling(&self) -> Option<Rc<Self>> {
        self.sibling.borrow().as_ref()?.upgrade()
    }
}

fn main() {
    let t = TreeNode::new_with(
        TreeNode::new_with(TreeNode::new(1).into(), TreeNode::new(2).into(), 3).into(),
        TreeNode::new(4).into(),
        5,
    );
    t.set_siblings();

    assert_eq!(t.left().and_then(|l| l.sibling()).unwrap().v, 4);
    let ll = t.left().and_then(|l| l.left());
    assert_eq!(ll.map(|ll| ll.v), Some(1));
    ll.unwrap().sibling().unwrap();
    assert_eq!(
        t.left()
            .and_then(|l| l.left())
            .and_then(|ll| ll.sibling())
            .unwrap()
            .v,
        2
    );
}
```
