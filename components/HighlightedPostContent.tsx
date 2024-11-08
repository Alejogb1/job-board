// components/HighlightedPostContent.tsx
"use client";

import { useEffect } from 'react';
import hljs from 'highlight.js';
import 'highlight.js/styles/default.css'; // Import your preferred theme

interface HighlightedPostContentProps {
  contentHtml: string;
}

const HighlightedPostContent: React.FC<HighlightedPostContentProps> = ({ contentHtml }) => {
  // Apply syntax highlighting once the component mounts
  useEffect(() => {
    hljs.highlightAll();
  }, []);

  return (
    <div
      className="mt-4"
      dangerouslySetInnerHTML={{ __html: contentHtml }}
    />
  );
};

export default HighlightedPostContent;
