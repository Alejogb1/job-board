// components/HighlightedPostContent.tsx
"use client";

import { useEffect } from 'react';
import MarkdownIt from 'markdown-it';
import 'highlight.js/styles/default.css';
import styles from './HighlightedPostContent.module.css';

interface HighlightedPostContentProps {
  contentHtml: string;
}

const HighlightedPostContent: React.FC<HighlightedPostContentProps> = ({ contentHtml }) => {
  useEffect(() => {
    const md = new MarkdownIt({
      html: true,
      linkify: true,
      typographer: true,
      breaks: true,
      tables: true
    });
    
    // Enable table support
    md.enable('table');
  }, []);

  return (
    <div
      className={`mt-4 prose prose-slate max-w-none ${styles.markdown}`}
      dangerouslySetInnerHTML={{ __html: contentHtml }}
    />
  );
};

export default HighlightedPostContent;
