// app/sitemap.xml/route.ts

import { getAllPostIds } from '@/lib/blogPosts';
import { NextResponse } from 'next/server';

function generateSiteMap(posts: { id: string }[]) {
  const baseUrl = 'https://www.jobseekr.ai';

  // Structure each blog post entry with consistent XML formatting
  const postUrls = posts
    .map((post) => {
      return `
    <url>
      <loc>${baseUrl}/blog/${post.id}</loc>
      <changefreq>daily</changefreq>
      <priority>0.8</priority>
    </url>`;
    })
    .join('');

  // Final sitemap structure with header and main website entry
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${baseUrl}</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>${postUrls}
</urlset>`;
}

export async function GET() {
  const postsWithParams = await getAllPostIds();
  const posts = postsWithParams.map((post) => post.params); // Adjust shape if needed

  const sitemap = generateSiteMap(posts);

  return new NextResponse(sitemap, {
    status: 200,
    headers: {
      'Content-Type': 'application/xml',
    },
  });
}
