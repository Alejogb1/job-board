// app/sitemap.xml/route.ts

import { getAllPostIds } from '@/lib/blogPosts';
import { getAllPosts } from '@/lib/getAllPosts'; // Assume you have this function or create it
import { NextResponse } from 'next/server';

function generateSiteMap(blogPosts: { id: string }[], jobPosts: { slug: string; url: string }[]) {
  const baseUrl = 'https://www.jobseekr.ai';

  // Generate XML entries for blog posts
  const blogPostUrls = blogPosts
    .map((post) => {
      return `
    <url>
      <loc>${baseUrl}/blog/${post.id}</loc>
      <changefreq>daily</changefreq>
      <priority>0.8</priority>
    </url>`;
    })
    .join('');

  // Generate XML entries for job posts
  const jobPostUrls = jobPosts
    .map((post) => {
      return `
    <url>
      <loc>${post.url}</loc>
      <changefreq>daily</changefreq>
      <priority>0.7</priority>
    </url>`;
    })
    .join('');

  // Final sitemap structure with main website and combined blog & job posts
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${baseUrl}</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  ${blogPostUrls}
  ${jobPostUrls}
</urlset>`;
}

export async function GET() {
  const blogPostsWithParams = await getAllPostIds();
  const blogPosts = blogPostsWithParams.map((post) => post.params); // Adjust shape if needed

  const jobPosts = await getAllJobPostSlugs(); // Fetch job post slugs and URLs

  const sitemap = generateSiteMap(blogPosts, jobPosts);

  return new NextResponse(sitemap, {
    status: 200,
    headers: {
      'Content-Type': 'application/xml',
    },
  });
}
