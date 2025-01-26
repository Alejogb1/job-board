// pages/blog/[id]/page.tsx

import Layout from '@/components/layout';
import { getSortedGpuPostsData, getPostData } from '@/lib/gpuBlogPosts';
import Head from 'next/head';
import Date from '@/components/date';
import HighlightedPostContent from '@/components/HighlightedPostContent';

export async function generateStaticParams() {
  const posts = await getSortedGpuPostsData();
  return posts.map((post) => ({
    id: post.id,
  }));
}

export default async function Post({ params }: { params: { id: string } }) {
  const postData = await getPostData(params.id) as { 
    id: string; 
    contentHtml: string; 
    date: string; 
    title: string 
  };

  return (
    <Layout home={false}>
      <Head>
        <title>{postData.title}</title>
      </Head>
      <article className="max-w-2xl mx-auto p-4">
        <h1 className="text-3xl sm:text-4xl font-bold">{postData.title}</h1>
        <div className="text-gray-500 text-sm sm:text-base">
          <Date dateString={postData.date} />
        </div>
        <HighlightedPostContent contentHtml={postData.contentHtml} />
      </article>
    </Layout>
  );
}
