import Layout from '@/components/layout';
import { getAllPostIds, getPostData as fetchPostData } from '@/lib/blogPosts';
import Head from 'next/head';
import Date from '@/components/date'

export default async function Post({ params }: { params: { id: string } }) {
  const postData = await fetchPostData(params.id) as { id: string; contentHtml: string; date: string; title: string };
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
        <div className="mt-4" dangerouslySetInnerHTML={{ __html: postData.contentHtml }} />
      </article>
    </Layout>
  );
}