import Head from 'next/head';
import Layout, { siteTitle } from '@/components/layout'
import { getSortedPostsData } from '@/lib/blogPosts';
import Link from 'next/link';
import Date from '@/components/date';

type PostData = {
  id: string;
  date: string;
  title: string;
  contentHtml: string;
};

export default async function BlogPage() {
  const allPostsData:any = await getSortedPostsData();

  const totalPosts = allPostsData?.length || 0;

  return (
    <section className="sm:text-md text-lg sm:pt-10 pt-20 mx-auto max-w-2xl">
      <h2 className="text-2xl font-bold">Wei Jiang</h2>
      <p className="text-gray-700 mb-4">Total Posts: {totalPosts}</p>
      <ul className="list-disc pl-0">
        {allPostsData?.map(({ id, date, title }: PostData) => (
          <div className="mb-2" key={id}>
            <Link href={`blog/${id}`} className="text-blue-600 hover:underline">
              {title}
            </Link>
            <br />
            <small className="text-gray-500">{date}</small>
          </div>
        ))}
      </ul>
    </section>
  );
}