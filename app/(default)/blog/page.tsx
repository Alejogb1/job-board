import Head from 'next/head';
import Layout, { siteTitle } from '@/components/layout'
import { getSortedPostsData } from '@/lib/blogPosts';
import Link from 'next/link';
import Date from '@/components/date';

type PostData = {
  id: string;
  date: string;
  title: string;
};

export default async function BlogPage() {
  const allPostsData = await getSortedPostsData();          
  return (
      <section className="sm:text-md text-lg sm:pt-10  pt-20 mx-auto max-w-2xl">
        <h2 className="text-2xl font-bold">Wei Jiang</h2>
        <ul className="list-disc pl-0">
          {allPostsData.map(({ id, date, title }: PostData) => (
            <div className="mb-2" key={id}>
              <Link href={`blog/${id}`} className="text-blue-600 hover:underline">{title}</Link>
              <br />
              <small className="text-gray-500">
                <Date dateString={date} />
              </small>
            </div>
          ))}
        </ul>
      </section>
  );
}

// Remove getStaticProps as it's not supported in the app directory
// export async function getStaticProps(): Promise<{ props: Props }> {
//   const allPostsData = await getSortedPostsData();
//   return {
//     props: {
//       allPostsData,
//     },
//   };
// }
