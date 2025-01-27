import Head from 'next/head';
import Layout, { siteTitle } from '@/components/layout'
import { getSortedGpuPostsData } from '@/lib/gpuBlogPosts';
import Link from 'next/link';
import Date from '@/components/date';

type PostData = {
  id: string;
  date: string;
  title: string;
};

export default async function BlogPage() {
  const allGpuPostsData = await getSortedGpuPostsData();   
  const totalPosts = allGpuPostsData.length;      
  return (
      <section className="sm:text-md text-lg sm:pt-10 pt-20 mx-auto max-w-2xl">
        <h2 className="text-2xl font-bold">Questions answered about GPU</h2>
        <p className="text-gray-700 mb-4">Total Posts: {totalPosts}</p>
        <ul className="list-disc pl-0">
          {allGpuPostsData.map(({ id, date, title }: PostData) => (
            <div className="mb-2" key={id}>
              <Link href={`/gpu/${id}`} className="text-blue-600 hover:underline">{title}</Link>
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
