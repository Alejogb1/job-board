{/*
Note: This code includes an example of how to fetch data from an external JSON file that is hosted at https://raw.githubusercontent.com/cruip/cruip-dummy/main/job-board-posts.json. To facilitate this, we've included a lib directory in the root which contains a function that can fetch the JSON content. Additionally, we've defined the Post types in the types.d.ts file located in the root.
*/}

import getAllPosts from '@/lib/getAllPosts'
import PostItem from './post-item'
import Newsletter from '@/components/newsletter'
import getCompanies from '@/lib/getCompany'
import Link from 'next/link'
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useSearchParams, usePathname, useRouter } from 'next/navigation';
import Pagination from '@/components/ui/pagination'
import type { InferGetServerSidePropsType, GetServerSideProps } from 'next'

interface Post {
    id: number,
      post_by_id: number,
      is_active: boolean,
      is_remote: boolean,
      is_sponsored: boolean,
      job_title: string,
      job_body: string,
      slug: string,
      job_post_url: string,
      created_at: Date,
  }

  interface PaginationProps {
    totalPages: number
    currentPage: number
  }

export default async function PostsList({
  query,
  currentPage,
}: {
  query: string;
  currentPage: number;
})  {

  const postsData: Promise<any> =  getAllPosts(currentPage)
  const posts:[Post] = await postsData

  const totalPages:number = 9

  console.log(" QUERY: ", query)
  console.log(" PAGE: ", currentPage)

  if(posts){
      return (
        <div className="pb-8 md:pb-16">
          {/* List container */}
          <div className="flex flex-col">
          {posts.map(post => {
              return (
                  <PostItem  key={post.id} {...post}/>
              )
            })}
            {/* Newletter CTA */}
            <div className="py-8 border-b border-gray-200 -order-1">
              <Newsletter />
            </div>
    
          </div>
          <div className="mt-5 flex w-full justify-center">
              <Pagination totalPages={totalPages}/>
          </div>
        </div>
      )
  }
}
