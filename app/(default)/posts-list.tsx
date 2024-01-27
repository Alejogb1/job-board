{/*
Note: This code includes an example of how to fetch data from an external JSON file that is hosted at https://raw.githubusercontent.com/cruip/cruip-dummy/main/job-board-posts.json. To facilitate this, we've included a lib directory in the root which contains a function that can fetch the JSON content. Additionally, we've defined the Post types in the types.d.ts file located in the root.
*/}

import getAllPosts from '@/lib/getAllPosts'
import PostItem from './post-item'
import Newsletter from '@/components/newsletter'
import getCompanies from '@/lib/getCompany'
import { usePathname } from 'next/navigation'
import Link from 'next/link'
interface Post {
    jobs : { id: number,
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
  }

  interface PaginationProps {
    totalPages: number
    currentPage: number
  }

  function Pagination({ totalPages, currentPage }: PaginationProps) {
    const pathname = usePathname()
    const basePath = pathname.split('/')[1]
    const prevPage = currentPage - 1 > 0
    const nextPage = currentPage + 1 <= totalPages
  
    return (
      <div className="space-y-2 pb-8 pt-6 md:space-y-5">
        <nav className="flex justify-between">
          {!prevPage && (
            <button className="cursor-auto disabled:opacity-50" disabled={!prevPage}>
              Previous
            </button>
          )}
          {prevPage && (
            <Link
              href={currentPage - 1 === 1 ? `/${basePath}/` : `/${basePath}/page/${currentPage - 1}`}
              rel="prev"
            >
              Previous
            </Link>
          )}
          <span>
            {currentPage} of {totalPages}
          </span>
          {!nextPage && (
            <button className="cursor-auto disabled:opacity-50" disabled={!nextPage}>
              Next
            </button>
          )}
          {nextPage && (
            <Link href={`/${basePath}/page/${currentPage + 1}`} rel="next">
              Next
            </Link>
          )}
        </nav>
      </div>
    )
  }
export default async function PostsList() {
  const postsData: Promise<any> = getAllPosts()
  const posts:[Post] = await postsData

  if(posts){
      return (
        <div className="pb-8 md:pb-16">
          {/* List container */}
          <div className="flex flex-col">
          {posts.jobs.map(post => {
              return (
                <PostItem key={post.id} {...post}/>
              )
            })}
            {/* Newletter CTA */}
            <div className="py-8 border-b border-gray-200 -order-1">
              <Newsletter />
            </div>
    
          </div>
        </div>
      )
  }
}
