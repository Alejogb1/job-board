'use server'
// Import necessary components and libraries
import PostItem from './post-item'
import Newsletter from '@/components/newsletter'
import Pagination from '@/components/ui/pagination'
import getFilteredPosts from '@/lib/getFilteredPosts'
import getFilteredPostsWithTags from '@/lib/getFilteredPostsTags'
import getTagsPosts from '@/lib/sidebar/getJobsRole'
import { usePosts } from '@/lib/usePosts'
// Define the Post interface
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
// Define the component
export default async function PostsList({query, currentPage }:{query: string, currentPage: number}) {
      const postsData: Promise<any> = getFilteredPosts(currentPage, query); // Fetch data for the first page
      const posts: Post[] = await postsData; 
      console.log("data acquired ", posts.length)
      const totalPages: number = 9; // Set the total number of pages
      if (posts.length > 0) {
        return (  
          <div className="pb-8 md:pb-16">
          {query ? (
          <p className="ml mb-4 text-xs">
          {posts.length === 0
            ? 'There are no posts that match '
            : `Showing ${posts.length} results for `}
          <span className="font-semibold">{query}</span>
          </p>
          ) : null}
          {/* List container */}
          <div className="flex flex-col">
              {posts.map(post => {
                  return (
                    <PostItem key={post.id} {...post} />
                  );
                })}
            {/* Newletter CTA */}
            <div className="py-8 border-b border-gray-200 -order-1">
              <Newsletter />
            </div>
          </div>
          <div className="mt-5 flex w-full justify-center">
            <Pagination totalPages={totalPages} />
          </div>
        </div>
        )       
      } 
    return (
      <p className='ml mb-4 text-xs'>No posts found for {query}</p>
    )
  } 
