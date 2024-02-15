// Import necessary components and libraries
import PostItem from './post-item'
import Newsletter from '@/components/newsletter'
import Pagination from '@/components/ui/pagination'
import getFilteredPosts from '@/lib/getFilteredPosts'

import getFilteredTags from '@/lib/getFilteredPostsTags'
import getTagsPosts from '@/lib/sidebar/getJobsRole'
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
export default async function PostsList({tags, location, query, currentPage}:{tags:any, location: string, query:string , currentPage:number}) {
  
  const tagsPostData: Promise<any> = getTagsPosts(tags);
  const tagsPosts:any = await tagsPostData

  const totalPages: number = 9; // Set the total number of pages

  if (tagsPosts) {
      console.log("TAGS RETRIEVED: ", tags)
      const postsData: Promise<any> = getFilteredTags(currentPage, query, location,tags); // Fetch data for the first page
      const posts: Post[] = await postsData;
      return (
        <div className="pb-8 md:pb-16">
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
      );
    } else {
      const postsData: Promise<any> = getFilteredPosts(currentPage, query, location); // Fetch data for the first page
      const posts: Post[] = await postsData;
      return (
        <div className="pb-8 md:pb-16">
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
      );
  }

}