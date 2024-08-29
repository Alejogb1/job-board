import Hero from '@/components/hero'
import PressLogos from '@/components/press-logos'
import Sidebar from '@/components/sidebar'
import PostsList from './posts-list'
import SearchField from '@/components/search-field'
import { useSearchParams } from 'next/navigation'
import { useCallback } from 'react'
import getFilteredPosts from '@/lib/getFilteredPosts'
import { useQuery } from '@tanstack/react-query'
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
export default async function Home({
  searchParams,
}: {
  searchParams?: {
    query?: string;
    loc?: string;
    page?: string;
    filter?: any;
    remote?:string;
    salary_range?:string;
  };
}) {
  const query = searchParams?.query || '';
  const location = searchParams?.loc || '';
  const tags = searchParams?.filter || '';
  const remote = searchParams?.remote || '';
  const salary_range = searchParams?.salary_range || '';
  const currentPage = Number(searchParams?.page) || 1;
  console.log(currentPage)
  return (
    <>
      <Hero/>
      <PressLogos/> 
      <section>
        <div className="max-w-6xl mx-auto px-6 sm:px mt-10">
          <div className="">
            <div className="md:flex md:justify-between" data-sticky-container>
              <Sidebar/>
              <h2 className="text-xl font-bold">Building!</h2>
              {/* <div className="md:grow">
                <SearchField />
                { query ? (
                  <PostsList query={query} currentPage={currentPage}/>  
                ): <PostsList query={""} currentPage={currentPage}/> }
              </div> */}
            </div>
          </div>
        </div>
      </section>
    </>
  )
}