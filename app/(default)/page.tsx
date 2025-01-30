import Hero from '@/components/hero'
import PressLogos from '@/components/press-logos'
import Sidebar from '@/components/sidebar'
import SearchField from '@/components/search-field'
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
              <div className="md:grow">
                <SearchField />
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}