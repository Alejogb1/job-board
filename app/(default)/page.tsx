import Hero from '@/components/hero'
import PressLogos from '@/components/press-logos'
import Sidebar from '@/components/sidebar'
import PostsList from './posts-list'
import Testimonials from '@/components/testimonials'
import Header from '@/components/ui/header'
import SearchBar from '@/components/searchbar'
import { useSearchParams, usePathname, useRouter } from 'next/navigation';
export default function Home({
  searchParams,
}: {
  searchParams?: {
    query?: string;
    loc?: string;
    page?: string;
    filter?: any;
    remote?:string;
  };
}) {
  const query = searchParams?.query || '';
  const location = searchParams?.loc || '';
  const tags = searchParams?.filter || '';
  const remote = searchParams?.remote || '';

  const currentPage = Number(searchParams?.page) || 1;

  return (
    <>
      <Hero/>
      {/*  Page content */}
      <PressLogos/>
      <section>
        <div className="max-w-6xl mx-auto px-6 sm:px">
          <div className="pb-16 md:pb-16 py-4">
            <div className="md:flex md:justify-between" data-sticky-container>
                <Sidebar/>
              {/* Main content */}
              <div className="md:grow">
                <SearchBar/>
                <PostsList remote={remote} tags={tags} location={location} query={query}  currentPage={currentPage} />
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
