import Hero from '@/components/hero'
import PressLogos from '@/components/press-logos'
import Sidebar from '@/components/sidebar'
import PostsList from './posts-list'
import Testimonials from '@/components/testimonials'
import Header from '@/components/ui/header'
import SearchBar from '@/components/searchbar'
export default function Home() {
  return (
    <>
      {/*  Page content */}
      <section>
        <div className="max-w-6xl mx-auto px sm:px">
          <div className="py-16 md:py-16">
            <h2 className='text-3xl pb-2 font-semibold text-black mt-6 mb-4'>Search for AI jobs</h2>
            <div className="md:flex md:justify-between" data-sticky-container>
                <Sidebar/>
              {/* Main content */}
              <div className="md:grow">
                <SearchBar/>
                <PostsList />
              </div>

            </div>
          </div>
        </div>
      </section>
    </>
  )
}
