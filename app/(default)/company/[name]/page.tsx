import Link from 'next/link'
import Header from '@/components/ui/header'
export default function CompanyProfile() {
  return (
        <div className="content lg:max-w-6xl mx-auto my-0 lg:pt-20">
            <div className="styles-box">
                <div className="ml-2 lg:ml-6">
                    <div className="space-x-1 text-dark-a flex flex-row items-center">
                        <a className="text-xs" href="/discover">Discover</a>
                        <svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="w-2"><path fill-rule="evenodd" clip-rule="evenodd" d="M8 21a.997.997 0 00.707-.293l8-8a.999.999 0 000-1.414l-8-8a.999.999 0 10-1.414 1.414L14.586 12l-7.293 7.293A.999.999 0 008 21z" fill="currentColor"></path></svg>
                        <a className="text-xs" href="/discover/startups">Startups</a>
                        <svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="w-2"><path fill-rule="evenodd" clip-rule="evenodd" d="M8 21a.997.997 0 00.707-.293l8-8a.999.999 0 000-1.414l-8-8a.999.999 0 10-1.414 1.414L14.586 12l-7.293 7.293A.999.999 0 008 21z" fill="currentColor"></path></svg>
                        <a className="text-xs" href="/company/thumbtack">Thumbtack</a>
                    </div>
                </div>
                <div className="styles-content">
                    <section className="max-w-6xl mx-auto sm:px-6 mt-10 max-h-20">
                        <div className="max-h-8 flex flex-row items-center">
                            <div className="logo mr-3">
                                <a className="">
                                    <div className="overflow-hidden inline-flex flex-row items-center relative border border-gray-200 bg-gray-100 rounded-md h-18 w-18">
                                        <img className='rounded-md'  height="70" width="70" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTFR-psuzkHzPNlfPTZDenQbbaGWQz4TtNtPaRlyVJXJYgktUrLJuTjmN6BypL8w5oRQs&usqp=CAU" alt="" />
                                    </div>
                                </a>
                            </div>
                            <div className="name">
                                <div className="name flex flex-row space-between max-h-8">
                                    <h3 className="text-xl font-semibold font-inter mb-10">Thumbstack</h3>
                                </div>
                                <h2 className=''>A technology company helping millions of people care for and improve their homes</h2>
                            </div>
                        </div>
                    </section>
                    <div className="styles-main flex flex-row space-between items-start">
                        <div className="w-full">
                            <div className="component-header">
                                <div className="mx-8 pt-2 border-b px-2 pt-10">
                                    <nav className='m-0 flex align-end justify-start flex-row min-w-0 min-h-unset w-full pt-unset'>
                                        <a href="" className='leading-5 pb-1'>
                                            <div className="flex flex-row items-center text-xs">
                                                <span>Overview</span>
                                            </div>
                                        </a>
                                        <a href="" className='leading-5 pb-1 ml-8'>
                                            <div className="flex flex-row items-center text-xs">
                                                <span>People</span>
                                            </div>
                                        </a>
                                        <a href="" className='leading-5 pb-1 ml-8'>
                                            <div className="flex flex-row items-center text-xs">
                                                <span>Culture and benefits</span>
                                            </div>
                                        </a>
                                        <a href="" className='leading-5 pb-1 ml-8'>
                                            <div className="flex flex-row items-center text-xs">
                                                <span>Funding</span>
                                            </div>
                                        </a>
                                        <a href="" className='leading-5 pb-2 ml-8 flex flex-row border-b-2 border-gray-500 font-semibold'>
                                            <div className="flex flex-row items-center text-xs">
                                                <span>Jobs</span>
                                            </div>
                                            <div className="ml-1 border-solid border-gray-200 text-center font-medium uppercase leading-none antialiased p-1 text-xs bg-gray-200 text-gray-800 rounded-md">
                                                <div className="flex justify-center gap-1">
                                                    <span>8</span>
                                                </div>
                                            </div>
                                        </a>
                                    </nav>
                                </div>
                            </div>
                        </div>
                        <div className="component-sidebar"></div>

                        <aside></aside>
                    </div>
                </div>
                
            </div>
        </div>
  )
}
