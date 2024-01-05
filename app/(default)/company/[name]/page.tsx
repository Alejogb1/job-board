import Link from 'next/link'
import Header from '@/components/ui/header'
import Sidebar from '@/components/sidebar'
export default function CompanyProfile() {
  return (
        <div className="content lg:max-w-6xl mx-auto my-0 lg:pt-20">
            <div className="styles-box">
                <div className="ml-2 lg:ml-6">
                    <div className="space-x-1 text-dark-a flex flex-row items-center">
                        <a className="text-xs" href="/discover">Discover</a>
                        <svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="w-2"><path fill-rule="evenodd" clip-rule="evenodd" d="M8 21a.997.997 0 00.707-.293l8-8a.999.999 0 000-1.414l-8-8a.999.999 0 10-1.414 1.414L14.586 12l-7.293 7.293A.999.999 0 008 21z" fill="currentColor"></path></svg>
                        <a className="text-xs" href="/discover/startups">Company</a>
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
                            <div className="component-header pt-10">
                                <div className="mx-8 pt-2 border-b px-2 ">
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
                            <div className="profile flex flex-row align-start w-full">
                                <section className='lg:py-6 lg:px-8'>
                                    <h1 className='text-xl font-medium text-dark-aaaa antialiased'>Jobs at Thumbstack</h1>
                                    <div className="mb-6">
                                        <div className="text-xs">
                                        Thumbtack empowers people to DO. We empower customers to get more done and small business owners to do the work they love.
                                        </div>
                                    </div>
                                    <div className="styles-browser flex">
                                        <div className="filter md:w-42 lg:w-72">
                                            <aside className="md:shrink-0 md:order-1">
                                                <div data-sticky="" data-margin-top="32" data-sticky-for="768" data-sticky-wrap="">
                                                    <div className="relative bg-gray-50 rounded-sm border border-gray-200 p-5">
                                                    <div className="grid grid-cols-2 md:grid-cols-1 gap-6">
                                                        {/* Group 1 */}
                                                        <div className="text-xs text-gray-400 font-semibold">FILTER BY</div>
                                                        <div>
                                                            <div className="text-xs text-gray-800 font-semibold mb-3">TEAM</div>
                                                            <ul className="space-y-2">
                                                                <li>
                                                                <label className="flex items-center">
                                                                    <input type="checkbox" className="form-checkbox" />
                                                                    <span className="text-xs text-gray-600 ml-2">Full-time</span>
                                                                </label>
                                                                </li>
                                                                <li>
                                                                <label className="flex items-center">
                                                                    <input type="checkbox" className="form-checkbox" />
                                                                    <span className="text-xs text-gray-600 ml-2">Part-time</span>
                                                                </label>
                                                                </li>
                                                                <li>
                                                                <label className="flex items-center">
                                                                    <input type="checkbox" className="form-checkbox" />
                                                                    <span className="text-xs text-gray-600 ml-2">Intership</span>
                                                                </label>
                                                                </li>
                                                                <li>
                                                                <label className="flex items-center">
                                                                    <input type="checkbox" className="form-checkbox" />
                                                                    <span className="text-xs text-gray-600 ml-2">Contract / Freelance</span>
                                                                </label>
                                                                </li>
                                                                <li>
                                                                <label className="flex items-center">
                                                                    <input type="checkbox" className="form-checkbox" />
                                                                    <span className="text-xs text-gray-600 ml-2">Co-founder</span>
                                                                </label>
                                                                </li>
                                                            </ul>
                                                        </div>
                                                        {/* Group 2 */}
                                                        <div>
                                                        <div className="text-xs text-gray-800 font-semibold mb-3">TYPE</div>
                                                        <ul className="space-y-2">
                                                            <li>
                                                            <label className="flex items-center">
                                                                <input type="checkbox" className="form-checkbox" />
                                                                <span className="text-xs text-gray-600 ml-2">$20K - $50K</span>
                                                            </label>
                                                            </li>
                                                            <li>
                                                            <label className="flex items-center">
                                                                <input type="checkbox" className="form-checkbox" />
                                                                <span className="text-xs text-gray-600 ml-2">$50K - $100K</span>
                                                            </label>
                                                            </li>
                                                            <li>
                                                            <label className="flex items-center">
                                                                <input type="checkbox" className="form-checkbox" />
                                                                <span className="text-xs text-gray-600 ml-2">&gt; $100K</span>
                                                            </label>
                                                            </li>
                                                            <li>
                                                            <label className="flex items-center">
                                                                <input type="checkbox" className="form-checkbox" />
                                                                <span className="text-xs text-gray-600 ml-2">Drawing / Painting</span>
                                                            </label>
                                                            </li>
                                                        </ul>
                                                        </div>
                                                        {/* Group 3 */}
                                                        <div>
                                                        <div className="text-xs text-gray-800 font-semibold mb-3">Location</div>
                                                        <label className="sr-only">Location</label>
                                                        <select className="form-select w-full">
                                                            <option>Anywhere</option>
                                                            <option>London</option>
                                                            <option>San Francisco</option>
                                                            <option>New York</option>
                                                            <option>Berlin</option>
                                                        </select>
                                                        </div>
                                                    </div>
                                                    </div>
                                                </div>
                                            </aside>
                                        </div>
                                        <div className="jobslist filter-1 relative overflow-hidden min-w-0 box-inherit w-full">
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                            <div className="styles-component flex border-b border-gray-200 py-2 border-w-3/4">
                                                    <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
                                                            <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0"/>
                                                                    <div>
                                                                        <div className="flex datas-start space-x-2">
                                                                            <div className="text-sm text-gray-500 font-semibold mb-1">Operations</div>
                                                                        </div>
                                                                        <div className="mb-2">
                                                                                <a className="text-lg text-gray-800 font-bold" href="/posts/1">Software Engineer Backend</a>   
                                                                        </div>
                                                                        <div className="-m-1">
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">Full Time</a>
                                                                            <a className="text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 rounded-md m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-100" href="#0">🌎 Remote</a>
                                                                        </div>
                                                                    </div>
                                                                    <div className="min-w-[60px] flex items-center lg:justify-end space-x-1 lg:space-x-0">
                                                                        <div className="lg:hidden group-hover:lg:block">
                                                                            <a className="btn-sm py-1.5 px-3 text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" href="/posts/1">Apply Now <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">-&gt;</span></a>
                                                                        </div>
                                                                        <div className="group-hover:lg:hidden text-sm italic text-gray-500">2h</div>
                                                                    </div>
                                                            </div>
                                            </div>
                                        </div>  
                                    </div>
                                </section>
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
