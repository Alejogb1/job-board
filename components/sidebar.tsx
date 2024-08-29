'use client'
// import { useSearchParams, usePathname, useRouter } from 'next/navigation';
// import { useCallback, useEffect } from 'react';
// import { useState } from 'react'
// import PropTypes from 'prop-types';
import Link from 'next/link'

export default function Sidebar() {
  // const router= useRouter();
  // const searchParams = useSearchParams();
  // const pathname = usePathname();
  // const params = new URLSearchParams(searchParams)

  //  const handleTag = (term:string, checked:boolean) => {
  //     const params = new URLSearchParams(searchParams)
  //     params.set('page', '1')
  //     if(searchParams.has("filter")) {
  //       if (term && checked) {
  //         params.append('filter', term);
  //       } else if (checked == false) {
  //         params.delete('filter', term);
  //       }
  //     } else if (term && checked == true) {
  //       params.set('filter', term);
  //     } else if (checked == false) {
  //       params.delete('filter');
  //     }      
  //     router.replace(`${pathname}?${params.toString()}`);
  //   }

  //   const handleSalaryRangeChange = (input:string, event:any) => {
  //     event.preventDefault
  //     const params = new URLSearchParams(searchParams)
  //     params.set('page', '1')
  //     params.set('salary_range', input)
  //     router.replace(`${pathname}?${params.toString()}`);
  //   }  
  return (
    <aside className="hidden lg:block md:block mb-8 md:mb-0 md:w-64 lg:w-72 md:ml-12 lg:ml-20 md:shrink-0 md:order-1">
      <div data-sticky="" data-margin-top="32" data-sticky-for="768" data-sticky-wrap="">
        <div className="relative bg-gray-50 rounded-xl border border-gray-200 p-5">
          {/* <div className="absolute top-5 right-5 leading-none">
            <button className="text-sm font-medium text-indigo-500 hover:underline">Clear</button>
          </div> */}
          {/* <div className="grid grid-cols-2 md:grid-cols-1 gap-6">
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Job Type</div>
              <ul className="space-y-2">
                <li>
                  <label className="flex items-center">
                    <input 
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="fulltime"
                      type="checkbox" 
                      className="form-checkbox" 
                    />
                    <span className="text-sm text-gray-600 ml-2">Full-time</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input 
                      type="checkbox" 
                      className="form-checkbox"
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="partTime"
                    />
                    <span className="text-sm text-gray-600 ml-2">Part-time</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input 
                    type="checkbox" 
                    className="form-checkbox" 
                    onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                    value="intern"
                    />
                    <span className="text-sm text-gray-600 ml-2">Internship</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input 
                    type="checkbox" 
                    className="form-checkbox" 
                    onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                    value="contract"
                    />
                    <span className="text-sm text-gray-600 ml-2">Contract / Freelance</span>
                  </label>
                </li>
              </ul>
            </div> */}
            {/* <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Job Level</div>
              <ul className="space-y-2">
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" 
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="senior"
                    />
                    <span className="text-sm text-gray-600 ml-2">Senior-level / Expert</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" 
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="mid"
                    />
                    <span className="text-sm text-gray-600 ml-2">Mid-level / Intermediate</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" 
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="entry"
                    />
                    <span className="text-sm text-gray-600 ml-2">Entry-level / Junior</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" 
                      onChange={(e) => {handleTag(e.target.value, e.target.checked);}}
                      value="executive"
                    />
                    <span className="text-sm text-gray-600 ml-2">Executive-level / Director</span>
                  </label>
                </li>
              </ul>
            </div> */}

            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Salary Range</div>
              <div className="space-y-2">
                <div>
                  <Link 
                  className="text-sm text-gray-600 cursor-pointer" 
                  href={{
                    pathname: '/salary/100',
                  }}>
                    Up to $ 100,000
                  </Link>
                </div>
                <div>
                  <Link className="text-sm text-gray-600 cursor-pointer" href="/salary/100-150">$ 100,000 to $ 150,000</Link>
                </div>
                <div>
                  <Link className="text-sm text-gray-600 cursor-pointer" href="/salary/100-250">$ 150,000 to $ 250,000</Link>
                </div>
                <div>
                  <Link className="text-sm text-gray-600 cursor-pointer" href="/salary/250">Over $ 250,000</Link>
                </div>
              </div>        
              </div>
          </div>
      </div>
    </aside>
  )
}
