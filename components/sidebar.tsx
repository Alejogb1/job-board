'use client'
import { useSearchParams, usePathname, useRouter } from 'next/navigation';
import { useCallback } from 'react';
import { useState } from 'react'

export default function Sidebar() {

  const [remoteJobCondition, setremoteJobCondition] = useState<boolean>(false)
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { replace } = useRouter();
  console.log("page refresh")
  const params = new URLSearchParams(searchParams)

  if (remoteJobCondition){
      params.set('remote', "yes");
      replace(`${pathname}?${params.toString()}`);
  } else if(!remoteJobCondition){
    params.delete('remote');
    replace(`${pathname}?${params.toString()}`);
  }
  const handleTag = (term:string, checked:boolean) => {
      const params = new URLSearchParams(searchParams)
      params.set('page', '1')
      if(searchParams.has("filter")) {
        if (term && checked) {
          params.append('filter', term);
        } else if (checked == false) {
          params.delete('filter', term);
        }
      } else if (term && checked == true) {
        params.set('filter', term);
      } else if (checked == false) {
        params.delete('filter');
      }      
      replace(`${pathname}?${params.toString()}`);
    }
    const handleSalaryRangeChange = (input:string) => {
      const params = new URLSearchParams(searchParams)
      params.set('page', '1')
      params.set('salary_range', input)
      replace(`${pathname}?${params.toString()}`);
    }
  return (
    <aside className="hidden lg:block md:block mb-8 md:mb-0 md:w-64 lg:w-72 md:ml-12 lg:ml-20 md:shrink-0 md:order-1">
      <div data-sticky="" data-margin-top="32" data-sticky-for="768" data-sticky-wrap="">
        <div className="relative bg-gray-50 rounded-xl border border-gray-200 p-5">
          <div className="absolute top-5 right-5 leading-none">
            <button className="text-sm font-medium text-indigo-500 hover:underline">Clear</button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-1 gap-6">
            {/* Group 1 */}
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
            </div>
            {/* Group 2 */}
            <div>
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
            </div>
            {/* Group 3 */}
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Remote Only</div>
              <div className="flex items-center">
                <div className="form-switch">
                  <input type="checkbox" id="remote-toggle" className="sr-only" checked={remoteJobCondition} onClick={() => setremoteJobCondition(!remoteJobCondition)} />
                  <label className="bg-gray-300" htmlFor="remote-toggle">
                    <span className="bg-white shadow-sm" aria-hidden="true" />
                    <span className="sr-only">Remote Only</span>
                  </label>
                </div>
                <div className="text-sm text-gray-400 italic ml-2">{remoteJobCondition ? 'On' : 'Off'}</div>
              </div>
            </div>
            {/* Group 3 */}
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Salary Range</div>
              <ul className="space-y-2">
                <li>
                  <button 
                  className="text-sm text-gray-600 cursor-pointer" 
                  onClick={() => handleSalaryRangeChange("less_100")}
                  >Up to $ 100.000</button>
                </li>
                <li>
                  <button className="text-sm text-gray-600 cursor-pointer" onClick={() => handleSalaryRangeChange("100-150")}>$ 100.000 to $ 150.000</button>
                </li>
                <li>
                    <button className="text-sm text-gray-600 cursor-pointer" onClick={() => handleSalaryRangeChange("150-250")}>$ 150.000 to $ 250.000</button>
                </li>
                <li>
                    <button className="text-sm text-gray-600 cursor-pointer" onClick={() => handleSalaryRangeChange("more_250")}>Over $ 250.000</button>
                </li>
                <li>
                  <div className="flex gap-2">
                    <div className="w-1/4">
                      <input type="number" id="number-input" aria-describedby="helper-text-explanation" className="rounded-sm h-4/4 max-w-full relative border border-gray-300 max-w-4/12 text-gray-900 text-xs p-1" placeholder="Min" required/>
                    </div>
                    <span>-</span>
                    <div className="w-1/4">
                      <input type="number" id="number-input" aria-describedby="helper-text-explanation" className="rounded-sm h-4/4 max-w-full relative border border-gray-300 max-w-4/12 text-gray-900 text-xs p-1" placeholder="Max" required/>
                    </div>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}