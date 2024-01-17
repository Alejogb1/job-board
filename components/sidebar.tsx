'use client'

import { useState } from 'react'

export default function Sidebar() {

  const [remoteJob, setRemoteJob] = useState<boolean>(false)

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
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Full-time</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Part-time</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Internship</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Contract / Freelance</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Co-founder</span>
                  </label>
                </li>
              </ul>
            </div>
            {/* Group 2 */}
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Job Roles</div>
              <ul className="space-y-2">
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" defaultChecked />
                    <span className="text-sm text-gray-600 ml-2">Programming</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Design</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Management / Finance</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Customer Support</span>
                  </label>
                </li>
                <li>
                  <label className="flex items-center">
                    <input type="checkbox" className="form-checkbox" />
                    <span className="text-sm text-gray-600 ml-2">Sales / Marketing</span>
                  </label>
                </li>
              </ul>
            </div>
            {/* Group 3 */}
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Remote Only</div>
              <div className="flex items-center">
                <div className="form-switch">
                  <input type="checkbox" id="remote-toggle" className="sr-only" checked={remoteJob} onChange={() => setRemoteJob(!remoteJob)} />
                  <label className="bg-gray-300" htmlFor="remote-toggle">
                    <span className="bg-white shadow-sm" aria-hidden="true" />
                    <span className="sr-only">Remote Only</span>
                  </label>
                </div>
                <div className="text-sm text-gray-400 italic ml-2">{remoteJob ? 'On' : 'Off'}</div>
              </div>
            </div>
            {/* Group 3 */}
            <div>
              <div className="text-sm text-gray-800 font-semibold mb-3">Salary Range</div>
              <ul className="space-y-2">
                <li>
                  <a className="text-sm text-gray-800 cursor-pointer">Up to $ 50.000</a>

                </li>
                <li>
                  <a className="text-sm text-gray-800 cursor-pointer">$ 50.000 to $ 100.000</a>
                </li>
                <li>
                    <a className="text-sm text-gray-800 cursor-pointer">$ 100.000 to $ 200.000</a>
                </li>
                <li>
                    <a className="text-sm text-gray-800 cursor-pointer">More than $ 200.000</a>
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