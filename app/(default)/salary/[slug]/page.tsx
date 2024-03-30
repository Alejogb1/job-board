'use client'

import Sidebar from "@/components/sidebar";
import PostsList from "../../posts-list";
import { useRouter } from 'next/router'
import { useEffect } from "react";

export default function Page() {

    return (
      <>
        <section className="pt-12 lg:pt-24 pb-2">
          <div className="max-w-6xl mx-auto px-6 sm:px mt-10">
            <div className="">
              <div className="md:flex md:justify-between" data-sticky-container>
                <Sidebar/>
                <div className="md:grow">
                    <p className='ml mb-4 text-xs'>No posts found for this</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </>
    )
  }