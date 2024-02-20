'use client'

import { useEffect } from 'react'
import Header from '@/components/ui/header'

const Sticky = require('sticky-js')

export default function DefaultLayout({
  children,
}: {
  children: React.ReactNode
}) {

  useEffect(() => {
    const stickyEls = document.querySelectorAll('[data-sticky]');
    if (stickyEls.length > 0) {
      const sticky = new Sticky('[data-sticky]');
    }
  })

  return (
    <>      
      <main className="grow">
        <div className="bg-white min-h-screen h-full box-inherit">
          <Header/>
          {children}
        </div>
      </main>
    </>
  )
}
