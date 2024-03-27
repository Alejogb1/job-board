'use client'

import { useEffect } from 'react'
import Header from '@/components/ui/header'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const Sticky = require('sticky-js')
// Create a client
const queryClient = new QueryClient()

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
          <QueryClientProvider client={queryClient}>
            {children}
            <ReactQueryDevtools initialIsOpen={false} />
          </QueryClientProvider>
        </div>
      </main>
    </>
  )
}
