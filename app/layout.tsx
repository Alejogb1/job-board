"use client"
import Header from '@/components/ui/header'
import './css/style.css'
import { Analytics } from '@vercel/analytics/react';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { Inter, Nothing_You_Could_Do } from 'next/font/google'
import Footer from '@/components/ui/footer'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap'
})

const nycd = Nothing_You_Could_Do({
  subsets: ['latin'],
  variable: '--font-nycd',
  weight: '400',
  display: 'swap'
})

const queryClient = new QueryClient()

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${nycd.variable} font-inter antialiased bg-white text-gray-800 tracking-tight`}>
        <div className="flex flex-col min-h-screen overflow-hidden supports-[overflow:clip]:overflow-clip">  
          {children}
        </div>
        <Footer/>
        <Analytics />
      </body>
    </html>
  )
}
