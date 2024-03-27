import Link from 'next/link'
import Image from 'next/image'
import Illustration from '@/public/images/hero-illustration.svg'

export default function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Bg */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-100 to-white pointer-events-none -z-10" aria-hidden="true" />

      {/* Illustration */}
      <div className="hidden md:block absolute left-1/2 -translate-x-1/2 pointer-events-none -z-10" aria-hidden="true">
        <Image src={Illustration} className="max-w-none" priority alt="Hero Illustration" />
      </div>

      <div className="max-w-6xl mx-auto px-4 mt-6 lg:mt-0 sm:px-6">
        <div className="pt-12 lg:pt-24 pb-2">
          {/* Hero content */}
          <div className="max-w-3xl text-center md:text-left">
            {/* Copy */}
            <h1 className="h1 font-inter mb-2">
              Get the best tech jobs in <span className="font-nycd text-black-500 font-normal">Artificial Intelligence</span>
            </h1>
            <p className="text-lg text-gray-500">
              Secure top-tier tech roles, drive future breakthroughs in Artificial Intelligence
              <br className="hidden md:block" /> with our intelligent job search platform.
            </p>
            
          </div>
        </div>
      </div>
    </section>
  )
}
