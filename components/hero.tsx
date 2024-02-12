import Link from 'next/link'
import Image from 'next/image'
import Illustration from '@/public/images/hero-illustration.svg'
import Avatar01 from '@/public/images/avatar-01.jpg'
import Avatar02 from '@/public/images/avatar-02.jpg'
import Avatar03 from '@/public/images/avatar-03.jpg'
import Avatar04 from '@/public/images/avatar-04.jpg'

export default function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Bg */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-100 to-white pointer-events-none -z-10" aria-hidden="true" />

      {/* Illustration */}
      <div className="hidden md:block absolute left-1/2 -translate-x-1/2 pointer-events-none -z-10" aria-hidden="true">
        <Image src={Illustration} className="max-w-none" priority alt="Hero Illustration" />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="pt-28 pb-2 md:pt-36 md:pb-6">
          {/* Hero content */}
          <div className="max-w-3xl text-center md:text-left">
            {/* Copy */}
            <h1 className="h1 font-inter mb-6">
              Get the best tech jobs in <span className="font-nycd text-indigo-500 font-normal">Artificial Intelligence</span>
            </h1>
            <p className="text-lg text-gray-500 mb-8">
              Secure top-tier tech roles, drive future breakthroughs in Artificial Intelligence
              <br className="hidden md:block" /> with our intelligent job search platform.
            </p>
            
          </div>
        </div>
      </div>
    </section>
  )
}
