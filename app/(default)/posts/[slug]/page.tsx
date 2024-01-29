{/*
Note: This code includes an example of how to fetch data from an external JSON file that is hosted at https://raw.githubusercontent.com/cruip/cruip-dummy/main/job-board-posts.json. To facilitate this, we've included a lib directory in the root which contains a function that can fetch the JSON content. Additionally, we've defined the Post types in the types.d.ts file located in the root.
*/}

import getAllPosts from '@/lib/getAllPosts'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import PostItem from '../../post-item'
import Newsletter from '@/components/newsletter'
import getCompany from '@/lib/getCompany'
import extractDomain from '@/lib/extractDomain'
import createSlug from '@/lib/slug'

interface Post {
    id: number,
    post_by_id: number,
    is_active: boolean,
    is_remote: boolean,
    is_sponsored: boolean,
    job_title: string,
    job_body: string,
    slug: string,
    job_post_url: string,
    created_at: Date,
}

export async function generateStaticParams() {
  const postsData: Promise<any> = getAllPosts()
  const posts:[Post] = await postsData

  return posts.map(post => ({
    slug: post.slug
  }))
}

export async function generateMetadata({ params }: {
  params: { slug: number }
}): Promise<Metadata> {
  const postsData: Promise<any> = getAllPosts()
  const posts:[Post] = await postsData
  const post = posts.find((post) => post.slug === String(params.slug))

  if (!post) {
    return {
      title: 'Post Not Found'
    }
  }  

  return {
    title: post.job_title,
    description: 'Page description',
  }

}

export default async function SinglePost({ params }: {
  params: { slug: number }
}) {
  function getRandomIntegers(min:number, max:number) {
     // Ensure there's enough room for the gap
      if (max - min < 4) {
        throw new Error('Not enough room for the specified gap.');
      }

      const randomInt1 = Math.floor(Math.random() * (max - min - 4 + 1)) + min;
      const randomInt2 = randomInt1 + 4;

      return [randomInt1, randomInt2];
    }
  
  // Example usage

  
  const postsData: Promise<any> = getAllPosts()
  const posts:[Post] = await postsData

  const post:any = posts.find((post) => post.slug === String(params.slug))

  const minRange = 1;
  const maxRange = posts.length;
  const randomIntegers = getRandomIntegers(minRange, maxRange);
  const companyData: Promise<any> = getCompany(post.company_code)
  const company:any = await companyData
  try {
    posts.slice(randomIntegers[0],randomIntegers[1])
  } catch (error) {
    console.error(error);
  }
  if (!post) {
    notFound()
  }

  return (
    <section>
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="pt-28 pb-8 md:pt-36 md:pb-16">
          <div className="md:flex md:justify-between" data-sticky-container>
            {/* Sidebar */}
            <aside className="mb-8 md:mb-0 md:w-64 lg:w-72 md:ml-12 lg:ml-20 md:shrink-0 md:order-1">
              <div data-sticky data-margin-top="32" data-sticky-for="768" data-sticky-wrap>
                <div className="relative bg-gray-50 rounded-xl border border-gray-200 p-5" >
                  <a className="text-center mb-6 group items-center" href={`/company/${createSlug(company.company.company_name)}`}>
                    <Image className="mx-auto mb-2" src={`https://logo.clearbit.com/${extractDomain(company.company.company_webiste_url)}`} width={72} height={72} alt={post.job_title} />
                    <h2 className="text-lg font-bold text-gray-800 group-hover:underline">{post.job_title}</h2>
                  </a>

                  <div className="flex justify-center md:justify-start mb-5">
                    <ul className="inline-flex flex-col space-y-2">
                      <li className="flex items-center">
                        <svg className="shrink-0 fill-gray-400 mr-3" width="14" height="14" xmlns="http://www.w3.org/2000/svg">
                          <path d="M9.707 4.293a1 1 0 0 0-1.414 1.414L10.586 8H2V2h3a1 1 0 1 0 0-2H2a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h8.586l-2.293 2.293a1 1 0 1 0 1.414 1.414l4-4a1 1 0 0 0 0-1.414l-4-4Z" />
                        </svg>
                        <span className="text-sm text-gray-600">1 week</span>
                      </li>
                      <li className="flex items-center">
                        <svg className="shrink-0 fill-gray-400 mr-3" width="14" height="16" xmlns="http://www.w3.org/2000/svg">
                          <circle cx="7" cy="7" r="2" />
                          <path d="M6.3 15.7c-.1-.1-4.2-3.7-4.2-3.8C.7 10.7 0 8.9 0 7c0-3.9 3.1-7 7-7s7 3.1 7 7c0 1.9-.7 3.7-2.1 5-.1.1-4.1 3.7-4.2 3.8-.4.3-1 .3-1.4-.1Zm-2.7-5 3.4 3 3.4-3c1-1 1.6-2.2 1.6-3.6 0-2.8-2.2-5-5-5S2 4.2 2 7c0 1.4.6 2.7 1.6 3.7 0-.1 0-.1 0 0Z" />
                        </svg>
                        <span className="text-sm text-gray-600">Remote - US</span>
                      </li>
                      <li className="flex items-center">
                        <svg className="shrink-0 fill-gray-400 mr-3" width="16" height="12" xmlns="http://www.w3.org/2000/svg">
                          <path d="M15 0H1C.4 0 0 .4 0 1v10c0 .6.4 1 1 1h14c.6 0 1-.4 1-1V1c0-.6-.4-1-1-1Zm-1 10H2V2h12v8Z" />
                          <circle cx="8" cy="6" r="2" />
                        </svg>
                        <span className="text-sm text-gray-600">{post.tag1}</span>
                      </li>
                    </ul>
                  </div>

                  <div className="max-w-xs mx-auto mb-5">
                    <a className="btn w-full text-white bg-indigo-500 hover:bg-indigo-600 group shadow-sm" target='_blank' href={`${post.job_post_url}`}>
                      Apply Now{' '}
                      <span className="tracking-normal text-indigo-200 group-hover:translate-x-0.5 transition-transform duration-150 ease-in-out ml-1">
                        -&gt;
                      </span>
                    </a>
                  </div>

                  <div className="text-center">
                    <a className="text-sm text-indigo-500 font-medium hover:underline" href={`/company/${createSlug(company.company.company_name)}`}>
                      View company
                    </a>
                  </div>
                </div>
              </div>
            </aside>

            {/* Main content */}
            <div className="md:grow">
              {/* Job description */}
              <div className="pb-8">
                <div className="mb-4">
                  <Link className="text-indigo-500 font-medium" href="/">
                    <span className="tracking-normal">&lt;-</span> All Jobs
                  </Link>
                </div>
                <h1 className="text-4xl font-extrabold font-inter mb-4">{post.job_title}</h1>
                <a
                        className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                        href="#0"
                      >
                        Senior-level / Expert
                </a>
                <a
                        className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                        href="#0"
                      >
                        🌎 Remote
                </a>
                {/* Job description */}
                <div className="space-y-8 mb-8">
                  <div>
                    <div className="text-gray-500 space-y-3">
                      <p><span className='text-gray-800 font-semibold'>Employment type:</span> Full-time</p>                    
                      <p><span className='text-gray-800 font-semibold'>Experience required:</span> 3-5 years</p>                    
                      <p><span className='text-gray-800 font-semibold'>Mission:</span> To help more and more people experience financial well-being</p>                                        </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Opportunity</h3>
                    <div className="text-gray-500 space-y-3">
                      <p>
                        BlackRock is looking for a Market Data Operations Associate to join the Index & Data Solutions team. The Associate will be responsible for handling the end-to-end administration of contracts, maintaining the market data inventory, validating invoices, and supporting the contractual usage declaration process.
                      </p>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Responsibilities</h3>
                    <div className="text-gray-500 space-y-3">
                      <ul className="list-disc list-inside space-y-3">
                        <li>Maintaining the market data inventory, validating invoices, and handling provider change notifications</li>
                      </ul>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-3">Requirements:</h3>
                    <div className="text-gray-500 space-y-3">
                      <ul className="list-disc list-inside space-y-3">
                        <li>Graduate or post-graduate degree in a related field</li>
                        <li>Proficient with Excel and other MS Office applications. Experience with SQL, Power BI, and Python is a plus.</li>
                        <li>3-5 years relevant work experience</li>
                        <li>Knowledge of the Financial Services industry and market and index data providers</li>
                        <li>Strong problem-solving and analytical skills, excellent communication skills (written and verbal)</li>
                        <li>Knowledge of Market and Index Data providers and ability to interpret contract terms and conditions</li>
                      </ul>
                    </div>
                  </div>
                </div>
                {/* Job skills here */}
                <div className="">
                  <h3 className="text-md font-semibold text-gray-800 mb-3">Skills</h3>
                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Big Data
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Computer Science
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Data analysis
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          R
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Research
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          SQL
                  </a>
              </div>
                <div className="">
                  <h3 className="text-md font-semibold text-gray-800 mb-3">Perks</h3>
                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Career development
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Competitive pay
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Fitness / gym
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Flex hours
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Health care
                  </a>

                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Insurance
                  </a>
                  <a
                            className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-indigo-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                            href="#0"
                        >
                          Startup environment
                  </a>
              </div>
                {/* Social share */}
                <div className="flex items-center justify-end space-x-4">
                  <div className="text-xl font-nycd text-gray-400">Share job</div>
                  <ul className="inline-flex space-x-3">
                    <li>
                      <a
                        className="flex justify-center items-center text-indigo-500 bg-indigo-100 hover:text-white hover:bg-indigo-500 rounded-full transition duration-150 ease-in-out"
                        href="#0"
                        aria-label="Twitter"
                      >
                        <svg className="w-8 h-8 fill-current" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                          <path d="m13.063 9 3.495 4.475L20.601 9h2.454l-5.359 5.931L24 23h-4.938l-3.866-4.893L10.771 23H8.316l5.735-6.342L8 9h5.063Zm-.74 1.347h-1.457l8.875 11.232h1.36l-8.778-11.232Z" />
                        </svg>
                      </a>
                    </li>
                    <li>
                      <a
                        className="flex justify-center items-center text-indigo-500 bg-indigo-100 hover:text-white hover:bg-indigo-500 rounded-full transition duration-150 ease-in-out"
                        href="#0"
                        aria-label="Facebook"
                      >
                        <svg className="w-8 h-8 fill-current" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                          <path d="M14.023 24 14 17h-3v-3h3v-2c0-2.7 1.672-4 4.08-4 1.153 0 2.144.086 2.433.124v2.821h-1.67c-1.31 0-1.563.623-1.563 1.536V14H21l-1 3h-2.72v7h-3.257Z" />
                        </svg>
                      </a>
                    </li>
                    <li>
                      <a
                        className="flex justify-center items-center text-indigo-500 bg-indigo-100 hover:text-white hover:bg-indigo-500 rounded-full transition duration-150 ease-in-out"
                        href="#0"
                        aria-label="Telegram"
                      >
                        <svg className="w-8 h-8 fill-current" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                          <path d="M22.968 10.276a.338.338 0 0 0-.232-.253 1.192 1.192 0 0 0-.63.045s-14.019 5.038-14.82 5.596c-.172.121-.23.19-.259.272-.138.4.293.573.293.573l3.613 1.177a.388.388 0 0 0 .183-.011c.822-.519 8.27-5.222 8.7-5.38.068-.02.118 0 .1.049-.172.6-6.606 6.319-6.64 6.354a.138.138 0 0 0-.05.118l-.337 3.528s-.142 1.1.956 0a30.66 30.66 0 0 1 1.9-1.738c1.242.858 2.58 1.806 3.156 2.3a1 1 0 0 0 .732.283.825.825 0 0 0 .7-.622s2.561-10.275 2.646-11.658c.008-.135.021-.217.021-.317a1.177 1.177 0 0 0-.032-.316Z" />
                        </svg>
                      </a>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Related jobs */}
              <div className="mb-8">
                <h4 className="text-2xl font-bold font-inter mb-8">Related Jobs</h4>
                {/* List container */}
                <div className="flex flex-col border-t border-gray-200">
                  {
                  posts.slice(randomIntegers[0],randomIntegers[1]).map(post => {
                    return (
                      <PostItem key={post.id} {...post} />
                    )
                  })}
                </div>
              </div>

              <div>
                <Newsletter />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}