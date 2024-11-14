  
  import type { Metadata } from 'next'
  import { notFound } from 'next/navigation'
  import Link from 'next/link'
  import Image from 'next/image'
  import PostItem from '../../post-item'
  import Newsletter from '@/components/newsletter'
  import getCompany from '@/lib/getCompany'
  import extractDomain from '@/lib/extractDomain'
  import {getAllPosts} from '@/lib/getAllPosts'
  import createSlug from '@/lib/slug'
  import { MDXRemote } from 'next-mdx-remote/rsc'
  import ReactMarkdown from 'react-markdown'
  import React from 'react'
  import { render } from 'react-dom'
  import Job from "../../../_posts/job.mdx"
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


  export default async function SinglePost({ params }: {
    params: { slug: number }
  }) {
    function getRandomIntegers(min: number, max: number) {
      // Ensure there's enough room for the gap
        if (max - min < 4) {
          // Adjusted error message for clarity
          throw new Error(`Not enough room for the specified gap. Ensure max is at least ${min + 4}.`);
        }

        const randomInt1 = Math.floor(Math.random() * (max - min - 4 + 1)) + min;
        const randomInt2 = randomInt1 + 4;

        return [randomInt1, randomInt2];
      }
    const postsData: Promise<any> = getAllPosts()
    const posts:[Post] = await postsData
    const md = `

  **Loom Video:**

  Our Founder/CEO, [Gabe Greenberg](https://twitter.com/gabe_g2i), created a more in depth Loom video that we highly recommend you watch! Check it out here: [https://www.loom.com/share/a58abfd8e16a43e499d81c59f123db4d](https://www.loom.com/share/a58abfd8e16a43e499d81c59f123db4d)

  **Overview:**

  You’ll join an expert annotation team to create training data for the world's most advanced AI models. No previous AI experience is necessary. You'll get your foot in the door with one of the biggest players in the AI/LLM space today. We are seeking eager to learn future developers who wants to join an experience to train AI large language models, helping cutting-edge generative AI models write better code. Projects typically include discrete, highly variable problems that involve engaging with these models as they learn to code.

  **What Will I Be Doing:**

  \- You will have a 4 days of paid training when you start.

  \- Evaluating the quality of AI-generated code, including human-readable summaries of your rationale.

  \- Writing robust test cases to confirm code works efficiently and effectively.

  **Pay Rate:**

  \- US Based (50 USD /hr)

  \- Expectations are 15+ hours per week however there is no upper limit. We have engineers working 20-40 hrs per week and some that are working 40+ hours per week. You can work as much as you want to. You'll get paid on a weekly basis. You get paid per hour of work done on the platform.

  **Contract Length:**

  \- Contract is long term, there is no end date.

  \- You can end the contract at any time. Our hope is that you would commit to 6 months of work, but if you start and it's not a fit for you, we totally understand.

  **Flexible Schedules:**

  You are expected to work 15+ hours a week with the ability to work 40+ hours. Developers can set their own hours. Take a 3-hour lunch, no problem. You are paid according to time spent on the platform, which is calculated in the coding exercises, as opposed to tracking your own hours.

  **Interview Process:**

  1) Apply using this Ashby form.

  2) If you look like a good fit, we'll send an async video interview (4 minutes) with two questions.

  3) You'll perform a 35-minute live technical interview at any stack with the G2i technical interview team. If you solidly pass, you'll get the job. We are performing these interviews on behalf of the company.

  4) We'll set up a group call to answer further questions about the role and the company and can help you sign G2i's terms of service as a developer.

  5) We'll then work with the company to help get you onboarded.

  **Tech Stack Priorities:**

  JavaScript, Python, Java, C

  **Required Qualifications:**

  \- You are currently attending or have graduated from an Ivy League college, Top 50 college in the US, or University of Waterloo in Canada

  \- Professional programming experience is NOT required but intro level computer science and and data structure courses are required.

  \- Complete fluency in the English language.

  \- Ability to articulate complex technical concepts clearly and engagingly.

  \- Excellent attention to detail and ability to maintain consistency in writing.
    
    `
    const post:any = posts.find((post) => post.slug === String(params.slug))

    const minRange = 1;
    const maxRange = posts.length;
    const randomIntegers = getRandomIntegers(minRange, maxRange);
    const companyData = post.company_code ? getCompany(post.company_code) : Promise.resolve(null);
    const company: any = await companyData;
    const companyName = company?.company?.company_name || 'Company Name Not Found';
    const companyUrl = company?.company?.company_webiste_url || 'https://example.com';
    const logoUrl = company?.company?.company_webiste_url
      ? `https://logo.clearbit.com/${extractDomain(company.company.company_webiste_url)}`
      : '/default-logo.png'; // Fallback to a default logo
  
  /*   const bufferData = Buffer.from(post.job_body);
    const descriptionString = bufferData.toString('utf-8');

  */  try {
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
                    <div className="text-center mb-6 group items-center">
                      <Image className="mx-auto mb-2" src={`https://logo.clearbit.com/${extractDomain(companyUrl)}`} width={72} height={72} alt={post.job_title} />
                      <p className="text-lg font-bold text-gray-800">{companyName}</p>
                    </div>
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
                          <span className="text-sm text-gray-600">{post.region}</span>
                        </li>
                        <li className="flex items-center">
                          <svg className="shrink-0 fill-gray-400 mr-3" width="16" height="12" xmlns="http://www.w3.org/2000/svg">
                            <path d="M15 0H1C.4 0 0 .4 0 1v10c0 .6.4 1 1 1h14c.6 0 1-.4 1-1V1c0-.6-.4-1-1-1Zm-1 10H2V2h12v8Z" />
                            <circle cx="8" cy="6" r="2" />
                          </svg>
                          <span className="text-sm text-gray-600">{post.min_salary <= 0 ? "Salary range not found" : `USD ${post.min_salary}K - ${post.max_salary}K`}</span>
                        </li>
                      </ul>
                    </div>

                    <div className="max-w-xs mx-auto mb-5">
                      <a className="btn w-full text-white bg-black hover:bg-gray-600 group shadow-sm" target='_blank' href={`${post.job_post_url}`}>
                        Apply Now{' '}
                      </a>
                    </div>

                    <div className="text-center">
                      <a className="text-sm text-gray-500 font-medium hover:underline cursor-pointer" href={companyUrl} target='_blank'>
                        Visit company
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
                    <Link className="text-gray-500 font-medium" href="/">
                      <span className="tracking-normal">&lt;-</span> All Jobs
                    </Link>
                  </div>
                  <h1 className="text-4xl font-extrabold font-inter mb-4">{post.job_title}</h1>
                  <a
                          className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                          href="#0"
                        >
                          Senior-level / Expert
                  </a>
                  <a
                          className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                          href="#0"
                        >
                          🌎 Remote
                  </a>
                  {/* Job description */}
                  <div className="space-y-8 mb-8">
                  <ReactMarkdown>{String(post.job_body)}</ReactMarkdown>
                  </div>
                  {/* Job skills here */}
                  <div className="">
                    <h3 className="text-md font-semibold text-gray-800 mb-3">Skills</h3>
                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Big Data
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Computer Science
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Data analysis
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            R
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Research
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            SQL
                    </a>
                </div>
                  <div className="">
                    <h3 className="text-md font-semibold text-gray-800 mb-3">Perks</h3>
                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Career development
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Competitive pay
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Fitness / gym
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Flex hours
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Health care
                    </a>

                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Insurance
                    </a>
                    <a
                              className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 bg-gray-50 rounded-md mr-1 mb-5 whitespace-nowrap transition duration-150 ease-in-out`}
                              href="#0"
                          >
                            Startup environment
                    </a>
                </div>
                  {/* Social share */}
                  <div className="flex items-center justify-end space-x-4">
                    {/* <div className="text-xl font-nycd text-gray-400">Share job</div> */}
                    <ul className="inline-flex space-x-3">
                      <li>
                        <a
                          className="flex justify-center items-center text-gray-500 bg-gray-100 hover:text-white hover:bg-gray-500 rounded-full transition duration-150 ease-in-out"
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
                          className="flex justify-center items-center text-gray-500 bg-gray-100 hover:text-white hover:bg-gray-500 rounded-full transition duration-150 ease-in-out"
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
                          className="flex justify-center items-center text-gray-500 bg-gray-100 hover:text-white hover:bg-gray-500 rounded-full transition duration-150 ease-in-out"
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
                    Still on the working
  {/*                   {
                    posts.slice(randomIntegers[0],randomIntegers[1]).map(post => {
                      return (
                        <PostItem key={post.id} {...post} />
                      )
                    })} */}
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
