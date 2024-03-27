import Link from 'next/link'
import getCompany from '@/lib/getCompany'
import extractDomain from '@/lib/extractDomain'
import getTag from '@/lib/getTag'
import getTagsOnPosts from '@/lib/getTags'

export default async function PostItem({...props }) {
    const companyData: Promise<any> = getCompany(props.company_code)
    const company:any = await companyData

    const tagsOnPostData: Promise<any> = getTagsOnPosts(props.id)
    const tags:any = await tagsOnPostData
    const one_tagData: Promise<any> = getTag(tags[6]?.tagId) 
    const tag:any = await one_tagData

    const second_tagData: Promise<any> = getTag(tags[4]?.tagId) 
    const second_tag:any = await second_tagData

    let salary_index = 0
    let salaryTag = ''
    if (props.min_salary > 0) {
      salaryTag = `$${props.min_salary}K - $${props.max_salary}K`
      salary_index = 1
    } 
    if(salary_index > 0 ) {
      return (
        <div className={`[&:nth-child(-n+12)]:-order-1 group border-b border-gray-200`}>
          <div className={`px-4 py-6`}>
            <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
              <div className="shrink-0 ">
                <img className=' border border-gray-200' src={`https://logo.clearbit.com/${extractDomain(company.company.company_webiste_url)}`} width="56" height="56" alt="https://logo.clearbit.com/clearbit.com" />
              </div>
              <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0">
                <div>
                  <div className="flex datas-start space-x-2">
                    <div className="text-sm text-gray-800 font-semibold mb-1">{company.company.company_name}</div>
                  </div>
                  <div className="mb-2">
                    <Link className="text-lg text-gray-800 font-bold" href={`/posts/${props.slug}`}>
                      {props.job_title}
                    </Link>
                  </div>
                  <div className="-m-1">
                    <a
                      className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600  m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-50`}
                      href="#0"
                    >
                      {salaryTag}
                    </a>
                    <a
                      className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600  m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-50`}
                      href="#0"
                    >
                     {tag[0].title}
                    </a>
                    <a className="inline-flex items-center gap-x-1.5 rounded-md px-1.5 py-0.5 text-sm/5 font-medium sm:text-xs/5 forced-colors:outline bg-lime-500/20 text-lime-700 group-data-[hover]:bg-lime-500">Well-paid</a>
                  </div>
                </div>
                <div className="min-w-[120px] flex items-center lg:justify-end space-x-3 lg:space-x-0">
                  <div className="lg:hidden group-hover:lg:block">
                    <Link className="btn-sm py-1.5 px-3 text-white bg-blue-500 hover:bg-blue-600 group shadow-sm" href={`/posts/${props.slug}`}>
                      Apply Now
                    </Link>
                  </div>
                  <div className="group-hover:lg:hidden text-sm italic text-gray-500">{props.date}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    } else {
      return (
        <div className={`[&:nth-child(-n+12)]:-order-1 group border-b border-gray-200`}>
          <div className={`px-4 py-6`}>
            <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
              <div className="shrink-0">
                <img src={`https://logo.clearbit.com/${extractDomain(company.company.company_webiste_url)}`} width="56" height="56" alt="https://logo.clearbit.com/clearbit.com" />
              </div>
              <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0">
                <div>
                  <div className="flex datas-start space-x-2">
                    <div className="text-sm text-gray-800 font-semibold mb-1">{company.company.company_name}</div>
                  </div>
                  <div className="mb-2">
                    <Link className="text-lg text-gray-800 font-bold" href={`/posts/${props.slug}`}>
                      {props.job_title}
                    </Link>
                  </div>
                  <div className="-m-1">
                    <a
                      className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600  m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-50`}
                      href="#0"
                    >
                      {tag[0].title}
                    </a>
                    <a
                      className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600  m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-50`}
                      href="#0"
                    >
                     {second_tag[0].title}
                    </a>
                    <a className="inline-flex items-center gap-x-1.5 rounded-md px-1.5 py-0.5 text-sm/5 font-medium sm:text-xs/5 forced-colors:outline bg-purple-500/20 text-purple-700 group-data-[hover]:bg-purple-500">Urgent</a>
                  </div>
                </div>
                <div className="min-w-[120px] flex items-center lg:justify-end space-x-3 lg:space-x-0">
                  <div className="lg:hidden group-hover:lg:block">
                    <Link className="btn-sm py-1.5 px-3 text-white bg-blue-500 hover:bg-blue-600 group shadow-sm" href={`/posts/${props.slug}`}>
                      Apply Now
                    </Link>
                  </div>
                  <div className="group-hover:lg:hidden text-sm italic text-gray-500">{props.date}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    }
}
  

