import Link from 'next/link'
import getCompany from '@/lib/getCompany'
import extractDomain from '@/lib/extractDomain'

export default async function PostItem({ ...props }) {
  const companyData: Promise<any> = getCompany(props.company_code)
  const company: any = await companyData

  // Log company to inspect its structure
  console.log('company:', company)

  let salary_index = 0
  let salaryTag = ''
  if (props.min_salary > 0) {
    salaryTag = `$${props.min_salary / 1000}K - $${props.max_salary / 1000}K`
    salary_index = 1
  }

  return (
    <div className={`[&:nth-child(-n+12)]:-order-1 group border-b border-gray-200`}>
      <div className={`px-4 py-6`}>
        <div className="sm:flex items-center space-y-3 sm:space-y-0 sm:space-x-5">
          <div className="shrink-0 ">
            <img
              className='border border-gray-200'
              src={`https://logo.clearbit.com/${extractDomain(company?.company_url || '')}`}
              width="56"
              height="56"
              alt="Company Logo"
            />
          </div>
          <div className="grow lg:flex items-center justify-between space-y-5 lg:space-x-2 lg:space-y-0">
            <div>
              <div className="flex data-start space-x-2">
                <div className="text-sm text-gray-800 font-semibold mb-1">
                  {company?.company_name || 'Company Name'}
                </div>
              </div>
              <div className="mb-2">
                <Link className="text-lg text-gray-800 font-bold" href={`/posts/${props.slug}`}>
                  {props.job_title}
                </Link>
              </div>
              <div className="-m-1">
                {salaryTag && (
                  <a
                    className={`text-xs text-gray-500 font-medium inline-flex px-2 py-0.5 hover:text-gray-600 m-1 whitespace-nowrap transition duration-150 ease-in-out bg-gray-50`}
                    href="#0"
                  >
                    {salaryTag}
                  </a>
                )}
                <a
                  className="inline-flex items-center gap-x-1.5 rounded-md px-1.5 py-0.5 text-sm font-medium sm:text-xs bg-lime-500/20 text-lime-700"
                >
                  Well-paid
                </a>
              </div>
            </div>
            <div className="min-w-[120px] flex items-center lg:justify-end space-x-3 lg:space-x-0">
              <div className="lg:hidden group-hover:lg:block">
                <Link
                  target="_blank"
                  className="btn-sm py-1.5 px-3 text-white bg-blue-500 hover:bg-blue-600 group shadow-sm"
                  href={`${props.job_post_url}`}
                >
                  Apply Now
                </Link>
              </div>
              <div className="group-hover:lg:hidden text-sm italic text-gray-500">
                {props.date}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
