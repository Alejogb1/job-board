'use client'
import { usePathname, useSearchParams } from 'next/navigation';
import { useRouter } from 'next/router';
import { useDebouncedCallback } from 'use-debounce';
import { generatePagination } from '@/lib/utils/generatePagination';

export default function Pagination({totalPages} : {totalPages:number}) {
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const currentPage = Number(searchParams.get('page')) || 1;

    const allPages = generatePagination(currentPage, totalPages);

    const createPageURL = (pageNumber: number | string) => {
        const params = new URLSearchParams(searchParams);
        params.set('page', pageNumber.toString());
        return `${pathname}?${params.toString()}`;
      };
    
    return (
       <nav aria-label="" className=' md:block lg:block'>
        <ul className="max-w-8/12 inline-flex -space-x-px text-xs mt-4">
          <li>
            <a href={createPageURL(currentPage - 1)}  aria-current="page" className="flex items-center justify-center px-3 h-8 ms-0 leading-tight text-white bg-black rounded-l-md border border-e-0 border-white hover:bg-gray-600">Previous</a>
          </li>
          {allPages.map((page, index) => {
          let position: 'first' | 'last' | 'single' | 'middle' | undefined;

          if (index === 0) position = 'first';
          if (index === allPages.length - 1) position = 'last';
          if (allPages.length === 1) position = 'single';
          if (page === '...') position = 'middle';

            return (
                <li key={index} className='hidden md:block'>
                    <a href={createPageURL(page)}className="flex items-center justify-center px-3 h-8 leading-tight text-white bg-black border border-white hover:bg-gray-600 ">{page}</a>
                </li>
            );
        })}
          <li>
            <a href={createPageURL(currentPage + 1)} className="flex items-center justify-center px-3 h-8 leading-tight text-white bg-black rounded-r-md border border-white  hover:bg-gray-600">Next</a>
          </li>
        </ul>
      </nav>
    )
}