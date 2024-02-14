"use client"
import { useSearchParams, usePathname, useRouter } from 'next/navigation';
import { useDebouncedCallback } from 'use-debounce';

export default function SearchBar(){
    const searchParams = useSearchParams();
    const pathname = usePathname();
    const { replace } = useRouter();

    const handleTermSearch = useDebouncedCallback((term) => {

        const params = new URLSearchParams(searchParams);
        params.set('page', '1');
        if (term) {
          params.set('query', term);
        } else {
          params.delete('query');
        }
        replace(`${pathname}?${params.toString()}`);
      }, 300);
    const handleLocationSearch = useDebouncedCallback((term) => {     
      const params = new URLSearchParams(searchParams);
      params.set('page', '1');

      if (term) {
        params.set('loc', term);
      } else {
        params.delete('loc');
      }
      replace(`${pathname}?${params.toString()}`);
    }, 300);
    
    return(
        <form className="styles-form-component sticky pt-4 top-0 z-10 bg-white w-full lg:pb-2 border-gray-200 lg:mb-4">
            <div className="styles-filter lg:p-0">
            <div className="search-bar">
                <div className="styles">
                <div className="gap-2 flex flex-col lg:flex-row">
                    <div className="role-wrapper grow lg:basis-6/12 max-w-full">
                        <div className="rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-10 w-full flex flex-center">
                                <input   
                                  defaultValue={searchParams.get('query')?.toString()}       
                                  onChange={(e) => {handleTermSearch(e.target.value);}}
                                  type="text" placeholder="Search for job title or keyword" className={`text-md rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            
                                />                        
                            </div>
                    </div>
                    <div className="location-wrapper grow basis-6/12 max-w-full">
                        <div className="rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-10 cursor-pointer w-full" >
                            <input 
                              defaultValue={searchParams.get('loc')?.toString()}       
                              onChange={(e) => {handleLocationSearch(e.target.value);}}
                              type="text" 
                              placeholder="City or region" 
                              className={`text-md rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            
                            />
                        </div>
                    </div>
                </div>
                <div className="mt-2"></div>
                </div>
            </div>
        </div>
      </form>
    )
}