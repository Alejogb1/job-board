'use client';

import { useCallback, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createUrl } from '@/lib/utils/createUrl';
import { DEFAULT_MIN_VERSION } from 'tls';
import { ReactSearchAutocomplete } from 'react-search-autocomplete'
import Select from "react-tailwindcss-select";


interface Props {
  searchHandler: (value:string) => void
  inputRef: string
}
const options = [
  { 
    label: "",
    options :[
      { value: "AI&engineer", label: "AI engineer" },
      { value: "AI&researcher", label: "AI researcher" },
      { value: "AI&scientist", label: "AI scientist" },
      { value: "AI&specialist", label: "AI specialist" },
      { value: "AI&consultant", label: "AI consultant" },
      { value: "AI&developer", label: "AI developer" },
    ]
  },
  { value: "data&engineer", label: "data engineer" },
  { value: "deep&learning", label: "deep learning engineer" },
  { value: "data&scientist", label: "data scientist" },
  { value: "machine learning research", label: "machine learning researcher" },
  { value: "analyst", label: "analyst" },
  { value: "product&manager", label: "product manager" },
  { value: "generative", label: "generative AI" },
  { value: "engineer", label: "engineering" },
  { value: "analyst", label: "analyst" },
  { value: "architecture", label: "architecture" },
  { value: "vision", label: "computer vision" },
  { value: "nlp", label: "NLP" },
  { value: "security", label: "cybersecurity" },
  { value: "operations", label: "MLOps" },
  { value: "visualization", label: "visualization" },

];


export default function SearchField({ inputRef, onSearch }: any){
    const router = useRouter();
    const searchParams = useSearchParams();
    const [item, setItem] = useState(null);

    const onSubmit = (e:any) => {
      e.preventDefault();
      const newParams = new URLSearchParams(searchParams.toString());

      if (item) {
        const val = (item as { value: string }).value;
        const search = val;
        newParams.set('query', search);
      } else {
        newParams.delete('query');
      }
  
      router.push(createUrl('/', newParams));
    }

    const handleChange = (value:any) => {
      setItem(value);
    };
    // const onEnterHandler = useCallback((event) => {
    //   const isEnterPressed = event.which === ENTER_KEY
    //     || event.keyCode === ENTER_KEY;
    //   if (isEnterPressed && TypeChecker.isFunction(onEnter)) {
    //     onEnter(event.target.value, event);
    //   }
    // }, [onEnter]);

    useEffect(() => {
      const keyDownHandler = (event:any) => {
      if (event.key == 'Enter') {
        event.preventDefault();
        const newParams = new URLSearchParams(searchParams.toString());

        if (item) {
        const val = (item as { value: string }).value;
        const search = val;
        newParams.set('query', search);
        } else {
        newParams.delete('query');
        }
      
        router.push(createUrl('/', newParams));
      }
      };

      document.addEventListener('keydown', keyDownHandler);

      return () => {
      document.removeEventListener('keydown', keyDownHandler);
      };
    }, [item]);
  
  
  return(
        <div className="text-gray-200 ">
            <div className="styles-filter lg:p-0">
            <div className="search-bar">
                <div className="styles">
                <form onSubmit={onSubmit} className="flex gap-2">
                    <div className="w-full lg:w-6/12">
                        <div className="relative block w-full before:absolute before:inset-px before:rounded-[calc(theme(borderRadius.lg)-1px)] before:bg-white before:shadow after:pointer-events-none after:absolute after:inset-0 after:rounded-lg after:ring-inset after:ring-transparent sm:after:focus-within:ring-2 sm:after:focus-within:ring-blue-500 has-[[data-disabled]]:opacity-50 before:has-[[data-disabled]]:bg-zinc-950/5 before:has-[[data-disabled]]:shadow-none before:has-[[data-invalid]]:shadow-red-500/10">
                                {/* <input  
                                  defaultValue={searchParams?.get('query') || ''}
                                  key={searchParams?.get('query')}
                                  name="search"
                                  type="text" 
                                  placeholder="Search for job title or keyword" 
                                  className={`relative block w-full appearance-none rounded-lg px-[calc(theme(spacing[3.5])-1px)] py-[calc(theme(spacing[2.5])-1px)] sm:px-[calc(theme(spacing[3])-1px)] sm:py-[calc(theme(spacing[1.5])-1px)] text-base/6 text-zinc-950 placeholder:text-zinc-500 sm:text-sm/6  border border-zinc-950/10 data-[hover]:border-zinc-950/20 bg-transparent focus:outline-none data-[invalid]:border-red-500 data-[invalid]:data-[hover]:border-red-500 data-[disabled]:border-zinc-950/20`}                            
                                />           */}            
                                  <Select
                                    value={item}
                                    onChange={handleChange}
                                    options={options} 
                                    isSearchable={true}
                                    primaryColor={''}                          
                                  />

                        </div>
                        <div className="">
                        </div>
                    </div>
                    <div className="md:mt">
                      <button 
                        className="relative isolate inline-flex items-center justify-center gap-x-2 rounded-lg border text-base/6 font-semibold px-[calc(theme(spacing[3.5])-1px)] py-[calc(theme(spacing[2.5])-1px)] sm:px-[calc(theme(spacing.3)-1px)] sm:py-[calc(theme(spacing[1.5])-1px)] sm:text-sm/6 focus:outline-none data-[focus]:outline data-[focus]:outline-2 data-[focus]:outline-offset-2 data-[focus]:outline-blue-500 data-[disabled]:opacity-50 [&amp;>[data-slot=icon]]:-mx-0.5 [&amp;>[data-slot=icon]]:my-0.5 [&amp;>[data-slot=icon]]:size-5 [&amp;>[data-slot=icon]]:shrink-0 [&amp;>[data-slot=icon]]:text-[--btn-icon] [&amp;>[data-slot=icon]]:sm:my-1 [&amp;>[data-slot=icon]]:sm:size-4 forced-colors:[--btn-icon:ButtonText] forced-colors:data-[hover]:[--btn-icon:ButtonText] border-transparent bg-[--btn-border] before:absolute before:inset-0 before:-z-10 before:rounded-[calc(theme(borderRadius.lg)-1px)] before:bg-[--btn-bg] before:shadow  after:absolute after:inset-0 after:-z-10 after:rounded-[calc(theme(borderRadius.lg)-1px)] after:shadow-[shadow:inset_0_1px_theme(colors.white/15%)] after:data-[active]:bg-[--btn-hover-overlay] after:data-[hover]:bg-[--btn-hover-overlay]  before:data-[disabled]:shadow-none after:data-[disabled]:shadow-none text-white [--btn-bg:theme(colors.zinc.900)] [--btn-border:theme(colors.zinc.950/90%)] [--btn-hover-overlay:theme(colors.white/10%)]  [--btn-icon:theme(colors.zinc.400)] data-[active]:[--btn-icon:theme(colors.zinc.300)] data-[hover]:[--btn-icon:theme(colors.zinc.300)] cursor-pointer isolate inline-flex items-center justify-center gap-x-2 rounded-lg border text-base/6 font-semibold px-[calc(theme(spacing[3.5])-1px)] py-[calc(theme(spacing[2.5])-1px)] sm:px-[calc(theme(spacing.3)-1px)] sm:py-[calc(theme(spacing[1.5])-1px)] sm:text-sm/6 focus:outline-none data-[focus]:outline data-[focus]:outline-2 data-[focus]:outline-offset-2 data-[focus]:outline-blue-500 data-[disabled]:opacity-50 [&amp;>[data-slot=icon]]:-mx-0.5 [&amp;>[data-slot=icon]]:my-0.5 [&amp;>[data-slot=icon]]:size-5 [&amp;>[data-slot=icon]]:shrink-0 [&amp;>[data-slot=icon]]:text-[--btn-icon] [&amp;>[data-slot=icon]]:sm:my-1 [&amp;>[data-slot=icon]]:sm:size-4 forced-colors:[--btn-icon:ButtonText] forced-colors:data-[hover]:[--btn-icon:ButtonText] border-transparent bg-[--btn-border] before:absolute before:inset-0 before:-z-10 before:rounded-[calc(theme(borderRadius.lg)-1px)] before:bg-[--btn-bg] before:shadow  after:absolute after:inset-0 after:-z-10 after:rounded-[calc(theme(borderRadius.lg)-1px)] after:shadow-[shadow:inset_0_1px_theme(colors.white/15%)] after:data-[active]:bg-[--btn-hover-overlay] after:data-[hover]:bg-[--btn-hover-overlay]  before:data-[disabled]:shadow-none after:data-[disabled]:shadow-none text-white [--btn-bg:theme(colors.zinc.900)] [--btn-border:theme(colors.zinc.950/90%)] [--btn-hover-overlay:theme(colors.white/10%)]  [--btn-icon:theme(colors.zinc.400)] data-[active]:[--btn-icon:theme(colors.zinc.300)] data-[hover]:[--btn-icon:theme(colors.zinc.300)]" 
                      >
                        Search
                      </button>
                    </div>
                </form>
                <div className="mt-2"></div>
                </div>
            </div>
        </div>
      </div>
    )
}
