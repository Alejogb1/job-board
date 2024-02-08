"use client"
import {useState} from "react"
export default function SearchBar(){
    
    const [isActiveRole, setIsActiveRole] = useState(false);
    const [isActiveLocation, setIsActiveLocation] = useState(false);

    const handleRoleButtonClick = (event:any) => {
        event.preventDefault();
        setIsActiveRole(true);
    };
    const handleLocationButtonClick = (event:any) => {
        event.preventDefault();
        setIsActiveLocation(true);
    };
    return(
        <form className="styles-form-component sticky top-0 z-0 bg-white w-full lg:pb-2 border-gray-200 lg:mb-4">
            <div className="styles-filter lg:p-0">
            <div className="search-bar">
                <div className="styles">
                <div className="gap-2 flex flex-col lg:flex-row">
                    <div className="role-wrapper grow lg:basis-6/12 max-w-full">
                        <div className="rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-10 w-full flex flex-center">
                                <input type="text" placeholder="Search for job title or keyword" className={`text-md rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            />                        
                            </div>
                    </div>
                    <div className="location-wrapper grow basis-6/12 max-w-full">
                        <div className="rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-10 cursor-pointer w-full" >
                            <input type="text" placeholder="City or region" className={`text-md rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            />
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