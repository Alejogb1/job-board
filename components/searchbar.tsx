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
        <form className="styles-form-component sticky top-0 z-0 bg-white w-full lg:pb-2 lg:pt-4 border border-gray-200 lg:px-4 lg:mb-4">
            <div className="styles-filter lg:p-0">
            <div className="search-bar">
                <div className="styles px-4 py-4">
                <div className="gap-2 flex flex-col lg:flex-row">
                    <div className="role-wrapper grow lg:basis-6/12 max-w-full">
                        <button onClick={handleRoleButtonClick} className="focus-button rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-12 cursor-pointer w-full" type='button'>
                        {isActiveRole && (
                            <input type="text" placeholder="Search.." className={`text-lg rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none  ${isActiveRole ? '' : 'hidden'}  bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            />
                        )}
                        <div className="flex flex-row items-center pl-2 pr-4">
                            <svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className={`haloIcon w-4 max-w-full ${isActiveRole ? 'hidden' : ''}`}><path fill-rule="evenodd" clip-rule="evenodd" d="M7 16h10a5 5 0 015 5 1 1 0 01-1.993.117l-.012-.293a3 3 0 00-2.819-2.819L17 18H7a3 3 0 00-3 3 1 1 0 11-2 0 5 5 0 014.783-4.995L7 16h10H7zm5-14a6 6 0 110 12 6 6 0 010-12zm0 2a4 4 0 100 8 4 4 0 000-8z" fill="currentColor"></path></svg>
                            <span className={`styles-label text-lg pl-4 whitespace-nowrap overflow-hidden overflow-ellipsis ${isActiveRole ? 'hidden' : ''}`}>Job title</span>
                        </div>
                        </button>
                  
                    </div>
                    <div className="location-wrapper grow basis-6/12 max-w-full">
                    <button onClick={handleLocationButtonClick} className="focus-button rounded-sm border border-gray-300 bg-slate-50 text-gray-400 text-base h-12 cursor-pointer w-full" type='button'>
                            {isActiveLocation && (
                                <input type="text" placeholder="Search.." className={`text-lg rounded-sm whitespace-nowrap boverflow-hidden outline-none border-0 active:outline-none focus:outline-none  ${isActiveLocation ? '' : 'hidden'}  bg-slate-50 text-gray-800 text-base h-full cursor-text w-full`}                            />
                            )}
                        <div className="flex flex-row items-center pl-2 pr-4">
                            <svg className={`shrink-0 ${isActiveLocation ? 'hidden' : ''}`} height="1em" version="1.1" viewBox="0 0 16 16" width="1em"><title></title><g fill="none" fill-rule="evenodd" id="Symbols" stroke="none" stroke-width="1"><g fill="currentColor" id="icn_office"><path d="M13.6666667,3.00000121 C13.8507616,3.00000121 14,3.15316695 14,3.34210647 L14,3.34210647 L14,14.9736854 C14,15.540504 13.5522847,16.0000012 13,16.0000012 L13,16.0000012 L9.66666667,16.0000012 C9.48257175,16.0000012 9.33333333,15.8468355 9.33333333,15.6578959 L9.33333333,15.6578959 L9.33333333,14.2894749 C9.33333333,13.5337168 8.73637967,12.9210538 8,12.9210538 C7.26362033,12.9210538 6.66666667,13.5337168 6.66666667,14.2894749 L6.66666667,14.2894749 L6.66666667,15.6578959 C6.66666667,15.8468355 6.51742825,16.0000012 6.33333333,16.0000012 L6.33333333,16.0000012 L3,16.0000012 C2.44771525,16.0000012 2,15.540504 2,14.9736854 L2,14.9736854 L2,3.34210647 C2,3.15316695 2.14923842,3.00000121 2.33333333,3.00000121 L2.33333333,3.00000121 Z M4.33333333,8.47368542 C3.78104858,8.47368542 3.33333333,8.93318265 3.33333333,9.50000121 L3.33333333,9.50000121 L3.33333333,10.8684223 C3.33333333,11.0573618 3.48257175,11.2105275 3.66666667,11.2105275 L3.66666667,11.2105275 L5,11.2105275 C5.18409492,11.2105275 5.33333333,11.0573618 5.33333333,10.8684223 L5.33333333,10.8684223 L5.33333333,9.50000121 C5.33333333,8.93318265 4.88561808,8.47368542 4.33333333,8.47368542 Z M8,8.47368542 C7.44771525,8.47368542 7,8.93318265 7,9.50000121 L7,9.50000121 L7,10.8684223 C7,11.0573618 7.14923842,11.2105275 7.33333333,11.2105275 L7.33333333,11.2105275 L8.66666667,11.2105275 C8.85076158,11.2105275 9,11.0573618 9,10.8684223 L9,10.8684223 L9,9.50000121 C9,8.93318265 8.55228475,8.47368542 8,8.47368542 Z M11.6666667,8.47368542 C11.1143819,8.47368542 10.6666667,8.93318265 10.6666667,9.50000121 L10.6666667,9.50000121 L10.6666667,10.8684223 C10.6666667,11.0573618 10.8159051,11.2105275 11,11.2105275 L11,11.2105275 L12.3333333,11.2105275 C12.5174282,11.2105275 12.6666667,11.0573618 12.6666667,10.8684223 L12.6666667,10.8684223 L12.6666667,9.50000121 C12.6666667,8.93318265 12.2189514,8.47368542 11.6666667,8.47368542 Z M4.33333333,4.36842226 C3.78104858,4.36842226 3.33333333,4.82791949 3.33333333,5.39473805 L3.33333333,5.39473805 L3.33333333,6.7631591 C3.33333333,6.95209862 3.48257175,7.10526436 3.66666667,7.10526436 L3.66666667,7.10526436 L5,7.10526436 C5.18409492,7.10526436 5.33333333,6.95209862 5.33333333,6.7631591 L5.33333333,6.7631591 L5.33333333,5.39473805 C5.33333333,4.82791949 4.88561808,4.36842226 4.33333333,4.36842226 Z M8,4.36842226 C7.44771525,4.36842226 7,4.82791949 7,5.39473805 L7,5.39473805 L7,6.7631591 C7,6.95209862 7.14923842,7.10526436 7.33333333,7.10526436 L7.33333333,7.10526436 L8.66666667,7.10526436 C8.85076158,7.10526436 9,6.95209862 9,6.7631591 L9,6.7631591 L9,5.39473805 C9,4.82791949 8.55228475,4.36842226 8,4.36842226 Z M11.6666667,4.36842226 C11.1143819,4.36842226 10.6666667,4.82791949 10.6666667,5.39473805 L10.6666667,5.39473805 L10.6666667,6.7631591 C10.6666667,6.95209862 10.8159051,7.10526436 11,7.10526436 L11,7.10526436 L12.3333333,7.10526436 C12.5174282,7.10526436 12.6666667,6.95209862 12.6666667,6.7631591 L12.6666667,6.7631591 L12.6666667,5.39473805 C12.6666667,4.82791949 12.2189514,4.36842226 11.6666667,4.36842226 Z M12.0130926,1.20520679e-06 C12.3167616,-0.000435902675 12.6040299,0.118046322 12.7931105,0.3217165 L12.7931105,0.3217165 L13.9271366,1.53600145 C14.007083,1.62181261 14.0225757,1.73930323 13.9669914,1.83824387 C13.911407,1.93718452 13.7945831,2.00006439 13.666464,2.00000121 L13.666464,2.00000121 L2.33286959,2.00000121 C2.20487083,1.99984498 2.08828097,1.93688193 2.03286416,1.83798658 C1.97744735,1.73909122 1.99299217,1.621732 2.0728636,1.53600145 L2.0728636,1.53600145 L3.20622304,0.3217165 C3.39530366,0.118046322 3.68257198,-0.000435902675 3.98624101,1.20520679e-06 L3.98624101,1.20520679e-06 Z" id="Combined-Shape"></path></g></g></svg>                              
                            <span className={`text-lg styles-label pl-4 whitespace-nowrap overflow-hidden overflow-ellipsis ${isActiveLocation ? 'hidden' : ''}`}>Location</span>
                        </div>
                        </button>
                    </div>
                </div>
                <div className="mt-2"></div>
                </div>
            </div>
        </div>
      </form>
    )
}