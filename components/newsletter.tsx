'use client'

import React, { useRef, useState } from 'react';


export default function Newsletter() {
    // 1. Create a reference to the input so we can fetch/clear it's value.
    const inputEl:any = useRef(null);
    // 2. Hold a message in state to handle the response from our API.
    const [message, setMessage] = useState('');
  
    const subscribe = async (e:any) => {
      e.preventDefault();
  
      // 3. Send a request to our API with the user's email address.
      const res = await fetch('/api/subscribe', {
        body: JSON.stringify({
          email: inputEl.current.value,
        }),
        headers: {
          'Content-Type': 'application/json',
        },
        method: 'POST',
      });

  
      // 5. Clear the input value and show a success message.
      inputEl.current.value = '';
      setMessage('Success! You are now subscribed to the newsletter.');
    }
  return (
    <div className="relative text-center px-4 py-6 group ">
      <div
        className="absolute inset-0 rounded-xl bg-gray-50 border border-gray-200 -rotate-1 -z-10 group-hover:rotate-0 transition duration-150 ease-in-out "
        aria-hidden="true"
      />
      <div className="font-nycd text-xl text-gray-500 mb-1">Land your dream job</div>
      <div className="text-2xl font-bold mb-5">Get a weekly email with the latest AI jobs.</div>
      <form onSubmit={subscribe} className="inline-flex max-w-sm ">
              {message ? <div className='text-sm'>{message}</div> : 
                <div className="flex flex-col sm:flex-row justify-center max-w-xs mx-auto sm:max-w-none">
                  <input  
                  className="form-input py-1.5 w-full mb-2 sm:mb-0 sm:mr-2" 
                  id="email-input"
                  name="email"
                  placeholder="you@awesome.com"
                  ref={inputEl}
                  required
                  type="email"
                  />
                    <button className="btn-sm text-white bg-black hover:bg-gray-700 shadow-sm whitespace-nowrap" type="submit">
                      Join Newsletter
                    </button>
                </div>
              }
      </form>
    </div>
  )
}