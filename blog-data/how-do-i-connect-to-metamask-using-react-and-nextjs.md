---
title: "How do I connect to Metamask using React and Next.js?"
date: "2024-12-23"
id: "how-do-i-connect-to-metamask-using-react-and-nextjs"
---

Alright, let's talk about connecting to Metamask within a React and Next.js environment. This isn't a straightforward plug-and-play scenario, though the libraries available make it significantly easier than it once was. I remember having to deal with raw provider handling years ago – quite the headache, let me tell you. We've moved on to much more elegant solutions.

The crux of the issue lies in bridging the browser's web3 provider (injected by Metamask) with your React application, and in a Next.js environment, you're also dealing with server-side rendering quirks. So, the primary goal is to safely access the `window.ethereum` object, which Metamask injects, and then use it to interact with the user's wallet.

Essentially, you need to handle four main aspects:

1.  **Provider Detection:** Checking if Metamask (or a compatible provider) is actually present.
2.  **Account Connection:** Requesting the user to connect their wallet to your application.
3.  **Account Information Retrieval:** Obtaining the user's connected address(es) and chain id.
4.  **Transaction Sending (optional):**  If you plan to send transactions, you’ll need to use the provider for that, which involves different steps.

For all of this, the `ethers.js` library is almost indispensable, although `web3.js` is another viable option. In my experience, `ethers.js` tends to be slightly easier to work with and more modern, offering a cleaner API. I'd recommend starting with the official `ethers.js` documentation; it's very well written and comprehensive. Another fantastic resource is the "Mastering Ethereum" book by Andreas Antonopoulos and Gavin Wood, which provides fundamental knowledge about blockchain concepts and how they function, a good foundation if you need to understand the 'why' behind things, too. Also, check out the Ethereum developer documentation, it often contains useful info on interacting with Ethereum networks, and is often kept very updated.

Let's break down how you'd achieve this in code. I’ll be focusing on using `ethers.js` here, as it’s my preferred approach.

**Snippet 1: Detecting and connecting to the provider**

```jsx
import { useState, useEffect } from 'react';
import { ethers } from 'ethers';

const useMetamask = () => {
    const [provider, setProvider] = useState(null);
    const [signer, setSigner] = useState(null);
    const [address, setAddress] = useState(null);
    const [chainId, setChainId] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);


    useEffect(() => {
        const detectProvider = async () => {
            if (typeof window !== 'undefined' && window.ethereum) {
                try {
                    const newProvider = new ethers.BrowserProvider(window.ethereum);
                     setProvider(newProvider);
                     const currentSigner = await newProvider.getSigner();
                      setSigner(currentSigner);
                    const network = await newProvider.getNetwork();
                    setChainId(network.chainId);

                    // Check if already connected
                    const currentAccounts = await window.ethereum.request({ method: 'eth_accounts' });
                    if (currentAccounts.length > 0) {
                        setAddress(currentAccounts[0]);
                        setIsConnected(true);
                        
                    }

                     window.ethereum.on('accountsChanged', handleAccountsChanged)
                      window.ethereum.on('chainChanged', handleChainChanged)
                                    
                } catch (error) {
                    console.error("Error detecting provider:", error);
                }
            } else {
                console.log("Metamask is not installed!");
            }
        };
         
        detectProvider();


       return () =>{
          if (window.ethereum){
            window.ethereum.removeListener('accountsChanged', handleAccountsChanged)
            window.ethereum.removeListener('chainChanged', handleChainChanged)
          }
        }


    }, []);

   

    const handleAccountsChanged = async (accounts) => {
         if (accounts.length === 0) {
            setAddress(null);
            setIsConnected(false);
            return;
        }
        setAddress(accounts[0]);
        setIsConnected(true);
    };

    const handleChainChanged = async (newChainId) =>{
        setChainId(parseInt(newChainId))
         if (provider) {
            const currentSigner = await provider.getSigner();
                setSigner(currentSigner);
             }

    }
    
    const connectWallet = async () => {
        setIsLoading(true)
        if(provider){

           try{
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            if (accounts.length > 0) {
              setAddress(accounts[0]);
                setIsConnected(true);
            }

           }catch (error){
            console.error('Error connecting:', error)
            setIsConnected(false)
           }
        }
        setIsLoading(false)
    };
    const disconnectWallet = () =>{
        setAddress(null);
        setIsConnected(false);
    }

    return { provider, signer, address, chainId, isConnected, connectWallet, disconnectWallet, isLoading };
};

export default useMetamask;
```

This React hook (`useMetamask`) does a lot. Let me break it down:

*   **`useState`**: It uses React's state hooks to manage the provider, signer, address, chainId, connection status, and loading state.
*   **`useEffect`**: The heart of our connection is the `useEffect` hook which is responsible for detecting the provider when the component mounts. I've specifically handled the unmounting phase by removing the event listeners to prevent memory leaks.
*   **`detectProvider`**: Inside the effect, we check if `window.ethereum` exists before creating an ethers provider. If it exists, it also verifies if a user is already connected and sets the user's address, and if a user is not connected it initializes the event listeners. If it doesn’t exist, a message informs the user that Metamask is missing.
*   **`handleAccountsChanged`**: Handles account changes when the connected Metamask account is changed in the Metamask browser extension.
*    **`handleChainChanged`**: Handles network changes if the user changes the connected network in the browser extension.
*   **`connectWallet`**:  Triggers the Metamask popup to request connection, retrieves the first connected address if successful.
*    **`disconnectWallet`**: Sets the address to null and updates the connection status.
*   **Return**: It returns all the important state values and connection functions.

You can then use this hook in any of your React components to access the wallet information.

**Snippet 2: Using the hook in a component**

```jsx
import useMetamask from './hooks/useMetamask';

function WalletButton() {
    const { address, isConnected, connectWallet, disconnectWallet, isLoading } = useMetamask();

    return (
        <div>
           {isLoading ? <p>Loading...</p> : isConnected ? (
                <>
                    <p>Connected Account: {address.slice(0, 6)}...{address.slice(-4)}</p>
                     <button onClick={disconnectWallet}>Disconnect</button>
                </>
            ) : (
                <button onClick={connectWallet}>Connect Wallet</button>
            )}
        </div>
    );
}

export default WalletButton;

```

This component demonstrates how to utilize the `useMetamask` hook. It renders a button to connect the wallet and displays the user's address after connection. When connected, it displays a disconnect button as well.

**Snippet 3: Example transaction sending (Requires a deeper understanding of smart contracts)**

```jsx
import useMetamask from './hooks/useMetamask';
import {useState} from "react"
import {ethers} from 'ethers';


const SendEth = () => {
     const { signer, address, provider, isConnected } = useMetamask();
    const [toAddress, setToAddress] = useState('');
    const [ethAmount, setEthAmount] = useState('');
    const [txHash, setTxHash] = useState('')
    const [isSending, setIsSending] = useState(false)
    const [error, setError] = useState('');
     
    const handleSendEth = async () => {
        if (!signer) {
            setError('No signer available.');
            return;
        }
          if(!isConnected) {
            setError('Please connect your wallet')
             return;
          }

        if (!ethers.isAddress(toAddress)) {
              setError('Invalid recipient address.');
            return;
        }

        if(isNaN(parseFloat(ethAmount)) || parseFloat(ethAmount) <= 0){
              setError('Invalid ETH amount')
             return;
        }
         setError('')
        setIsSending(true)

      try {
        const tx = await signer.sendTransaction({
            to: toAddress,
            value: ethers.parseEther(ethAmount)
        });
         setTxHash(tx.hash)
         const receipt = await tx.wait()
         if (receipt.status === 1){
              
         } else{
             setError('Transaction failed')
         }
          
       } catch (error){
            console.error('Error sending transaction:', error);
            setError(error.message)

      } finally {
        setIsSending(false)
      }
    };

  return (
       <div>
            <h2>Send ETH</h2>
            {error && <p style={{ color: 'red' }}>{error}</p>}
             {txHash &&  <p>Transaction Successful <a href={`https://sepolia.etherscan.io/tx/${txHash}`}> View Transaction </a> </p>}

            <input type="text" placeholder='To Address' value={toAddress} onChange={(e) => setToAddress(e.target.value)} />
            <input type="text" placeholder='ETH Amount' value={ethAmount} onChange={(e) => setEthAmount(e.target.value)} />
           
            <button onClick={handleSendEth} disabled={isSending}>{isSending ? 'Sending' : 'Send'}</button>
        </div>
    )
}

export default SendEth;

```

This component illustrates how to send a simple ETH transaction using the signer from our hook. It validates the input, sends the transaction using `signer.sendTransaction`, and handles errors and displays the txHash once it's successful.

**A few words of caution.**

*   **Error Handling:** Never assume everything will work flawlessly. Implement robust error handling, especially around user inputs and network requests.
*   **Security:** Never store private keys on the client-side, and always double-check transaction details before signing.
*   **Server-Side Rendering:** If you are doing server-side rendering, you need to ensure the code accessing `window.ethereum` only runs client-side. That’s what the `typeof window !== 'undefined'` check is for.
*   **Network Specificity**: Ensure you're using the right network and inform the user about the connected network.
*   **Asynchronous operations**: Be careful to handle all your asynchronous operations with `async/await` and handle loading states to avoid unexpected UI behaviors.

This should give you a pretty solid foundation for connecting to Metamask in your React and Next.js applications. Always refer back to the official documentation and libraries, and always stay updated with the latest web3 developments, it’s a rapidly evolving landscape.
