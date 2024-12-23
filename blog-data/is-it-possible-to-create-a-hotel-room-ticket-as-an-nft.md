---
title: "Is it possible to create a hotel room ticket as an NFT?"
date: "2024-12-23"
id: "is-it-possible-to-create-a-hotel-room-ticket-as-an-nft"
---

Alright, let's dive into this. Thinking back to a project I tackled a few years ago involving a loyalty program for a small hotel chain, the idea of tokenizing hotel bookings as nfts is something I’ve considered extensively. The short answer is: yes, absolutely it's possible to create a hotel room ticket as an nft, and frankly, it opens up a lot of interesting possibilities.

The core concept revolves around representing a digital asset – in this case, a hotel booking – as a non-fungible token on a blockchain. Unlike fungible tokens like cryptocurrencies where one token is interchangeable with another, an nft is unique. This makes it ideal for representing something with specific, non-replicable attributes, such as a confirmed reservation for a specific room on particular dates.

Let’s examine how this could actually work from a technical standpoint. I remember when we initially explored this, the primary concern was ensuring the integrity of the booking details. One doesn’t want to issue nfts that can be easily altered or counterfeited. We decided to use a hash of the reservation details, combined with the hotel’s private key, to create the nft's unique identifier. This meant each nft was intrinsically tied to the reservation data and could be independently verified against the hotel’s records using their corresponding public key.

The data encoded within the nft would typically include: the hotel name, room type, check-in and check-out dates, guest names, and any associated booking number. All this data is encapsulated into a metadata json object associated with the token. This object is usually stored on a decentralized storage solution, like IPFS, with the nft pointing to its content identifier (CID). The token itself resides on a compatible blockchain, like Ethereum or Polygon, enabling transfer and verification.

Here's an example of how that process could look, using some hypothetical pseudocode to clarify:

```python
import hashlib
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

def create_hotel_nft_data(hotel_name, room_type, check_in, check_out, guest_names, booking_number, private_key_pem):
    reservation_data = {
        'hotel_name': hotel_name,
        'room_type': room_type,
        'check_in': check_in,
        'check_out': check_out,
        'guest_names': guest_names,
        'booking_number': booking_number
    }

    reservation_json = json.dumps(reservation_data, sort_keys=True).encode('utf-8') # Sort keys for consistency

    # Create a hash of the reservation data for immutability and uniqueness
    digest = hashes.Hash(hashes.SHA256())
    digest.update(reservation_json)
    hashed_data = digest.finalize()

    # Load the private key
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)

    # Sign the hashed data
    signature = private_key.sign(hashed_data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())

    # Pack data and signature for NFT creation
    nft_data = {
        'reservation_data': reservation_data,
        'hash': hashed_data.hex(),
        'signature': signature.hex()
    }

    return nft_data

# Example usage (using a placeholder for private key, you'd need your actual key)
private_key_pem = b"""
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC0eYn69yP7...
-----END PRIVATE KEY-----
"""
nft_data = create_hotel_nft_data("Grand Central Hotel", "Deluxe King", "2024-01-15", "2024-01-18", ["Alice", "Bob"], "GC12345", private_key_pem)
print(nft_data)
```
This python code snippet demonstrates the process of creating the nft data by hashing reservation details and signing it with a hotel's private key. In a real implementation, this would be paired with a smart contract for nft minting and a decentralized storage service to handle the associated metadata. This signature allows verification of the nft’s legitimacy without reliance on a central authority once the hotel’s public key is publicly available.

Now, let's look at a simplified representation of how a smart contract might function for minting these tokens. Bear in mind this is a very stripped-down example and would need further complexity for a live system:

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Strings.sol";


contract HotelNFT is ERC721 {

    string public baseURI;


    struct Reservation {
      string hotelName;
      string roomType;
      string checkIn;
      string checkOut;
      string[] guestNames;
      string bookingNumber;
      bytes signature;
    }

    mapping(uint256 => Reservation) public reservations;


    uint256 public nextTokenId = 1;


    constructor(string memory _baseURI) ERC721("HotelNFT", "HNFT"){
         baseURI = _baseURI;
    }


    function _baseURI() internal view virtual override returns (string memory) {
        return baseURI;
    }

  function mintNFT(
    string memory _hotelName,
    string memory _roomType,
    string memory _checkIn,
    string memory _checkOut,
    string[] memory _guestNames,
    string memory _bookingNumber,
    bytes memory _signature
) public  returns (uint256) {
        uint256 tokenId = nextTokenId;
        _safeMint(msg.sender, tokenId);

         reservations[tokenId] = Reservation({
           hotelName: _hotelName,
           roomType: _roomType,
           checkIn: _checkIn,
           checkOut: _checkOut,
           guestNames: _guestNames,
           bookingNumber: _bookingNumber,
           signature: _signature
         });
        nextTokenId++;
         return tokenId;
    }


    function verifySignature(uint256 _tokenId, bytes memory _hashed_data, address _hotelPublicAddress) public view returns (bool){
        bytes32 messageHash = keccak256(_hashed_data);
        bytes memory signature = reservations[_tokenId].signature;
        address signer = ecrecover(messageHash, uint8(signature[65]), bytes32(signature[0:32]), bytes32(signature[32:64]));
        return signer == _hotelPublicAddress;
    }
}

```
This Solidity code illustrates a simplified smart contract. It demonstrates the minting process for the nft, taking in the reservation data and a digital signature of its hash. It also provides a mechanism for verifying the validity of an nft against the hotel's public address. Again, this is a barebones example and a production-ready contract would need proper access control, error handling, and integration with a reliable oracle for data storage and retrieval.

Finally, let's explore how we could fetch the metadata from a decentralized storage service, like ipfs. Here is an abridged example using JavaScript:

```javascript
async function fetchMetadata(ipfsCid){

    try {
      const response = await fetch(`https://ipfs.io/ipfs/${ipfsCid}`);

        if(!response.ok)
        {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const metadata = await response.json();
        console.log("Metadata Fetched:",metadata)
        return metadata;
    }
    catch (error){
        console.error("Error fetching metadata: ",error);
        return null;
    }
}

// Example usage (replace with a real IPFS CID):
const cid = "QmUfL6W5j9624gqP81w6x87v9y253u02j9d6w971u82k9x";
fetchMetadata(cid);

```
This Javascript code provides an example of how to fetch metadata from a given ipfs cid, which would contain our reservation details, enabling off-chain verification and display of the booking information.

For those interested in further research, I'd suggest exploring the concepts surrounding ERC-721 and ERC-1155 standards for nfts, delving into the cryptographic details behind digital signatures (check out "Applied Cryptography" by Bruce Schneier).  Also, look at resources explaining distributed storage solutions like IPFS and Filecoin. In terms of smart contract development for Ethereum,  OpenZeppelin's documentation is excellent. Understanding these fundamentals will clarify the path to implementing hotel ticket nfts successfully.

To summarize: Yes, it's entirely feasible to create hotel room tickets as nfts. It offers enhanced security, improved transparency, and the potential for novel booking and reselling mechanisms. However, practical implementation necessitates a robust infrastructure and a deep understanding of both blockchain technology and its associated security implications. It's an exciting area, but definitely requires thoughtful design and meticulous development.
