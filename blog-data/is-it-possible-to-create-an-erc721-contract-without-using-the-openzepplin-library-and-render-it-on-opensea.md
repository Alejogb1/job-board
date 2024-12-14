---
title: "Is it possible to create an erc721 contract without using the openzepplin library and render it on opensea?"
date: "2024-12-14"
id: "is-it-possible-to-create-an-erc721-contract-without-using-the-openzepplin-library-and-render-it-on-opensea"
---

well, let's break this down. yeah, it is totally doable to craft an erc721 contract without relying on openzeppelin, and you can absolutely get it showing up on opensea. it's more work, certainly, but not rocket science. i've been in this space a good while, and i can tell you it's a pretty common learning experience for anyone getting serious with solidity. i remember, back in the early days, before openzeppelin's libraries became the de-facto standard, everyone was rolling their own contracts, and, let me tell you, it wasn't always pretty.

so, the heart of an erc721 contract, the minimal required stuff, is actually pretty concise. it all comes down to a few key functions and some storage variables. we need to manage token ownership, approvals, and handle token transfers. the core interface specification, defined in the erc721 standard, is what we need to adhere to. openzeppelin simply provides a nice, secure, and tested implementation, but we can replicate that functionality with our own code. it's more verbose, obviously, but it gives you much better understanding of what's happening behind the scenes.

i once spent a whole weekend debugging some custom erc721 contract, turned out i had a logic error in the transfer function that allowed someone to steal tokens. no, not me, thankfully, it was on the testnet. but lesson learned. since that day, I’ve always been cautious about rolling my own low-level implementation, you really need to be very careful. there are a lot of edge cases to consider, and that's where libraries like openzeppelin truly shine. but the exercise of building one from scratch, once, is invaluable to really understand the spec.

first things first, you’ll need variables to track ownership, token metadata, and approvals. a typical setup would look like this in solidity:

```solidity
pragma solidity ^0.8.0;

contract minimalERC721 {
    mapping(uint256 => address) public tokenOwners;
    mapping(uint256 => string) public tokenURIs;
    mapping(address => uint256) private _balances;
    mapping(uint256 => address) private _tokenApprovals;

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    string public name;
    string public symbol;
    uint256 private _currentTokenId;

    constructor(string memory _name, string memory _symbol) {
      name = _name;
      symbol = _symbol;
      _currentTokenId = 0;
    }

    function balanceOf(address owner) public view returns (uint256) {
        require(owner != address(0), "address zero is not a valid owner");
        return _balances[owner];
    }

    function ownerOf(uint256 tokenId) public view returns (address) {
        address owner = tokenOwners[tokenId];
        require(owner != address(0), "token does not exist");
        return owner;
    }

    function approve(address approved, uint256 tokenId) public payable {
      address owner = ownerOf(tokenId);
      require(msg.sender == owner, "not token owner");
      _tokenApprovals[tokenId] = approved;
      emit Approval(owner, approved, tokenId);
    }

    function getApproved(uint256 tokenId) public view returns (address) {
        return _tokenApprovals[tokenId];
    }

     function mint(address to, string memory tokenURI) public {
        _mint(to, _currentTokenId, tokenURI);
        _currentTokenId++;
     }
     
    function _mint(address to, uint256 tokenId, string memory tokenURI) private {
        require(tokenOwners[tokenId] == address(0), "token already minted");
        tokenOwners[tokenId] = to;
        tokenURIs[tokenId] = tokenURI;
        _balances[to] += 1;
        emit Transfer(address(0), to, tokenId);

    }


   function transferFrom(address from, address to, uint256 tokenId) public payable {
       require(_isApprovedOrOwner(msg.sender, tokenId), "caller is not token owner or approved address");
       _transfer(from, to, tokenId);
    }
    function _transfer(address from, address to, uint256 tokenId) private {
        require(from == tokenOwners[tokenId], "from is not the token owner");
        require(to != address(0), "to is zero address");

        _balances[from] -= 1;
        _balances[to] += 1;
        tokenOwners[tokenId] = to;
         delete _tokenApprovals[tokenId];
        emit Transfer(from, to, tokenId);
   }
    function _isApprovedOrOwner(address spender, uint256 tokenId) private view returns(bool) {
       address owner = ownerOf(tokenId);
       return (spender == owner || getApproved(tokenId) == spender);
    }

    function tokenURI(uint256 tokenId) public view returns (string memory) {
      require(tokenOwners[tokenId] != address(0), "token does not exist");
        return tokenURIs[tokenId];
    }


    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
      return interfaceId == 0x80ac58cd || // ERC721
              interfaceId == 0x5b5e139f; // ERC721Metadata
      }

}
```

this is the very basic structure. we have `tokenOwners` to track who owns which token, `tokenURIs` for the metadata, `balances` for the token counts, and `_tokenApprovals` to manage transfer approvals, along with `name` and `symbol` for the metadata. we also have the `transfer` and `approval` events that are very important, opensea relies on them to know when tokens are transferred. functions like `balanceOf`, `ownerOf`, `approve`, `getApproved` `mint`, `transferFrom` and `tokenURI` implement the required erc721 functionality, in the most minimal way.

note, that we also have the `supportsInterface` function that returns `true` if a specific interface is supported by our contract. that's extremely important because opensea uses that to verify if your contract conforms to the standard. we need it for both the core erc721 interface and the optional erc721metadata interface.

now, metadata. you see that the contract has a `tokenURIs` mapping? well, that maps a token id to a string. this string is expected to be a uri, usually an http link to a json document that contains all the metadata of the token, like name, description, and image url. usually people use ipfs to store these json metadata files because of its decentralized nature, but any http url will work. opensea will fetch this metadata when displaying your nft on its platform.

the format of this json file should follow the metadata standard defined in erc721, here is an example:

```json
{
  "name": "My first custom nft",
  "description": "a cool test nft, to test my custom erc721 contract",
  "image": "https://example.com/my-nft-image.png",
  "attributes": [
      {
        "trait_type": "color",
        "value": "blue"
      }
    ]
}
```

you will need to store these metadata files somewhere, obtain their urls and then when you mint your token, associate the correct uri to the token id using the `mint` function.

now let's say that we also want to have royalties in our contract, that's another extra functionality that is not included in the minimal implementation. we can add a simple royalty implementation for a sale on any marketplace, to demonstrate how easy it's to implement it, we’re gonna rely on the eip2981 standard.

```solidity
    // eip2981 royalty implementation
   mapping(address => uint96) private _royalties;

    function setRoyalty(address recipient, uint96 value) public {
        _royalties[recipient] = value;
    }


    function royaltyInfo(uint256 tokenId, uint256 salePrice)
        external
        view
        returns (address receiver, uint256 royaltyAmount)
    {
       address owner = ownerOf(tokenId);
        for(address royaltyRecipient;
            uint256 i < _royalties.length;
            unchecked { i++ }
        ){
            (receiver, royaltyAmount) = (royaltyRecipient, (salePrice * _royalties[royaltyRecipient]) / 10000);
            if(royaltyAmount>0) return (receiver, royaltyAmount);
        }
            return (address(0), 0);
    }
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
       return interfaceId == 0x80ac58cd || // ERC721
              interfaceId == 0x5b5e139f || // ERC721Metadata
               interfaceId == 0x2a55205a; // eip2981
    }
```

here, we added the `royaltyInfo` function to calculate the royalty amount, and we modified the supports interface, to include the eip2981 standard. if you want to implement more sophisticated logic, you would need to expand upon this. this is just a simple example.

now, can it be rendered on opensea? yes, absolutely. opensea relies on the erc721 standard events and metadata to correctly display an nft. as long as your contract emits the `transfer` and `approval` events and correctly maps the `tokenURI` to the correct metadata url, your token will be rendered correctly on opensea. the fact that your contract does not use the openzeppelin library does not matter at all. opensea does not care if you use openzeppelin, they only check if your contract correctly follows the erc721 interface.

of course, building your own contract without openzeppelin increases the chance of security bugs, so it requires a solid understanding of the solidity language, gas optimization techniques and all the best practices to build a secure and optimized contract.

if you are interested in a more deep understanding of erc721 standard and smart contract development in general, i would recommend to take a look at the yellow paper of ethereum it's a pretty dense paper but a must read. a book that i would recommend is "mastering ethereum" by andreas antonopoulos and gavin wood, it's a great resource for learning more about the underlying concepts of ethereum. and also read the eip documentation carefully, it contains everything you need to know about every standard. the documentation of solidity is also really useful to understand the language features. and, of course, practice. practice, practice, and practice. build things, break things, fix things. that’s how you truly learn. after all, solidity is not that hard, i think my hamster can learn it, with some help of course, it only needs to understand the concept of state changing and the event system.

that should give you a very good start. let me know if you have more questions.
