---
title: "How are ERC-721-based DAOs structured?"
date: "2024-12-23"
id: "how-are-erc-721-based-daos-structured"
---

, let's dive into the intricacies of how erc-721-based daos are structured. I recall tackling a similar architecture back in my early days working on a platform for digital collectibles—it involved a fascinating intersection of token mechanics and governance. We needed a robust, yet flexible system, which led us to explore these very principles.

At their core, erc-721-based daos leverage the unique properties of non-fungible tokens (nfts) to facilitate governance. Unlike erc-20 tokens, where each token is identical and represents a proportional share, erc-721 tokens represent unique assets. In this context, an erc-721 token might represent a single voting right, membership to the dao, or even a specific level of influence, depending on the implementation. This difference is not merely semantic; it fundamentally alters how governance mechanisms are designed. The inherent individuality of each token allows for more complex and nuanced voting and membership structures.

Here's a breakdown of how this usually works: instead of standard vote weighting based on token holdings, each erc-721 token (or a subset of them) typically represents a single vote. This makes sense. Each nft holder often represents a specific identity or entity within the dao's ecosystem. This can lead to a scenario where having many tokens doesn't grant disproportionate power, especially if the system intends to balance influence and limit a single entity's ability to dominate decision-making.

I have seen systems where specific collections of erc-721 nfts grant access to various privileges within the dao, a concept often referred to as “tiered access.” For example, holders of 'founder' nfts might get greater voting power, specific participation rights, or other privileges not granted to general members holding a standard membership nft. This stratification is not always the goal, of course, and systems might be designed with a purely democratic one-nft-one-vote principle as well. It really comes down to the purpose the dao is trying to achieve.

In practical terms, such a system usually involves a smart contract serving as the backbone of the dao. It maintains a record of all holders of the specified erc-721 tokens and typically includes functionalities such as proposal creation, voting, and execution of decisions. This smart contract is also where any restrictions, like tiered access rights, are encoded and enforced.

Now, let's get into a few code snippets. These examples are simplified to illustrate the key concepts. Keep in mind real-world implementations would need further error handling, access control, and other security considerations.

**Snippet 1: Basic Voting Mechanism (Solidity)**

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract BasicNftDao {
    using Counters for Counters.Counter;
    Counters.Counter public proposalCount;

    IERC721 public membershipNft;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    
    struct Proposal {
        string description;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    constructor(address _membershipNftAddress) {
        membershipNft = IERC721(_membershipNftAddress);
    }

    function createProposal(string memory _description) public {
        proposalCount.increment();
        uint256 proposalId = proposalCount.current();
        proposals[proposalId] = Proposal({
            description: _description,
            yesVotes: 0,
            noVotes: 0,
            executed: false
        });
    }

    function vote(uint256 _proposalId, bool _vote) public {
        require(membershipNft.balanceOf(msg.sender) > 0, "Must own an nft to vote.");
        require(!hasVoted[_proposalId][msg.sender], "Already voted.");

        if (_vote) {
            proposals[_proposalId].yesVotes += 1;
        } else {
           proposals[_proposalId].noVotes += 1;
        }
        hasVoted[_proposalId][msg.sender] = true;
    }

    function executeProposal(uint256 _proposalId) public {
      require(proposals[_proposalId].yesVotes > proposals[_proposalId].noVotes, "Proposal did not pass");
      require(!proposals[_proposalId].executed, "Proposal already executed");
      proposals[_proposalId].executed = true;
      // Implement logic to execute the proposal here.
    }

}
```
In this snippet, each nft owner has one vote. The code defines basic functionality for creating proposals, casting a vote (yes or no), and executing successful proposals. This provides an elementary structure.

**Snippet 2: Tiered Access Voting**

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract TieredNftDao {
    using Counters for Counters.Counter;
    Counters.Counter public proposalCount;

    IERC721 public membershipNft;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    mapping(uint256 => uint256) public tokenVoteWeight; // mapping tokenId -> voteWeight
    uint256 public founderTokenId;

    struct Proposal {
        string description;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    constructor(address _membershipNftAddress, uint256 _founderTokenId) {
        membershipNft = IERC721(_membershipNftAddress);
        founderTokenId = _founderTokenId;
        tokenVoteWeight[_founderTokenId] = 3; // Founder token gets 3 votes.
    }


    function createProposal(string memory _description) public {
        proposalCount.increment();
        uint256 proposalId = proposalCount.current();
        proposals[proposalId] = Proposal({
            description: _description,
            yesVotes: 0,
            noVotes: 0,
            executed: false
        });
    }

  function vote(uint256 _proposalId, uint256 _tokenId, bool _vote) public {
      require(membershipNft.ownerOf(_tokenId) == msg.sender, "Must own the token to vote.");
        require(!hasVoted[_proposalId][msg.sender], "Already voted.");

        uint256 voteWeight = tokenVoteWeight[_tokenId];
        if (voteWeight == 0) {
            voteWeight = 1; // Default to 1 vote if no other weight specified
        }
      
      if (_vote) {
          proposals[_proposalId].yesVotes += voteWeight;
        } else {
          proposals[_proposalId].noVotes += voteWeight;
      }
      hasVoted[_proposalId][msg.sender] = true;
  }


    function executeProposal(uint256 _proposalId) public {
       require(proposals[_proposalId].yesVotes > proposals[_proposalId].noVotes, "Proposal did not pass");
      require(!proposals[_proposalId].executed, "Proposal already executed");
       proposals[_proposalId].executed = true;
        // Implement logic to execute the proposal here.
    }

}
```
This modification introduces a `tokenVoteWeight` mapping, allowing the dao to assign different vote weights to different nfts. A specific token, such as the `founderTokenId` is initially assigned more voting power.

**Snippet 3: Minimal Voting with Erc-721 metadata**

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract MetadataBasedDao {
    using Counters for Counters.Counter;
    Counters.Counter public proposalCount;

    IERC721 public membershipNft;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    
    struct Proposal {
        string description;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    constructor(address _membershipNftAddress) {
        membershipNft = IERC721(_membershipNftAddress);
    }

    function createProposal(string memory _description) public {
        proposalCount.increment();
        uint256 proposalId = proposalCount.current();
        proposals[proposalId] = Proposal({
            description: _description,
            yesVotes: 0,
            noVotes: 0,
            executed: false
        });
    }

    function vote(uint256 _proposalId, uint256 _tokenId, bool _vote) public {
         require(membershipNft.ownerOf(_tokenId) == msg.sender, "Must own the token to vote.");
         require(!hasVoted[_proposalId][msg.sender], "Already voted.");

        // Example: Extract a 'level' attribute from the URI (very basic implementation)
         string memory tokenUri = membershipNft.tokenURI(_tokenId);
         uint256 voteWeight = getWeightFromMetadata(tokenUri);

         if (_vote) {
            proposals[_proposalId].yesVotes += voteWeight;
         } else {
             proposals[_proposalId].noVotes += voteWeight;
         }
        hasVoted[_proposalId][msg.sender] = true;
    }

    function getWeightFromMetadata(string memory _tokenUri) internal pure returns (uint256) {
         // This is highly simplified, and production systems use off-chain metadata processing.
         // Assumes token URI contains a numeric level (e.g., "metadata.json?level=3")

        bytes memory tokenUriBytes = bytes(_tokenUri);
        uint256 levelStartIndex = 0;

        for (uint i=0; i < tokenUriBytes.length - 7; i++){
            if(tokenUriBytes[i] == bytes1("l") &&
             tokenUriBytes[i+1] == bytes1("e") &&
            tokenUriBytes[i+2] == bytes1("v") &&
            tokenUriBytes[i+3] == bytes1("e") &&
            tokenUriBytes[i+4] == bytes1("l") &&
            tokenUriBytes[i+5] == bytes1("=")){
                levelStartIndex = i+6;
                break;
            }
         }
         if(levelStartIndex > 0) {
            return Strings.parseInt(bytes(tokenUriBytes[levelStartIndex:tokenUriBytes.length]));
         } else {
            return 1; // Default weight
         }
    }

    function executeProposal(uint256 _proposalId) public {
      require(proposals[_proposalId].yesVotes > proposals[_proposalId].noVotes, "Proposal did not pass");
       require(!proposals[_proposalId].executed, "Proposal already executed");
       proposals[_proposalId].executed = true;
       // Implement logic to execute the proposal here.
    }
}
```

In this example, I added a function that pulls information from the nft's token uri to determine voting weight dynamically, allowing for attributes encoded in the metadata to impact voting. This is a very basic example and requires an off-chain mechanism for storing and processing the metadata but it illustrates a potential direction.

For anyone looking to dive deeper, I'd recommend starting with the OpenZeppelin Contracts library, specifically the erc721 modules. “Mastering Ethereum” by Andreas M. Antonopoulos and Gavin Wood provides excellent foundations as well. Additionally, the EIP-712 documentation is invaluable when dealing with secure signature mechanisms, and you could further investigate how snapshot.org does its off-chain voting aggregation for an understanding of real world usage patterns.

In short, structuring an erc-721 based dao involves creatively applying these token's inherent characteristics to governance. It’s a flexible approach that can be tailored to very different requirements by adjusting the smart contract logic. It requires careful consideration of the dao's objectives, the level of control members should wield, and the desired balance of power.
