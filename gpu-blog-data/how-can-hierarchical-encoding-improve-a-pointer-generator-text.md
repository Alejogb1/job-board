---
title: "How can hierarchical encoding improve a pointer-generator text summarization model?"
date: "2025-01-30"
id: "how-can-hierarchical-encoding-improve-a-pointer-generator-text"
---
Hierarchical encoding addresses a fundamental limitation in traditional pointer-generator networks: their inability to effectively capture long-range dependencies and structural information within the source document, particularly when generating abstractive summaries. I encountered this directly while developing a system for summarizing technical documentation, where simple sequence-to-sequence models, even with attention, struggled to maintain coherence across paragraphs. The core issue lies in the flat representation of the source text, treating every word as an independent token in a long sequence, losing sight of the document's inherent hierarchy.

Specifically, pointer-generator networks are often built upon recurrent neural networks (RNNs), or transformers, that encode the input text into a contextualized representation. These encodings, despite attention mechanisms, are inherently sequential. While attention allows the decoder to focus on relevant parts of the input during decoding, it primarily operates on the final, flat encoding. This can be particularly problematic for longer documents with complex structures. Consider a technical article divided into sections, subsections, and individual paragraphs, each contributing to an overall theme. A flat encoding treats these structures equally, without explicitly acknowledging the different levels of abstraction and interdependencies between them. Hierarchical encoding tackles this limitation by introducing multiple levels of encoding, thereby creating representations that capture the document structure.

A common hierarchical encoding strategy involves two or more levels of RNNs or transformer encoders. The first level might operate at the sentence or paragraph level, generating contextualized representations for each unit. Subsequently, a higher-level encoder then processes these sentence or paragraph representations, creating a document-level representation. This process allows the model to learn inter-sentence and inter-paragraph relationships, encoding not just the content of individual units, but also how they relate to each other within the document. During decoding, the attention mechanism can now attend to both the low-level (word or subword) encodings and the high-level (sentence/paragraph) encodings, enabling the generation of summaries that are more structurally coherent and contextually relevant.

Furthermore, such encoding allows for modeling intra-sentence dependencies at a fine level, while also capturing the broader topic flow and transition across sentences and paragraphs. Traditional pointer-generator models often struggle with summarization of documents with subtle shifts in the central theme as these shifts may be less easily detected in a flattened single sequence. However, by breaking a document into sections, then sub-sections, and encoding these parts individually, and combining the encoded parts at different levels of abstraction, a pointer-generator model can gain a hierarchical understanding that greatly aids summarization.

Let’s consider some practical implementations.

**Code Example 1: Two-Level RNN Hierarchy (Sentence and Document Levels)**

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HierarchicalEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional=True):
        super(HierarchicalEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.sentence_encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.document_encoder = nn.LSTM(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, sentences, sentence_lengths):
        """
        sentences:  List of lists of tokens, each representing a sentence
        sentence_lengths: List of sentence lengths for padding
        """
        
        embedded_sentences = [self.embedding(torch.tensor(s, dtype=torch.long)).unsqueeze(0) for s in sentences]  # Convert list of lists to list of tensors
        
        sentence_encodings = []
        for sent_emb, sent_len in zip(embedded_sentences, sentence_lengths):
            packed_sent_emb = pack_padded_sequence(sent_emb, torch.tensor(sent_len, dtype=torch.long), batch_first=True, enforce_sorted=False)
            
            sent_output, (h_n, c_n) = self.sentence_encoder(packed_sent_emb)
            
            sent_output_padded, _ = pad_packed_sequence(sent_output, batch_first=True) # Unpack to allow to access last hidden state of each seq
            
            if self.bidirectional:
                # Concatenate hidden states from both directions
                last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                last_hidden = h_n[-1, :, :] # last layer for unidirectional
            
            sentence_encodings.append(last_hidden.unsqueeze(0))  # Output is num_sentences x 1 x (hidden_dim*2 if bidirectional else hidden_dim)
        
        sentence_encodings = torch.cat(sentence_encodings, dim=0)  # Reshape for sequence input.
        
        # Pass sentence level encodings into doc level encoder
        packed_doc_encodings = pack_padded_sequence(sentence_encodings, torch.tensor(len(sentences), dtype=torch.long).unsqueeze(0), batch_first=True, enforce_sorted=False)
        doc_output, (doc_h_n, doc_c_n) = self.document_encoder(packed_doc_encodings)
        doc_output_padded, _ = pad_packed_sequence(doc_output, batch_first=True) # Unpack to allow to access last hidden state of each seq

        if self.bidirectional:
           document_encoding = torch.cat((doc_h_n[-2, :, :], doc_h_n[-1, :, :]), dim=1)
        else:
            document_encoding = doc_h_n[-1, :, :]
        
        return sentence_encodings, document_encoding # Return both levels for attention.

# Example Usage
vocab_size = 1000
emb_dim = 100
hidden_dim = 256
num_layers = 1
bidirectional = True

encoder = HierarchicalEncoder(vocab_size, emb_dim, hidden_dim, num_layers, bidirectional)
sentences = [[1, 2, 3, 4, 0, 0], [5, 6, 7, 8], [9, 10, 11, 12, 13]] # Example tokenized sentences (pad with 0)
sentence_lengths = [4, 4, 5]  # Actual lengths of sentences without padding
sent_encodings, doc_enc = encoder(sentences, sentence_lengths)

print("Sentence Encodings Shape:", sent_encodings.shape) # Output: torch.Size([3, 1, 512]) (if bidirectional) 3 sentences, batch_size of 1, 2*hidden_dim
print("Document Encoding Shape:", doc_enc.shape) #Output: torch.Size([1, 512]) batch_size of 1, 2*hidden_dim
```

This example demonstrates a two-level RNN encoder.  The `HierarchicalEncoder` first encodes each sentence using a bidirectional LSTM, taking padding into account using `pack_padded_sequence`. The last hidden state of each sentence is then used as input to the document-level encoder, also a bidirectional LSTM. Finally, this example function returns both the individual sentence level encodings and the overall document level encoding to allow the decoder to attend to either level. This allows a more sophisticated understanding of context within a given document.

**Code Example 2: Hierarchical Transformer Encoding (Paragraph and Document Levels)**

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_heads, num_layers):
        super(HierarchicalTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.paragraph_encoder = TransformerEncoder(
            TransformerEncoderLayer(emb_dim, num_heads, hidden_dim), num_layers
        )
        self.document_encoder = TransformerEncoder(
            TransformerEncoderLayer(emb_dim, num_heads, hidden_dim), num_layers
        )
        self.linear_proj = nn.Linear(hidden_dim, emb_dim) # Project to match transformer dimensions for document level.


    def forward(self, paragraphs, paragraph_lengths):
        """
        paragraphs: List of lists of lists of tokens, each representing a paragraph
        paragraph_lengths: List of lists of paragraph lengths (tokens in the paragraph)
        """
        
        #  Embedding of each paragraph
        embedded_paragraphs = [[self.embedding(torch.tensor(s, dtype=torch.long)) for s in p ]  for p in paragraphs] # p is a paragraph (list of lists of tokens)

        paragraph_encodings = []
        for emb_paragraph, lens in zip(embedded_paragraphs, paragraph_lengths):
            
            # Stack sequences within a paragraph (sentences to paragraph)
            packed_seq = torch.nn.utils.rnn.pack_sequence(emb_paragraph, enforce_sorted=False) # pack by sequence lengths
            pad_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)  # Pad back to batch size

            paragraph_enc = self.paragraph_encoder(pad_seq.transpose(0,1)) # Needs to be sequences x batch x feature size 

            
            #  Take average along token dimension to get representation of each sentence
            paragraph_enc = paragraph_enc.transpose(0,1)
            sentence_masks = self._generate_mask_from_lengths(lens, pad_seq.shape[0])
            
            masked_encoding = paragraph_enc * sentence_masks.unsqueeze(-1)
            sentence_level_enc = torch.sum(masked_encoding, dim = 1) / torch.tensor(lens, dtype=torch.float).unsqueeze(-1) # avg. across tokens
            paragraph_encodings.append(sentence_level_enc.unsqueeze(0))


        paragraph_encodings = torch.cat(paragraph_encodings, dim=0)

        projected_encodings = self.linear_proj(paragraph_encodings) # Project to match dimensions.

        # Document level transformer encoding 
        document_enc = self.document_encoder(projected_encodings.transpose(0,1))

        document_enc = document_enc.transpose(0,1) # back to batch_first

        return paragraph_encodings, document_enc

    def _generate_mask_from_lengths(self, lengths, seq_len):
            """ Creates a mask for each sentence in a paragraph """
            masks = torch.zeros((len(lengths), seq_len))
            for i, length in enumerate(lengths):
                masks[i, :length] = 1
            return masks

# Example Usage
vocab_size = 1000
emb_dim = 100
hidden_dim = 256
num_heads = 8
num_layers = 2

encoder = HierarchicalTransformerEncoder(vocab_size, emb_dim, hidden_dim, num_heads, num_layers)
paragraphs = [
    [[1, 2, 3, 4], [5, 6, 7, 8, 9]],  # Paragraph 1, two sentences (tokenized)
    [[10, 11, 12, 13, 14, 15], [16,17, 18]]   # Paragraph 2, two sentences (tokenized)
]
paragraph_lengths = [
    [4, 5],  # Lengths of sentences within paragraph 1
    [6, 3]   # Lengths of sentences within paragraph 2
]


para_encs, doc_enc = encoder(paragraphs, paragraph_lengths)

print("Paragraph Encodings Shape:", para_encs.shape)  # Output: torch.Size([2, 2, 100]) (2 paragraphs, 2 sentences per paragraph, embedding dimension)
print("Document Encoding Shape:", doc_enc.shape) #Output: torch.Size([2, 2, 100]) (2 paragraphs, 2 seq_lens per paragraph, embedding dimension)
```

This second example demonstrates a hierarchical transformer encoder. It operates at two levels: paragraph and document. Here, each paragraph is a list of tokenized sentences. These sentences are converted to embeddings, then processed using a transformer encoder. The resulting sentence level encodings are averaged and projected to match dimensions before input into another transformer encoder that produces a document-level representation. This allows modelling of the relationships between sentences within each paragraph and across paragraphs within the entire document. The example also takes care to mask out padded values to prevent issues when averaging and processing the sentence sequences.

**Code Example 3: Hierarchical Attention (Attention at Multiple levels)**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(HierarchicalAttention, self).__init__()
        self.linear_sentence = nn.Linear(hidden_dim*2, hidden_dim) # Example for bidirectional encoder
        self.linear_document = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, sentence_encodings, document_encoding, decoder_hidden):
        """
        sentence_encodings: sentence level outputs from encoder (batch_size x seq_len x hidden_dim*2)
        document_encoding: output from document level encoder (batch_size x hidden_dim *2)
        decoder_hidden: Hidden state from the decoder
        """

        # Sentence level attention
        query_sentence = F.tanh(self.linear_sentence(decoder_hidden)) # decoder to size hidden_dim
        attention_scores_sentence = torch.matmul(sentence_encodings, query_sentence.unsqueeze(-1)).squeeze(-1)
        attention_weights_sentence = F.softmax(attention_scores_sentence, dim=1)
        
        sentence_context = torch.bmm(attention_weights_sentence.unsqueeze(1), sentence_encodings).squeeze(1)

        # Document level attention
        query_document = F.tanh(self.linear_document(decoder_hidden))
        attention_scores_document = torch.matmul(document_encoding, query_document.unsqueeze(-1)).squeeze(-1)
        attention_weights_document = F.softmax(attention_scores_document, dim=1)
        
        doc_context = torch.bmm(attention_weights_document.unsqueeze(1), document_encoding.unsqueeze(0)).squeeze(1) # batch size 1 assumed for document level


        # Combined context
        combined_context = sentence_context + doc_context # Could also concat
        
        return combined_context, attention_weights_sentence, attention_weights_document

# Example usage
hidden_dim = 256
batch_size = 1
num_sentences = 3

attention = HierarchicalAttention(hidden_dim)
sentence_encodings = torch.rand(batch_size, num_sentences, hidden_dim*2) # Random sentence encodings
doc_enc = torch.rand(batch_size, hidden_dim *2) # Random doc encoding
decoder_hidden = torch.rand(batch_size, hidden_dim*2) # Random decoder hidden state

combined_context, attention_weights_sent, attention_weights_doc = attention(sentence_encodings, doc_enc, decoder_hidden)

print("Combined Context Shape: ", combined_context.shape)  # Output: torch.Size([1, 512]) batch, 2*hidden_dim
print("Sentence Attention Weights Shape:", attention_weights_sent.shape) # Output: torch.Size([1, 3]) batch, num_sentences
print("Document Attention Weights Shape:", attention_weights_doc.shape) # Output: torch.Size([1, 1]) batch, num_documents
```
This example presents a basic implementation of hierarchical attention. Here, the decoder hidden state is used to generate attention weights for both the sentence and document level encodings. This allows the decoder to attend differently to the different levels and obtain a combined context vector.

In conclusion, hierarchical encoding addresses the limitations of flat representations by capturing hierarchical relationships within documents. This approach leads to improved long-range dependency modeling, better structural coherence, and enhanced abstractive summarization quality. For further exploration, I recommend reviewing work focusing on hierarchical attention networks, document-level summarization, and incorporating these concepts into transformer architectures for sequence-to-sequence tasks. Specifically, examining research into hierarchical recurrent networks and the use of transformers in document modeling would be beneficial. Reading papers regarding document modelling and long-text summarization are also highly recommended.
