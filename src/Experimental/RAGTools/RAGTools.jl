"""
    RAGTools

Provides Retrieval-Augmented Generation (RAG) functionality.

Requires: LinearAlgebra, SparseArrays, Unicode, PromptingTools for proper functionality.

This module is experimental and may change at any time. It is intended to be moved to a separate package in the future.
"""
module RAGTools

# External Dependencies
# ====================
using AbstractTrees
using AbstractTrees: PreOrderDFS
using HTTP
using JSON3
using JSON3: StructTypes
using Pinecone: Pinecone, PineconeContextv3, PineconeIndexv3, init_v3, Index, PineconeVector, upsert
using PromptingTools
using PromptingTools: pprint, AbstractMessage
using PromptingTools.Experimental.APITools: create_websearch
using UUIDs: UUIDs, uuid4

# Constants
# =========
const PT = PromptingTools

# Include Files 
# ==============
include("utils.jl")
# reexport
export pprint

## export trigrams, trigrams_hashed, text_to_trigrams, text_to_trigrams_hashed
## export STOPWORDS, tokenize, split_into_code_and_sentences
# export merge_kwargs_nested
export getpropertynested, setpropertynested

# eg, cohere_api
include("api_services.jl")

include("rag_interface.jl")

include("types/Types.jl")
export ChunkIndex, ChunkKeywordsIndex, ChunkEmbeddingsIndex, PineconeIndex, CandidateChunks, CandidateWithChunks, RAGResult
export MultiIndex, SubChunkIndex, MultiCandidateChunks

include("preparation.jl")
export build_index, get_chunks, get_embeddings, get_keywords, get_tags, SimpleIndexer,
       KeywordsIndexer, PineconeIndexer

include("rank_gpt.jl")

include("retrieval.jl")
export retrieve, SimpleRetriever, SimpleBM25Retriever, AdvancedRetriever
export find_closest, find_tags, rerank, rephrase

include("generation.jl")
export airag, build_context!, generate!, refine!, answer!, postprocess!
export SimpleGenerator, AdvancedGenerator, RAGConfig

include("annotation.jl")
export annotate_support, TrigramAnnotater, print_html

include("evaluation.jl")
export build_qa_evals, run_qa_evals

end
