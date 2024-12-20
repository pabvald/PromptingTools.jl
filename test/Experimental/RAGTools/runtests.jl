using Test
using HTTP, JSON3
using LinearAlgebra, Random, SparseArrays, Unicode
using PromptingTools
using PromptingTools.AbstractTrees
using PromptingTools.Experimental.RAGTools
using Snowball

const PT = PromptingTools
const RT = PromptingTools.Experimental.RAGTools

@testset "RAGTools" begin
    include("utils.jl")
    include("types/Types.jl")
    include("preparation.jl")
    include("rank_gpt.jl")
    include("retrieval.jl")
    include("generation.jl")
    include("annotation.jl")
    include("evaluation.jl")
end
