import { _ as _export_sfc, c as createElementBlock, o as openBlock, a7 as createStaticVNode } from "./chunks/framework.D6uUG_wu.js";
const __pageData = JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"prompts/RAG.md","filePath":"prompts/RAG.md","lastUpdated":null}');
const _sfc_main = { name: "prompts/RAG.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<p>The following file is auto-generated from the <code>templates</code> folder. For any changes, please modify the source files in the <code>templates</code> folder.</p><p>To use these templates in <code>aigenerate</code>, simply provide the template name as a symbol, eg, <code>aigenerate(:MyTemplate; placeholder1 = value1)</code></p><h2 id="Basic-Rag-Templates" tabindex="-1">Basic-Rag Templates <a class="header-anchor" href="#Basic-Rag-Templates" aria-label="Permalink to &quot;Basic-Rag Templates {#Basic-Rag-Templates}&quot;">​</a></h2><h3 id="Template:-RAGAnswerFromContext" tabindex="-1">Template: RAGAnswerFromContext <a class="header-anchor" href="#Template:-RAGAnswerFromContext" aria-label="Permalink to &quot;Template: RAGAnswerFromContext {#Template:-RAGAnswerFromContext}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Answers the provided Questions based on the Context. Placeholders: <code>question</code>, <code>context</code></p></li><li><p>Placeholders: <code>context</code>, <code>question</code></p></li><li><p>Word count: 375</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Act as a world-class AI assistant with access to the latest knowledge via Context Information. </span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions:**</span></span>\n<span class="line"><span>- Answer the question based only on the provided Context.</span></span>\n<span class="line"><span>- If you don&#39;t know the answer, just say that you don&#39;t know, don&#39;t try to make up an answer.</span></span>\n<span class="line"><span>- Be brief and concise.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Context Information:**</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{context}}</span></span>\n<span class="line"><span>---</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># Question</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{question}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Answer</span></span></code></pre></div><h2 id="Metadata-Templates" tabindex="-1">Metadata Templates <a class="header-anchor" href="#Metadata-Templates" aria-label="Permalink to &quot;Metadata Templates {#Metadata-Templates}&quot;">​</a></h2><h3 id="Template:-RAGExtractMetadataLong" tabindex="-1">Template: RAGExtractMetadataLong <a class="header-anchor" href="#Template:-RAGExtractMetadataLong" aria-label="Permalink to &quot;Template: RAGExtractMetadataLong {#Template:-RAGExtractMetadataLong}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Extracts metadata from the provided text using longer instructions set and examples. If you don&#39;t have any special instructions, provide <code>instructions=&quot;None.&quot;</code>. Placeholders: <code>text</code>, <code>instructions</code></p></li><li><p>Placeholders: <code>text</code>, <code>instructions</code></p></li><li><p>Word count: 1384</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You&#39;re a world-class data extraction engine built by OpenAI together with Google and to extract filter metadata to power the most advanced search engine in the world. </span></span>\n<span class="line"><span>    </span></span>\n<span class="line"><span>    **Instructions for Extraction:**</span></span>\n<span class="line"><span>    1. Carefully read through the provided Text</span></span>\n<span class="line"><span>    2. Identify and extract:</span></span>\n<span class="line"><span>       - All relevant entities such as names, places, dates, etc.</span></span>\n<span class="line"><span>       - Any special items like technical terms, unique identifiers, etc.</span></span>\n<span class="line"><span>       - In the case of Julia code or Julia documentation: specifically extract package names, struct names, function names, and important variable names (eg, uppercased variables)</span></span>\n<span class="line"><span>    3. Keep extracted values and categories short. Maximum 2-3 words!</span></span>\n<span class="line"><span>    4. You can only extract 3-5 items per Text, so select the most important ones.</span></span>\n<span class="line"><span>    5. Assign a search filter Category to each extracted Value</span></span>\n<span class="line"><span>    </span></span>\n<span class="line"><span>    **Example 1:**</span></span>\n<span class="line"><span>    - Document Chunk: &quot;Dr. Jane Smith published her findings on neuroplasticity in 2021. The research heavily utilized the DataFrames.jl and Plots.jl packages.&quot;</span></span>\n<span class="line"><span>    - Extracted keywords:</span></span>\n<span class="line"><span>      - Name: Dr. Jane Smith</span></span>\n<span class="line"><span>      - Date: 2021</span></span>\n<span class="line"><span>      - Technical Term: neuroplasticity</span></span>\n<span class="line"><span>      - JuliaPackage: DataFrames.jl, Plots.jl</span></span>\n<span class="line"><span>      - JuliaLanguage:</span></span>\n<span class="line"><span>      - Identifier:</span></span>\n<span class="line"><span>      - Other: </span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>    If the user provides special instructions, prioritize these over the general instructions.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># Text</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{text}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Special Instructions</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{instructions}}</span></span></code></pre></div><h3 id="Template:-RAGExtractMetadataShort" tabindex="-1">Template: RAGExtractMetadataShort <a class="header-anchor" href="#Template:-RAGExtractMetadataShort" aria-label="Permalink to &quot;Template: RAGExtractMetadataShort {#Template:-RAGExtractMetadataShort}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Extracts metadata from the provided text. If you don&#39;t have any special instructions, provide <code>instructions=&quot;None.&quot;</code>. Placeholders: <code>text</code>, <code>instructions</code></p></li><li><p>Placeholders: <code>text</code>, <code>instructions</code></p></li><li><p>Word count: 278</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Extract search keywords and their categories from the Text provided below (format &quot;value:category&quot;). Each keyword must be at most 2-3 words. Provide at most 3-5 keywords. I will tip you $50 if the search is successful.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># Text</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{text}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Special Instructions</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{instructions}}</span></span></code></pre></div><h2 id="Refinement-Templates" tabindex="-1">Refinement Templates <a class="header-anchor" href="#Refinement-Templates" aria-label="Permalink to &quot;Refinement Templates {#Refinement-Templates}&quot;">​</a></h2><h3 id="Template:-RAGAnswerRefiner" tabindex="-1">Template: RAGAnswerRefiner <a class="header-anchor" href="#Template:-RAGAnswerRefiner" aria-label="Permalink to &quot;Template: RAGAnswerRefiner {#Template:-RAGAnswerRefiner}&quot;">​</a></h3><ul><li><p>Description: For RAG applications (refine step), gives model the ability to refine its answer based on some additional context etc.. The hope is that it better answers the original query. Placeholders: <code>query</code>, <code>answer</code>, <code>context</code></p></li><li><p>Placeholders: <code>query</code>, <code>answer</code>, <code>context</code></p></li><li><p>Word count: 1074</p></li><li><p>Source: Adapted from <a href="https://github.com/run-llama/llama_index/blob/78af3400ad485e15862c06f0c4972dc3067f880c/llama-index-core/llama_index/core/prompts/default_prompts.py#L81" target="_blank" rel="noreferrer">LlamaIndex</a></p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Act as a world-class AI assistant with access to the latest knowledge via Context Information.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to refine an existing answer if it&#39;s needed.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>The original query is as follows: </span></span>\n<span class="line"><span>{{query}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>The AI model has provided the following answer:</span></span>\n<span class="line"><span>{{answer}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions:**</span></span>\n<span class="line"><span>- Given the new context, refine the original answer to better answer the query.</span></span>\n<span class="line"><span>- If the context isn&#39;t useful, return the original answer.</span></span>\n<span class="line"><span>- If you don&#39;t know the answer, just say that you don&#39;t know, don&#39;t try to make up an answer.</span></span>\n<span class="line"><span>- Be brief and concise.</span></span>\n<span class="line"><span>- Provide the refined answer only and nothing else.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>We have the opportunity to refine the previous answer (only if needed) with some more context below.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Context Information:**</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span>{{context}}</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Given the new context, refine the original answer to better answer the query.</span></span>\n<span class="line"><span>If the context isn&#39;t useful, return the original answer. </span></span>\n<span class="line"><span>Provide the refined answer only and nothing else. You MUST NOT comment on the web search results or the answer - simply provide the answer to the question.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Refined Answer:</span></span></code></pre></div><h3 id="Template:-RAGWebSearchRefiner" tabindex="-1">Template: RAGWebSearchRefiner <a class="header-anchor" href="#Template:-RAGWebSearchRefiner" aria-label="Permalink to &quot;Template: RAGWebSearchRefiner {#Template:-RAGWebSearchRefiner}&quot;">​</a></h3><ul><li><p>Description: For RAG applications (refine step), gives model the ability to refine its answer based on web search results. The hope is that it better answers the original query. Placeholders: <code>query</code>, <code>answer</code>, <code>search_results</code></p></li><li><p>Placeholders: <code>query</code>, <code>answer</code>, <code>search_results</code></p></li><li><p>Word count: 1392</p></li><li><p>Source: Adapted from <a href="https://github.com/run-llama/llama_index/blob/78af3400ad485e15862c06f0c4972dc3067f880c/llama-index-core/llama_index/core/prompts/default_prompts.py#L81" target="_blank" rel="noreferrer">LlamaIndex</a></p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Act as a world-class AI assistant with access to the latest knowledge via web search results.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to refine an existing answer if it&#39;s needed.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>The original query: </span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span>{{query}}</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>The AI model has provided the following answer:</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span>{{answer}}</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions:**</span></span>\n<span class="line"><span>- Given the web search results, refine the original answer to better answer the query.</span></span>\n<span class="line"><span>- Web search results are sometimes irrelevant and noisy. If the results are not relevant for the query, return the original answer from the AI model.</span></span>\n<span class="line"><span>- If the web search results do not improve the original answer, return the original answer from the AI model.</span></span>\n<span class="line"><span>- If you don&#39;t know the answer, just say that you don&#39;t know, don&#39;t try to make up an answer.</span></span>\n<span class="line"><span>- Be brief and concise.</span></span>\n<span class="line"><span>- Provide the refined answer only and nothing else.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>We have the opportunity to refine the previous answer (only if needed) with additional information from web search.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Web Search Results:**</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span>{{search_results}}</span></span>\n<span class="line"><span>-----------------</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Given the new context, refine the original answer to better answer the query.</span></span>\n<span class="line"><span>If the web search results are not useful, return the original answer without any changes.</span></span>\n<span class="line"><span>Provide the refined answer only and nothing else. You MUST NOT comment on the web search results or the answer - simply provide the answer to the question.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Refined Answer:</span></span></code></pre></div><h2 id="Evaluation-Templates" tabindex="-1">Evaluation Templates <a class="header-anchor" href="#Evaluation-Templates" aria-label="Permalink to &quot;Evaluation Templates {#Evaluation-Templates}&quot;">​</a></h2><h3 id="Template:-RAGCreateQAFromContext" tabindex="-1">Template: RAGCreateQAFromContext <a class="header-anchor" href="#Template:-RAGCreateQAFromContext" aria-label="Permalink to &quot;Template: RAGCreateQAFromContext {#Template:-RAGCreateQAFromContext}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Generate Question and Answer from the provided Context. If you don&#39;t have any special instructions, provide <code>instructions=&quot;None.&quot;</code>. Placeholders: <code>context</code>, <code>instructions</code></p></li><li><p>Placeholders: <code>context</code>, <code>instructions</code></p></li><li><p>Word count: 1396</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are a world-class teacher preparing contextual Question &amp; Answer sets for evaluating AI systems.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions for Question Generation:**</span></span>\n<span class="line"><span>1. Analyze the provided Context chunk thoroughly.</span></span>\n<span class="line"><span>2. Formulate a question that:</span></span>\n<span class="line"><span>   - Is specific and directly related to the information in the context chunk.</span></span>\n<span class="line"><span>   - Is not too short or generic; it should require a detailed understanding of the context to answer.</span></span>\n<span class="line"><span>   - Can only be answered using the information from the provided context, without needing external information.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions for Reference Answer Creation:**</span></span>\n<span class="line"><span>1. Based on the generated question, compose a reference answer that:</span></span>\n<span class="line"><span>   - Directly and comprehensively answers the question.</span></span>\n<span class="line"><span>   - Stays strictly within the bounds of the provided context chunk.</span></span>\n<span class="line"><span>   - Is clear, concise, and to the point, avoiding unnecessary elaboration or repetition.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Example 1:**</span></span>\n<span class="line"><span>- Context Chunk: &quot;In 1928, Alexander Fleming discovered penicillin, which marked the beginning of modern antibiotics.&quot;</span></span>\n<span class="line"><span>- Generated Question: &quot;What was the significant discovery made by Alexander Fleming in 1928 and its impact?&quot;</span></span>\n<span class="line"><span>- Reference Answer: &quot;Alexander Fleming discovered penicillin in 1928, which led to the development of modern antibiotics.&quot;</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>If the user provides special instructions, prioritize these over the general instructions.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># Context Information</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{context}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Special Instructions</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{instructions}}</span></span></code></pre></div><h3 id="Template:-RAGJudgeAnswerFromContext" tabindex="-1">Template: RAGJudgeAnswerFromContext <a class="header-anchor" href="#Template:-RAGJudgeAnswerFromContext" aria-label="Permalink to &quot;Template: RAGJudgeAnswerFromContext {#Template:-RAGJudgeAnswerFromContext}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Judge an answer to a question on a scale from 1-5. Placeholders: <code>question</code>, <code>context</code>, <code>answer</code></p></li><li><p>Placeholders: <code>question</code>, <code>context</code>, <code>answer</code></p></li><li><p>Word count: 1415</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You&#39;re an impartial judge. Your task is to evaluate the quality of the Answer provided by an AI assistant in response to the User Question on a scale from 1 to 5.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>1. **Scoring Criteria:**</span></span>\n<span class="line"><span>- **Relevance (1-5):** How well does the provided answer align with the context? </span></span>\n<span class="line"><span>  - *1: Not relevant, 5: Highly relevant*</span></span>\n<span class="line"><span>- **Completeness (1-5):** Does the provided answer cover all the essential points mentioned in the context?</span></span>\n<span class="line"><span>  - *1: Very incomplete, 5: Very complete*</span></span>\n<span class="line"><span>- **Clarity (1-5):** How clear and understandable is the provided answer?</span></span>\n<span class="line"><span>  - *1: Not clear at all, 5: Extremely clear*</span></span>\n<span class="line"><span>- **Consistency (1-5):** How consistent is the provided answer with the overall context?</span></span>\n<span class="line"><span>  - *1: Highly inconsistent, 5: Perfectly consistent*</span></span>\n<span class="line"><span>- **Helpfulness (1-5):** How helpful is the provided answer in answering the user&#39;s question?</span></span>\n<span class="line"><span>  - *1: Not helpful at all, 5: Extremely helpful*</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>2. **Judging Instructions:**</span></span>\n<span class="line"><span>- As an impartial judge, please evaluate the provided answer based on the above criteria. </span></span>\n<span class="line"><span>- Assign a score from 1 to 5 for each criterion, considering the original context, question and the provided answer.</span></span>\n<span class="line"><span>- The Final Score is an average of these individual scores, representing the overall quality and relevance of the provided answer. It must be between 1-5.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>```</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># User Question</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{question}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Context Information</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{context}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Assistant&#39;s Answer</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{answer}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Judge&#39;s Evaluation</span></span></code></pre></div><h3 id="Template:-RAGJudgeAnswerFromContextShort" tabindex="-1">Template: RAGJudgeAnswerFromContextShort <a class="header-anchor" href="#Template:-RAGJudgeAnswerFromContextShort" aria-label="Permalink to &quot;Template: RAGJudgeAnswerFromContextShort {#Template:-RAGJudgeAnswerFromContextShort}&quot;">​</a></h3><ul><li><p>Description: For RAG applications. Simple and short prompt to judge answer to a question on a scale from 1-5. Placeholders: <code>question</code>, <code>context</code>, <code>answer</code></p></li><li><p>Placeholders: <code>question</code>, <code>context</code>, <code>answer</code></p></li><li><p>Word count: 420</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You re an impartial judge. </span></span>\n<span class="line"><span>Read carefully the provided question and the answer based on the context. </span></span>\n<span class="line"><span>Provide a rating on a scale 1-5 (1=worst quality, 5=best quality) that reflects how relevant, helpful, clear, and consistent with the provided context the answer was.</span></span>\n<span class="line"><span>```</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># User Question</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{question}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Context Information</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{context}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Assistant&#39;s Answer</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{answer}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span># Judge&#39;s Evaluation</span></span></code></pre></div><h2 id="Query-Transformations-Templates" tabindex="-1">Query-Transformations Templates <a class="header-anchor" href="#Query-Transformations-Templates" aria-label="Permalink to &quot;Query-Transformations Templates {#Query-Transformations-Templates}&quot;">​</a></h2><h3 id="Template:-RAGJuliaQueryHyDE" tabindex="-1">Template: RAGJuliaQueryHyDE <a class="header-anchor" href="#Template:-RAGJuliaQueryHyDE" aria-label="Permalink to &quot;Template: RAGJuliaQueryHyDE {#Template:-RAGJuliaQueryHyDE}&quot;">​</a></h3><ul><li><p>Description: For Julia-specific RAG applications (rephrase step), inspired by the HyDE approach where it generates a hypothetical passage that answers the provided user query to improve the matched results. This explicitly requires and optimizes for Julia-specific questions. Placeholders: <code>query</code></p></li><li><p>Placeholders: <code>query</code></p></li><li><p>Word count: 390</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You&#39;re an world-class AI assistant specialized in Julia language questions.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to generate a BRIEF and SUCCINCT hypothetical passage from Julia language ecosystem documentation that answers the provided query.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Query: {{query}}</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Write a hypothetical snippet with 20-30 words that would be the perfect answer to the query. Try to include as many key details as possible. </span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Passage:</span></span></code></pre></div><h3 id="Template:-RAGQueryHyDE" tabindex="-1">Template: RAGQueryHyDE <a class="header-anchor" href="#Template:-RAGQueryHyDE" aria-label="Permalink to &quot;Template: RAGQueryHyDE {#Template:-RAGQueryHyDE}&quot;">​</a></h3><ul><li><p>Description: For RAG applications (rephrase step), inspired by the HyDE paper where it generates a hypothetical passage that answers the provided user query to improve the matched results. Placeholders: <code>query</code></p></li><li><p>Placeholders: <code>query</code></p></li><li><p>Word count: 354</p></li><li><p>Source: Adapted from <a href="https://github.com/run-llama/llama_index/blob/78af3400ad485e15862c06f0c4972dc3067f880c/llama-index-core/llama_index/core/prompts/default_prompts.py#L351" target="_blank" rel="noreferrer">LlamaIndex</a></p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are a world-class search expert specializing in query transformations.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to write a hypothetical passage that would answer the below question in the most effective way possible.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>It must have 20-30 words and be directly aligned with the intended search objective.</span></span>\n<span class="line"><span>Try to include as many key details as possible.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Query: {{query}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Passage:</span></span></code></pre></div><h3 id="Template:-RAGQueryOptimizer" tabindex="-1">Template: RAGQueryOptimizer <a class="header-anchor" href="#Template:-RAGQueryOptimizer" aria-label="Permalink to &quot;Template: RAGQueryOptimizer {#Template:-RAGQueryOptimizer}&quot;">​</a></h3><ul><li><p>Description: For RAG applications (rephrase step), it rephrases the original query to attract more diverse set of potential search results. Placeholders: <code>query</code></p></li><li><p>Placeholders: <code>query</code></p></li><li><p>Word count: 514</p></li><li><p>Source: Adapted from <a href="https://github.com/run-llama/llama_index/blob/78af3400ad485e15862c06f0c4972dc3067f880c/llama-index-packs/llama-index-packs-corrective-rag/llama_index/packs/corrective_rag/base.py#L11" target="_blank" rel="noreferrer">LlamaIndex</a></p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are a world-class search expert specializing in query rephrasing.</span></span>\n<span class="line"><span>Your task is to refine the provided query to ensure it is highly effective for retrieving relevant search results.</span></span>\n<span class="line"><span>Analyze the given input to grasp the core semantic intent or meaning.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Original Query: {{query}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective.</span></span>\n<span class="line"><span>Respond with the optimized query only.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Optimized query:</span></span></code></pre></div><h3 id="Template:-RAGQuerySimplifier" tabindex="-1">Template: RAGQuerySimplifier <a class="header-anchor" href="#Template:-RAGQuerySimplifier" aria-label="Permalink to &quot;Template: RAGQuerySimplifier {#Template:-RAGQuerySimplifier}&quot;">​</a></h3><ul><li><p>Description: For RAG applications (rephrase step), it rephrases the original query by stripping unnecessary details to improve the matched results. Placeholders: <code>query</code></p></li><li><p>Placeholders: <code>query</code></p></li><li><p>Word count: 267</p></li><li><p>Source: Adapted from <a href="https://python.langchain.com/docs/integrations/retrievers/re_phrase" target="_blank" rel="noreferrer">Langchain</a></p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are an assistant tasked with taking a natural language query from a user and converting it into a query for a vectorstore. </span></span>\n<span class="line"><span>In this process, you strip out information that is not relevant for the retrieval task.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Here is the user query: {{query}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Rephrased query:</span></span></code></pre></div>', 79);
const _hoisted_80 = [
  _hoisted_1
];
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _hoisted_80);
}
const RAG = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  RAG as default
};
