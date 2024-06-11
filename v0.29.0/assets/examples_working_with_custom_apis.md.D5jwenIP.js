import { _ as _export_sfc, c as createElementBlock, o as openBlock, a7 as createStaticVNode } from "./chunks/framework.BpxdJHRy.js";
const __pageData = JSON.parse('{"title":"Custom APIs","description":"","frontmatter":{},"headers":[],"relativePath":"examples/working_with_custom_apis.md","filePath":"examples/working_with_custom_apis.md","lastUpdated":null}');
const _sfc_main = { name: "examples/working_with_custom_apis.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<h1 id="Custom-APIs" tabindex="-1">Custom APIs <a class="header-anchor" href="#Custom-APIs" aria-label="Permalink to &quot;Custom APIs {#Custom-APIs}&quot;">​</a></h1><p>PromptingTools allows you to use any OpenAI-compatible API (eg, MistralAI), including a locally hosted one like the server from <code>llama.cpp</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PromptingTools</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PT </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PromptingTools</span></span></code></pre></div><h2 id="Using-MistralAI" tabindex="-1">Using MistralAI <a class="header-anchor" href="#Using-MistralAI" aria-label="Permalink to &quot;Using MistralAI {#Using-MistralAI}&quot;">​</a></h2><p>Mistral models have long been dominating the open-source space. They are now available via their API, so you can use them with PromptingTools.jl!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say hi!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;mistral-tiny&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># [ Info: Tokens: 114 @ Cost: $0.0 in 0.9 seconds</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># AIMessage(&quot;Hello there! I&#39;m here to help answer any questions you might have, or assist you with tasks to the best of my abilities. How can I be of service to you today? If you have a specific question, feel free to ask and I&#39;ll do my best to provide accurate and helpful information. If you&#39;re looking for general assistance, I can help you find resources or information on a variety of topics. Let me know how I can help.&quot;)</span></span></code></pre></div><p>It all just works, because we have registered the models in the <code>PromptingTools.MODEL_REGISTRY</code>! There are currently 4 models available: <code>mistral-tiny</code>, <code>mistral-small</code>, <code>mistral-medium</code>, <code>mistral-embed</code>.</p><p>Under the hood, we use a dedicated schema <code>MistralOpenAISchema</code> that leverages most of the OpenAI-specific code base, so you can always provide that explicitly as the first argument:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PT </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PromptingTools</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MistralOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say Hi!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;mistral-tiny&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, api_key</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;MISTRALAI_API_KEY&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p>As you can see, we can load your API key either from the ENV or via the Preferences.jl mechanism (see <code>?PREFERENCES</code> for more information).</p><h2 id="Using-other-OpenAI-compatible-APIs" tabindex="-1">Using other OpenAI-compatible APIs <a class="header-anchor" href="#Using-other-OpenAI-compatible-APIs" aria-label="Permalink to &quot;Using other OpenAI-compatible APIs {#Using-other-OpenAI-compatible-APIs}&quot;">​</a></h2><p>MistralAI are not the only ones who mimic the OpenAI API! There are many other exciting providers, eg, <a href="https://docs.perplexity.ai/" target="_blank" rel="noreferrer">Perplexity.ai</a>, <a href="https://app.fireworks.ai/" target="_blank" rel="noreferrer">Fireworks.ai</a>.</p><p>As long as they are compatible with the OpenAI API (eg, sending <code>messages</code> with <code>role</code> and <code>content</code> keys), you can use them with PromptingTools.jl by using <code>schema = CustomOpenAISchema()</code>:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Set your API key and the necessary base URL for the API</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">api_key </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;...&quot;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">provider_url </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;...&quot;</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # provider API URL</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">prompt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Say hi!&quot;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CustomOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), prompt; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&lt;some-model&gt;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, api_key, api_kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">provider_url))</span></span></code></pre></div><div class="tip custom-block github-alert"><p class="custom-block-title">If you register the model names with `PT.register_model!`, you won&#39;t have to keep providing the `schema` manually.</p><p></p></div><p>Note: At the moment, we only support <code>aigenerate</code> and <code>aiembed</code> functions.</p><h2 id="Using-llama.cpp-server" tabindex="-1">Using llama.cpp server <a class="header-anchor" href="#Using-llama.cpp-server" aria-label="Permalink to &quot;Using llama.cpp server {#Using-llama.cpp-server}&quot;">​</a></h2><p>In line with the above, you can also use the <a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md" target="_blank" rel="noreferrer"><code>llama.cpp</code> server</a>.</p><p>It is a bit more technically demanding because you need to &quot;compile&quot; <code>llama.cpp</code> first, but it will always have the latest models and it is quite fast (eg, faster than Ollama, which uses llama.cpp under the hood but has some extra overhead).</p><p>Start your server in a command line (<code>-m</code> refers to the model file, <code>-c</code> is the context length, <code>-ngl</code> is the number of layers to offload to GPU):</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">./server</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -m</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -c</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2048</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -ngl</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 99</span></span></code></pre></div><p>Then simply access it via PromptingTools:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CustomOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Count to 5 and say hi!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; api_kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;http://localhost:8080/v1&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span></code></pre></div><div class="tip custom-block github-alert"><p class="custom-block-title">If you register the model names with `PT.register_model!`, you won&#39;t have to keep providing the `schema` manually. It can be any `model` name, because the model is actually selected when you start the server in the terminal.</p><p></p></div><h2 id="Using-Databricks-Foundation-Models" tabindex="-1">Using Databricks Foundation Models <a class="header-anchor" href="#Using-Databricks-Foundation-Models" aria-label="Permalink to &quot;Using Databricks Foundation Models {#Using-Databricks-Foundation-Models}&quot;">​</a></h2><p>You can also use the Databricks Foundation Models API with PromptingTools.jl. It requires you to set ENV variables <code>DATABRICKS_API_KEY</code> (often referred to as &quot;DATABRICKS TOKEN&quot;) and <code>DATABRICKS_HOST</code>.</p><p>The long way to use it is:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DatabricksOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>\n<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;Say hi to the llama!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;databricks-llama-2-70b-chat&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    api_key </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DATABRICKS_API_KEY&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], api_kwargs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ENV</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DATABRICKS_HOST&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]))</span></span></code></pre></div><p>But you can also register the models you&#39;re hosting and use it as usual:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Quick registration of a model</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">register_model!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;databricks-llama-2-70b-chat&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        schema </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DatabricksOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">MODEL_ALIASES[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;dllama&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;databricks-llama-2-70b-chat&quot;</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # set alias to make your life easier</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Simply call:</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say hi to the llama!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;dllama&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Or even shorter</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ai</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say hi to the llama!&quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">dllama</span></span></code></pre></div><p>You can use <code>aiembed</code> as well.</p><p>Find more information <a href="https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html" target="_blank" rel="noreferrer">here</a>.</p><h2 id="Using-Together.ai" tabindex="-1">Using Together.ai <a class="header-anchor" href="#Using-Together.ai" aria-label="Permalink to &quot;Using Together.ai {#Using-Together.ai}&quot;">​</a></h2><p>You can also use the Together.ai API with PromptingTools.jl. It requires you to set ENV variable <code>TOGETHER_API_KEY</code>.</p><p>The corresponding schema is <code>TogetherOpenAISchema</code>, but we have registered one model for you, so you can use it as usual. Alias &quot;tmixtral&quot; (T for Together.ai and mixtral for the model name) is already set for you.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say hi&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;tmixtral&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">## [ Info: Tokens: 87 @ Cost: \\$0.0001 in 5.1 seconds</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">## AIMessage(&quot;Hello! I&#39;m here to help you. Is there something specific you&#39;d like to know or discuss? I can provide information on a wide range of topics, assist with tasks, and even engage in a friendly conversation. Let me know how I can best assist you today.&quot;)</span></span></code></pre></div><p>For embedding a text, use <code>aiembed</code>:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">aiembed</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TogetherOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;embed me&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;BAAI/bge-large-en-v1.5&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Note: You can register the model with <code>PT.register_model!</code> and use it as usual.</p><h2 id="Using-Fireworks.ai" tabindex="-1">Using Fireworks.ai <a class="header-anchor" href="#Using-Fireworks.ai" aria-label="Permalink to &quot;Using Fireworks.ai {#Using-Fireworks.ai}&quot;">​</a></h2><p>You can also use the Fireworks.ai API with PromptingTools.jl. It requires you to set ENV variable <code>FIREWORKS_API_KEY</code>.</p><p>The corresponding schema is <code>FireworksOpenAISchema</code>, but we have registered one model for you, so you can use it as usual. Alias &quot;fmixtral&quot; (F for Fireworks.ai and mixtral for the model name) is already set for you.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Say hi&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;fmixtral&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">## [ Info: Tokens: 78 @ Cost: \\$0.0001 in 0.9 seconds</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">## AIMessage(&quot;Hello! I&#39;m glad you&#39;re here. I&#39;m here to help answer any questions you have to the best of my ability. Is there something specific you&#39;d like to know or discuss? I can assist with a wide range of topics, so feel free to ask me anything!&quot;)</span></span></code></pre></div><p>In addition, at the time of writing (23rd Feb 2024), Fireworks is providing access to their new <em>function calling</em> model (fine-tuned Mixtral) <strong>for free</strong>.</p><p>Try it with <code>aiextract</code> for structured extraction (model is aliased as <code>firefunction</code>):</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;</span></span>\n<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Extract the food from the sentence. Extract any provided adjectives for the food as well.</span></span>\n<span class="line"></span>\n<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Example: &quot;I am eating a crunchy bread.&quot; -&gt; Food(&quot;bread&quot;, [&quot;crunchy&quot;])</span></span>\n<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Food</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    adjectives</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing,Vector{String}}</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">prompt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;I just ate a delicious and juicy apple.&quot;</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aiextract</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(prompt; return_type</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Food, model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;firefunction&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">content</span></span>\n<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Output: Food(&quot;apple&quot;, [&quot;delicious&quot;, &quot;juicy&quot;])</span></span></code></pre></div><p>For embedding a text, use <code>aiembed</code>:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">aiembed</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(PT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FireworksOpenAISchema</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;embed me&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;nomic-ai/nomic-embed-text-v1.5&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Note: You can register the model with <code>PT.register_model!</code> and use it as usual.</p>', 49);
const _hoisted_50 = [
  _hoisted_1
];
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _hoisted_50);
}
const working_with_custom_apis = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  working_with_custom_apis as default
};
