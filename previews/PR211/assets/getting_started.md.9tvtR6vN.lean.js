import { _ as _export_sfc, c as createElementBlock, a5 as createStaticVNode, o as openBlock } from "./chunks/framework.DWWQFVfn.js";
const __pageData = JSON.parse('{"title":"Getting Started","description":"","frontmatter":{},"headers":[],"relativePath":"getting_started.md","filePath":"getting_started.md","lastUpdated":null}');
const _sfc_main = { name: "getting_started.md" };
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _cache[0] || (_cache[0] = [
    createStaticVNode('<h1 id="Getting-Started" tabindex="-1">Getting Started <a class="header-anchor" href="#Getting-Started" aria-label="Permalink to &quot;Getting Started {#Getting-Started}&quot;">​</a></h1><h2 id="Prerequisites" tabindex="-1">Prerequisites <a class="header-anchor" href="#Prerequisites" aria-label="Permalink to &quot;Prerequisites {#Prerequisites}&quot;">​</a></h2><p><strong>OpenAI API key saved in the environment variable <code>OPENAI_API_KEY</code></strong></p><p>You will need to register with OpenAI and generate an API key:</p><ol><li><p>Create an account with <a href="https://platform.openai.com/signup" target="_blank" rel="noreferrer">OpenAI</a></p></li><li><p>Go to <a href="https://platform.openai.com/account/billing" target="_blank" rel="noreferrer">Account Billing</a> and buy some credits (prepayment, minimum 5 ). Your account must have credits for the API access to work.</p></li><li><p>Go to <a href="https://platform.openai.com/account/api-keys" target="_blank" rel="noreferrer">API Key page</a></p></li><li><p>Click on “Create new secret key”</p></li></ol><p>!!! Do not share it with anyone and do NOT save it to any files that get synced online.</p><p>Resources:</p><ul><li><p><a href="https://platform.openai.com/docs/quickstart?context=python" target="_blank" rel="noreferrer">OpenAI Documentation</a></p></li><li><p><a href="https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt" target="_blank" rel="noreferrer">Visual tutorial</a></p></li></ul><p>You will need to set this key as an environment variable before using PromptingTools.jl:</p><p>For a quick start, simply set it via <code>ENV[&quot;OPENAI_API_KEY&quot;] = &quot;your-api-key&quot;</code> Alternatively, you can:</p><ul><li><p>set it in the terminal before launching Julia: <code>export OPENAI_API_KEY = &lt;your key&gt;</code></p></li><li><p>set it in your <code>setup.jl</code> (make sure not to commit it to GitHub!)</p></li></ul><p>Make sure to start Julia from the same terminal window where you set the variable. Easy check in Julia, run <code>ENV[&quot;OPENAI_API_KEY&quot;]</code> and you should see your key!</p><p>For other options or more robust solutions, see the FAQ section.</p><p>Resources:</p><ul><li><a href="https://platform.openai.com/docs/quickstart?context=python" target="_blank" rel="noreferrer">OpenAI Guide</a></li></ul><h2 id="Installation" tabindex="-1">Installation <a class="header-anchor" href="#Installation" aria-label="Permalink to &quot;Installation {#Installation}&quot;">​</a></h2><p>PromptingTools can be installed using the following commands:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>\n<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;PromptingTools.jl&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Throughout the rest of this tutorial, we will assume that you have installed the PromptingTools package and have already typed <code>using PromptingTools</code> to bring all of the relevant variables into your current namespace.</p><h2 id="Quick-Start-with-@ai_str" tabindex="-1">Quick Start with <code>@ai_str</code> <a class="header-anchor" href="#Quick-Start-with-@ai_str" aria-label="Permalink to &quot;Quick Start with `@ai_str` {#Quick-Start-with-@ai_str}&quot;">​</a></h2><p>The easiest start is the <code>@ai_str</code> macro. Simply type <code>ai&quot;your prompt&quot;</code> and you will get a response from the default model (GPT-3.5 Turbo).</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ai</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;What is the capital of France?&quot;</span></span></code></pre></div><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ Info: Tokens: 31 @ Cost: $0.0 in 1.5 seconds --&gt; Be in control of your spending! </span></span>\n<span class="line"><span>AIMessage(&quot;The capital of France is Paris.&quot;)</span></span></code></pre></div><p>Returned object is a light wrapper with generated message in field <code>:content</code> (eg, <code>ans.content</code>) for additional downstream processing.</p><p>If you want to reply to the previous message, or simply continue the conversation, use <code>@ai!_str</code> (notice the bang <code>!</code>):</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ai!</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;And what is the population of it?&quot;</span></span></code></pre></div><p>You can easily inject any variables with string interpolation:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">country </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Spain&quot;</span></span>\n<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ai</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;What is the capital of </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\$</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">(country)?&quot;</span></span></code></pre></div><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ Info: Tokens: 32 @ Cost: $0.0001 in 0.5 seconds</span></span>\n<span class="line"><span>AIMessage(&quot;The capital of Spain is Madrid.&quot;)</span></span></code></pre></div><p>Pro tip: Use after-string-flags to select the model to be called, eg, <code>ai&quot;What is the capital of France?&quot;gpt4</code> (use <code>gpt4t</code> for the new GPT-4 Turbo model). Great for those extra hard questions!</p><h2 id="Using-aigenerate-with-placeholders" tabindex="-1">Using <code>aigenerate</code> with placeholders <a class="header-anchor" href="#Using-aigenerate-with-placeholders" aria-label="Permalink to &quot;Using `aigenerate` with placeholders {#Using-aigenerate-with-placeholders}&quot;">​</a></h2><p>For more complex prompt templates, you can use handlebars-style templating and provide variables as keyword arguments:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">msg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> aigenerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;What is the capital of {{country}}? Is the population larger than {{population}}?&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, country</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Spain&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, population</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;1M&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[ Info: Tokens: 74 @ Cost: $0.0001 in 1.3 seconds</span></span>\n<span class="line"><span>AIMessage(&quot;The capital of Spain is Madrid. And yes, the population of Madrid is larger than 1 million. As of 2020, the estimated population of Madrid is around 3.3 million people.&quot;)</span></span></code></pre></div><p>Pro tip: Use <code>asyncmap</code> to run multiple AI-powered tasks concurrently.</p><p>Pro tip: If you use slow models (like GPT-4), you can use the asynchronous version of <code>@ai_str</code> -&gt; <code>@aai_str</code> to avoid blocking the REPL, eg, <code>aai&quot;Say hi but slowly!&quot;gpt4</code> (similarly <code>@ai!_str</code> -&gt; <code>@aai!_str</code> for multi-turn conversations).</p><p>For more practical examples, see the <a href="/PromptingTools.jl/previews/PR211/examples/readme_examples#Various-Examples">Various Examples</a> section.</p>', 37)
  ]));
}
const getting_started = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  getting_started as default
};
