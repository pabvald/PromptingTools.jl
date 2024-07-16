import { _ as _export_sfc, c as createElementBlock, o as openBlock, a7 as createStaticVNode } from "./chunks/framework.BK5tW3ua.js";
const __pageData = JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"prompts/agents.md","filePath":"prompts/agents.md","lastUpdated":null}');
const _sfc_main = { name: "prompts/agents.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<p>The following file is auto-generated from the <code>templates</code> folder. For any changes, please modify the source files in the <code>templates</code> folder.</p><p>To use these templates in <code>aigenerate</code>, simply provide the template name as a symbol, eg, <code>aigenerate(:MyTemplate; placeholder1 = value1)</code></p><h2 id="Code-Fixing-Templates" tabindex="-1">Code-Fixing Templates <a class="header-anchor" href="#Code-Fixing-Templates" aria-label="Permalink to &quot;Code-Fixing Templates {#Code-Fixing-Templates}&quot;">​</a></h2><h3 id="Template:-CodeFixerRCI" tabindex="-1">Template: CodeFixerRCI <a class="header-anchor" href="#Template:-CodeFixerRCI" aria-label="Permalink to &quot;Template: CodeFixerRCI {#Template:-CodeFixerRCI}&quot;">​</a></h3><ul><li><p>Description: This template is meant to be used with <code>AICodeFixer</code>. It loosely follows the <a href="https://arxiv.org/pdf/2303.17491.pdf" target="_blank" rel="noreferrer">Recursive Critique and Improvement paper</a> with two steps Critique and Improve based on <code>feedback</code>. Placeholders: <code>feedback</code></p></li><li><p>Placeholders: <code>feedback</code></p></li><li><p>Word count: 2487</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span></span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Ignore all previous instructions. </span></span>\n<span class="line"><span>Your goal is to satisfy the user&#39;s request by using several rounds of self-reflection (Critique step) and improvement of the previously provided solution (Improve step).</span></span>\n<span class="line"><span>Always enclose the Julia code in triple backticks code fence (```julia\\n ... \\n```).</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>1. **Recall Past Critique:**</span></span>\n<span class="line"><span>- Summarize past critiques to refresh your memory (use inline quotes to highlight the few characters of the code that caused the mistakes). It must not be repeated.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>2. **Critique Step Instructions:** </span></span>\n<span class="line"><span>- Read the user request word-by-word. Does the code implementation follow the request to the letter? Let&#39;s think step by step.</span></span>\n<span class="line"><span>- Review the provided feedback in detail.</span></span>\n<span class="line"><span>- Provide 2-3 bullet points of criticism for the code. Each bullet point must refer to a different type of error or issue.</span></span>\n<span class="line"><span>    - If there are any errors, explain why and what needs to be changed to FIX THEM! Be specific. </span></span>\n<span class="line"><span>    - If an error repeats or critique repeats, the previous issue was not addressed. YOU MUST SUGGEST A DIFFERENT IMPROVEMENT THAN BEFORE.</span></span>\n<span class="line"><span>    - If there are no errors, identify and list specific issues or areas for improvement to write more idiomatic Julia code.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>3. **Improve Step Instructions:** </span></span>\n<span class="line"><span>- Specify what you&#39;ll change to address the above critique.</span></span>\n<span class="line"><span>- Provide the revised code reflecting your suggested improvements. Always repeat the function definition, as only the Julia code in the last message will be evaluated.</span></span>\n<span class="line"><span>- Ensure the new version of the code resolves the problems while fulfilling the original task. Ensure it has the same function name.</span></span>\n<span class="line"><span>- Write 2-3 correct and helpful unit tests for the function requested by the user (organize in `@testset &quot;name&quot; begin ... end` block, use `@test` macro).</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>3. **Response Format:**</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>### Past Critique</span></span>\n<span class="line"><span>&lt;brief bullet points on past critique&gt;</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>### Critique</span></span>\n<span class="line"><span>&lt;list of issues as bullet points pinpointing the mistakes in the code (use inline quotes)&gt;</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>### Improve</span></span>\n<span class="line"><span>&lt;list of improvements as bullet points with a clear outline of a solution (use inline quotes)&gt;</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>```julia</span></span>\n<span class="line"><span>&lt;provide improved code&gt;</span></span>\n<span class="line"><span>```</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Be concise and focused in all steps.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>### Feedback from the User</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{feedback}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>I believe in you. You can actually do it, so do it ffs. Avoid shortcuts or placing comments instead of code. I also need code, actual working Julia code.</span></span>\n<span class="line"><span>What are your Critique and Improve steps?</span></span>\n<span class="line"><span>  ### Feedback from the User</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{feedback}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Based on your past critique and the latest feedback, what are your Critique and Improve steps?</span></span></code></pre></div><h3 id="Template:-CodeFixerShort" tabindex="-1">Template: CodeFixerShort <a class="header-anchor" href="#Template:-CodeFixerShort" aria-label="Permalink to &quot;Template: CodeFixerShort {#Template:-CodeFixerShort}&quot;">​</a></h3><ul><li><p>Description: This template is meant to be used with <code>AICodeFixer</code> to ask for code improvements based on <code>feedback</code>. It uses the same message for both the introduction of the new task and for the iterations. Placeholders: <code>feedback</code></p></li><li><p>Placeholders: <code>feedback</code></p></li><li><p>Word count: 786</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span></span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span></span></span>\n<span class="line"><span>The above Julia code has been executed with the following results:</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>```plaintext</span></span>\n<span class="line"><span>{{feedback}}</span></span>\n<span class="line"><span>```</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>0. Read the user request word-by-word. Does the code implementation follow the request to the letter? Let&#39;s think step by step.</span></span>\n<span class="line"><span>1. Review the execution results in detail and, if there is an error, explain why it happened.</span></span>\n<span class="line"><span>2. Suggest improvements to the code. Be EXTREMELY SPECIFIC. Think step-by-step and break it down.</span></span>\n<span class="line"><span>3. Write an improved implementation based on your reflection.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>All code must be enclosed in triple backticks code fence (```julia\\n ... \\n```) and included in one message to be re-evaluated.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>I believe in you. Take a deep breath. You can actually do it, so do it ffs. Avoid shortcuts or placing comments instead of code. I also need code, actual working Julia code.</span></span></code></pre></div><h3 id="Template:-CodeFixerTiny" tabindex="-1">Template: CodeFixerTiny <a class="header-anchor" href="#Template:-CodeFixerTiny" aria-label="Permalink to &quot;Template: CodeFixerTiny {#Template:-CodeFixerTiny}&quot;">​</a></h3><ul><li><p>Description: This tiniest template to use with <code>AICodeFixer</code>. Iteratively asks to improve the code based on provided <code>feedback</code>. Placeholders: <code>feedback</code></p></li><li><p>Placeholders: <code>feedback</code></p></li><li><p>Word count: 210</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span></span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>### Execution Results</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>```plaintext</span></span>\n<span class="line"><span>{{feedback}}</span></span>\n<span class="line"><span>```</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Take a deep break. Think step-by-step and fix the above errors. I believe in you. You can do it! I also need code, actual working Julia code, no shortcuts.</span></span></code></pre></div><h2 id="Feedback-Templates" tabindex="-1">Feedback Templates <a class="header-anchor" href="#Feedback-Templates" aria-label="Permalink to &quot;Feedback Templates {#Feedback-Templates}&quot;">​</a></h2><h3 id="Template:-FeedbackFromEvaluator" tabindex="-1">Template: FeedbackFromEvaluator <a class="header-anchor" href="#Template:-FeedbackFromEvaluator" aria-label="Permalink to &quot;Template: FeedbackFromEvaluator {#Template:-FeedbackFromEvaluator}&quot;">​</a></h3><ul><li><p>Description: Simple user message with &quot;Feedback from Evaluator&quot;. Placeholders: <code>feedback</code></p></li><li><p>Placeholders: <code>feedback</code></p></li><li><p>Word count: 41</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span></span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>### Feedback from Evaluator</span></span>\n<span class="line"><span>{{feedback}}</span></span></code></pre></div>', 28);
const _hoisted_29 = [
  _hoisted_1
];
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _hoisted_29);
}
const agents = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  agents as default
};
