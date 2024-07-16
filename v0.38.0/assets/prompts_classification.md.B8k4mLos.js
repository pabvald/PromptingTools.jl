import { _ as _export_sfc, c as createElementBlock, o as openBlock, a7 as createStaticVNode } from "./chunks/framework.DWukBw7n.js";
const __pageData = JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"prompts/classification.md","filePath":"prompts/classification.md","lastUpdated":null}');
const _sfc_main = { name: "prompts/classification.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode('<p>The following file is auto-generated from the <code>templates</code> folder. For any changes, please modify the source files in the <code>templates</code> folder.</p><p>To use these templates in <code>aigenerate</code>, simply provide the template name as a symbol, eg, <code>aigenerate(:MyTemplate; placeholder1 = value1)</code></p><h2 id="Classification-Templates" tabindex="-1">Classification Templates <a class="header-anchor" href="#Classification-Templates" aria-label="Permalink to &quot;Classification Templates {#Classification-Templates}&quot;">​</a></h2><h3 id="Template:-InputClassifier" tabindex="-1">Template: InputClassifier <a class="header-anchor" href="#Template:-InputClassifier" aria-label="Permalink to &quot;Template: InputClassifier {#Template:-InputClassifier}&quot;">​</a></h3><ul><li><p>Description: For classification tasks and routing of queries with aiclassify. It expects a list of choices to be provided (starting with their IDs) and will pick one that best describes the user input. Placeholders: <code>input</code>, <code>choices</code></p></li><li><p>Placeholders: <code>choices</code>, <code>input</code></p></li><li><p>Word count: 366</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are a world-class classification specialist. </span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to select the most appropriate label from the given choices for the given user input.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Available Choices:**</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{choices}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions:**</span></span>\n<span class="line"><span>- You must respond in one word. </span></span>\n<span class="line"><span>- You must respond only with the label ID (e.g., &quot;1&quot;, &quot;2&quot;, ...) that best fits the input.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>User Input: {{input}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Label:</span></span></code></pre></div><h3 id="Template:-JudgeIsItTrue" tabindex="-1">Template: JudgeIsItTrue <a class="header-anchor" href="#Template:-JudgeIsItTrue" aria-label="Permalink to &quot;Template: JudgeIsItTrue {#Template:-JudgeIsItTrue}&quot;">​</a></h3><ul><li><p>Description: LLM-based classification whether the provided statement is true/false/unknown. Statement is provided via <code>it</code> placeholder.</p></li><li><p>Placeholders: <code>it</code></p></li><li><p>Word count: 151</p></li><li><p>Source:</p></li><li><p>Version: 1.1</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are an impartial AI judge evaluating whether the provided statement is &quot;true&quot; or &quot;false&quot;. Answer &quot;unknown&quot; if you cannot decide.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span># Statement</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>{{it}}</span></span></code></pre></div><h3 id="Template:-QuestionRouter" tabindex="-1">Template: QuestionRouter <a class="header-anchor" href="#Template:-QuestionRouter" aria-label="Permalink to &quot;Template: QuestionRouter {#Template:-QuestionRouter}&quot;">​</a></h3><ul><li><p>Description: For question routing tasks. It expects a list of choices to be provided (starting with their IDs), and will pick one that best describes the user input. Always make sure to provide an option for <code>Other</code>. Placeholders: <code>question</code>, <code>choices</code></p></li><li><p>Placeholders: <code>choices</code>, <code>question</code></p></li><li><p>Word count: 754</p></li><li><p>Source:</p></li><li><p>Version: 1.0</p></li></ul><p><strong>System Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>You are a highly capable question router and classification specialist. </span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Your task is to select the most appropriate category from the given endpoint choices to route the user&#39;s question or statement. If none of the provided categories are suitable, you should select the option indicating no appropriate category.</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Available Endpoint Choices:**</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span>{{choices}}</span></span>\n<span class="line"><span>---</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>**Instructions:**</span></span>\n<span class="line"><span>- You must respond in one word only. </span></span>\n<span class="line"><span>- You must respond with just the number (e.g., &quot;1&quot;, &quot;2&quot;, ...) of the endpoint choice that the input should be routed to based on the category it best fits.</span></span>\n<span class="line"><span>- If none of the endpoint categories are appropriate for the given input, select the choice indicating that no category fits.</span></span></code></pre></div><p><strong>User Prompt:</strong></p><div class="language-plaintext vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">plaintext</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>User Question: {{question}}</span></span>\n<span class="line"><span></span></span>\n<span class="line"><span>Endpoint Choice:</span></span></code></pre></div>', 21);
const _hoisted_22 = [
  _hoisted_1
];
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, _hoisted_22);
}
const classification = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  classification as default
};
