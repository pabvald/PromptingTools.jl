## Langdock Agent API
#
# This file provides integration with Langdock's Agent API.
# Langdock is an enterprise AI platform that provides a unified API for accessing
# various AI models with features like GDPR compliance, knowledge folders, and custom agents.
#
# See more information at: https://docs.langdock.com/api-endpoints/agent/agent
#

## Rendering of conversation history for the Langdock API
"""
    render(schema::AbstractLangdockSchema,
        messages::Vector{<:AbstractMessage};
        conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
        kwargs...)

Builds a history of the conversation to provide the prompt to the Langdock API.

Langdock uses a message format with `role` and `content` keys, similar to OpenAI.
"""
function render(schema::AbstractLangdockSchema,
        messages::Vector{<:AbstractMessage};
        conversation_msgs::AbstractVector{<:AbstractMessage} = AbstractMessage[],
        kwargs...)
    ##
    @assert count(issystemmessage, messages)<=1 "AbstractLangdockSchema only supports at most 1 System message"

    # Filter out annotation messages before any processing
    messages = filter(!isabstractannotationmessage, messages)

    ## First pass: keep the message types but make the replacements provided in `kwargs`
    messages_replaced = render(
        NoSchema(), messages; conversation_msgs, kwargs...)

    ## Second pass: convert to the message-based schema
    conversation = Dict{String, Any}[]
    system_message = nothing

    for msg in messages_replaced
        if issystemmessage(msg)
            # Langdock agents handle system message via the assistant configuration
            # We'll prepend it to the first user message or store it separately
            system_message = msg.content
        elseif isabstractannotationmessage(msg)
            continue
        elseif isusermessage(msg)
            content = msg.content
            # If there's a system message and this is the first user message, prepend it
            if !isnothing(system_message)
                content = "System Instructions: $(system_message)\n\n$(content)"
                system_message = nothing
            end
            push!(conversation, Dict("role" => "user", "content" => content))
        elseif isaimessage(msg)
            push!(conversation, Dict("role" => "assistant", "content" => msg.content))
        elseif isusermessagewithimages(msg)
            # Langdock supports attachments but not inline images in the same way
            # For now, just use the text content
            content = msg.content
            if !isnothing(system_message)
                content = "System Instructions: $(system_message)\n\n$(content)"
                system_message = nothing
            end
            push!(conversation, Dict("role" => "user", "content" => content))
        end
        # Note: Ignores DataMessage or other types
    end

    ## Sense check
    @assert !isempty(conversation) "AbstractLangdockSchema requires at least 1 User message, ie, no `prompt` provided!"

    return (; conversation)
end

## Token usage extraction
"""
    extract_usage(schema::AbstractLangdockSchema, resp; model_id="", elapsed=0.0) -> TokenUsage

Extract token usage from a Langdock API response into a standardized TokenUsage struct.

Note: Langdock Agent API responses may not include detailed token counts.
When not available, usage fields default to 0.
"""
function extract_usage(::AbstractLangdockSchema, resp; model_id::String = "", elapsed::Float64 = 0.0)
    # Langdock Agent API response doesn't include token usage in the standard response
    # We estimate tokens based on content length as a fallback
    result = get(resp, :result, [])

    # Estimate output tokens from response content
    output_text = ""
    for item in result
        if haskey(item, :content)
            for content_item in item[:content]
                if haskey(content_item, :text)
                    output_text *= content_item[:text]
                end
            end
        end
    end

    # Rough estimation: ~4 characters per token for English text
    estimated_output_tokens = max(1, length(output_text) ÷ 4)

    # We can't estimate input tokens without access to the original messages
    # Cost calculation would need the actual pricing from Langdock
    TokenUsage(;
        input_tokens = 0,
        output_tokens = estimated_output_tokens,
        model_id,
        cost = 0.0,  # Langdock pricing varies by underlying model
        elapsed
    )
end

## Model-calling
"""
    langdock_api(
        prompt_schema::AbstractLangdockSchema,
        messages::Vector{<:AbstractDict{String, <:Any}} = Vector{Dict{String, Any}}();
        api_key::AbstractString = "",
        assistant_id::Union{Nothing, AbstractString} = nothing,
        assistant::Union{Nothing, AbstractDict} = nothing,
        max_steps::Int = 10,
        stream::Bool = false,
        http_kwargs::NamedTuple = NamedTuple(),
        url::String = "https://api.langdock.com/assistant/v1/chat/completions",
        kwargs...)

Simple wrapper for a call to Langdock Agent API.

# Keyword Arguments
- `prompt_schema`: Defines which prompt template should be applied.
- `messages`: A vector of message dictionaries to send to the agent.
- `api_key`: API key for authentication. Defaults to `LANGDOCK_API_KEY` environment variable.
- `assistant_id`: ID of an existing Langdock agent. Either this or `assistant` must be provided.
- `assistant`: Configuration for a temporary agent. Either this or `assistant_id` must be provided.
- `max_steps`: Maximum number of agent steps (1-20). Defaults to 10.
- `stream`: Whether to stream the response. Defaults to `false`.
- `http_kwargs`: Additional keyword arguments for the HTTP request.
- `url`: The URL of the Langdock API endpoint.
- `kwargs`: Additional keyword arguments passed to the API body.

# Returns
A named tuple with `response` (parsed JSON body) and `status` (HTTP status code).
"""
function langdock_api(
        prompt_schema::AbstractLangdockSchema,
        messages::Vector{<:AbstractDict{String, <:Any}} = Vector{Dict{String, Any}}();
        api_key::AbstractString = "",
        assistant_id::Union{Nothing, AbstractString} = nothing,
        assistant::Union{Nothing, AbstractDict} = nothing,
        max_steps::Int = 10,
        stream::Bool = false,
        http_kwargs::NamedTuple = NamedTuple(),
        url::String = "https://api.langdock.com/assistant/v1/chat/completions",
        kwargs...)

    ## Validate that exactly one of assistant_id or assistant is provided
    @assert xor(!isnothing(assistant_id), !isnothing(assistant)) "Exactly one of `assistant_id` or `assistant` must be provided"
    @assert 1 <= max_steps <= 20 "max_steps must be between 1 and 20"

    ## Use explicit api_key if provided, otherwise fall back to environment variable
    api_key = !isempty(api_key) ? api_key : LANGDOCK_API_KEY
    @assert !isempty(api_key) "LANGDOCK_API_KEY is not set. Please set it via ENV variable or pass it as `api_key` argument."

    ## Build request body
    body = Dict{Symbol, Any}(
        :messages => messages,
        :maxSteps => max_steps,
        :stream => stream
    )

    # Add either assistantId or assistant configuration
    if !isnothing(assistant_id)
        body[:assistantId] = assistant_id
    else
        body[:assistant] = assistant
    end

    # Add any additional kwargs
    for (k, v) in pairs(kwargs)
        body[k] = v
    end

    ## Build headers
    headers = [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]

    ## Make the request
    resp = HTTP.post(url, headers, JSON3.write(body); http_kwargs...)
    response_body = JSON3.read(resp.body)

    return (; response = response_body, status = resp.status)
end

# For testing
function langdock_api(prompt_schema::TestEchoLangdockSchema,
        messages::Vector{<:AbstractDict{String, <:Any}} = Vector{Dict{String, Any}}();
        api_key::AbstractString = "",
        assistant_id::Union{Nothing, AbstractString} = nothing,
        assistant::Union{Nothing, AbstractDict} = nothing,
        kwargs...)
    prompt_schema.model_id = !isnothing(assistant_id) ? assistant_id : "temp-assistant"
    prompt_schema.inputs = (; messages = copy(messages), assistant_id, assistant)
    return (; response = prompt_schema.response, status = prompt_schema.status)
end

## User-Facing API
"""
    aigenerate(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        verbose::Bool = true,
        api_key::String = "",
        model::String = "langdock",
        return_all::Bool = false, dry_run::Bool = false,
        conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
        http_kwargs::NamedTuple = NamedTuple(),
        api_kwargs::NamedTuple = NamedTuple(),
        kwargs...)

Generate an AI response based on a given prompt using the Langdock Agent API.

# Arguments
- `prompt_schema`: An optional object to specify which prompt template should be applied.
- `prompt`: Can be a string representing the prompt for the AI conversation, a `UserMessage`,
   a vector of `AbstractMessage`, or an `AITemplate`.
- `verbose`: A boolean indicating whether to print additional information.
- `api_key`: API key for the Langdock API. Defaults to `LANGDOCK_API_KEY` environment variable.
- `model`: Model identifier (used for registry lookup, not sent to API).
- `return_all`: If `true`, returns the entire conversation history, otherwise returns only the last message.
- `dry_run`: If `true`, skips sending the messages to the model (for debugging).
- `conversation`: An optional vector of `AbstractMessage` objects representing the conversation history.
- `http_kwargs`: Additional keyword arguments for the HTTP request.
- `api_kwargs`: Additional keyword arguments for the Langdock API. Must include either:
  - `assistant_id::String`: ID of an existing Langdock agent
  - `assistant::Dict`: Configuration for a temporary agent with keys:
    - `name` (required): Agent name (max 64 characters)
    - `instructions` (required): System instructions (max 16384 characters)
    - `model` (optional): Model ID (e.g., "gpt-4o", "claude-3-5-sonnet")
    - `temperature` (optional): Temperature (0-1)
    - `capabilities` (optional): Feature toggles (webSearch, dataAnalyst, etc.)
    - `knowledgeFolderIds` (optional): IDs of knowledge folders to use
- `kwargs`: Prompt variables to be used to fill the prompt/template.

# Returns
- `msg`: An `AIMessage` object representing the generated AI message.

# Example

Using a Langdock agent by ID:
```julia
ENV["LANGDOCK_API_KEY"] = "your-api-key"

msg = aigenerate("What is the weather in Berlin?";
    model="langdock",
    api_kwargs=(; assistant_id="your-agent-id"))
```

Using a temporary agent:
```julia
msg = aigenerate("Summarize the key points";
    model="langdock",
    api_kwargs=(; assistant=Dict(
        :name => "Summarizer",
        :instructions => "You are a helpful assistant that summarizes text concisely.",
        :model => "gpt-4o"
    )))
```
"""
function aigenerate(
        prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        verbose::Bool = true,
        api_key::String = "",
        model::String = "langdock",
        return_all::Bool = false, dry_run::Bool = false,
        conversation::AbstractVector{<:AbstractMessage} = AbstractMessage[],
        http_kwargs::NamedTuple = NamedTuple(),
        api_kwargs::NamedTuple = NamedTuple(),
        kwargs...)
    ##
    global MODEL_ALIASES

    ## Validate api_kwargs has required fields
    has_assistant_id = haskey(api_kwargs, :assistant_id) && !isnothing(api_kwargs.assistant_id)
    has_assistant = haskey(api_kwargs, :assistant) && !isnothing(api_kwargs.assistant)
    @assert xor(has_assistant_id, has_assistant) "api_kwargs must include exactly one of `assistant_id` or `assistant`"

    ## Find the unique ID for the model alias provided
    model_id = get(MODEL_ALIASES, model, model)

    conv_rendered = render(prompt_schema, prompt; conversation, kwargs...)

    if !dry_run
        ## Extract API kwargs
        assistant_id = get(api_kwargs, :assistant_id, nothing)
        assistant = get(api_kwargs, :assistant, nothing)
        max_steps = get(api_kwargs, :max_steps, 10)
        stream = get(api_kwargs, :stream, false)
        output = get(api_kwargs, :output, nothing)

        ## Build remaining api_kwargs (excluding the ones we handle explicitly)
        remaining_kwargs = Dict{Symbol, Any}()
        for (k, v) in pairs(api_kwargs)
            if k ∉ (:assistant_id, :assistant, :max_steps, :stream, :output)
                remaining_kwargs[k] = v
            end
        end
        if !isnothing(output)
            remaining_kwargs[:output] = output
        end

        time = @elapsed resp = langdock_api(
            prompt_schema, conv_rendered.conversation;
            api_key,
            assistant_id,
            assistant,
            max_steps,
            stream,
            http_kwargs,
            remaining_kwargs...)

        ## Extract content from response
        result = get(resp.response, :result, [])
        content_parts = String[]

        for item in result
            if get(item, :role, "") == "assistant" && haskey(item, :content)
                for content_item in item[:content]
                    if haskey(content_item, :text)
                        push!(content_parts, content_item[:text])
                    end
                end
            end
        end

        content = isempty(content_parts) ? "" : join(content_parts, "\n")

        # Extract structured output if present
        structured_output = get(resp.response, :output, nothing)

        # Extract usage
        usage = extract_usage(prompt_schema, resp.response; model_id, elapsed = time)

        ## Build metadata
        extras = Dict{Symbol, Any}()
        !isnothing(structured_output) && (extras[:structured_output] = structured_output)
        haskey(resp.response, :result) && (extras[:raw_result] = resp.response[:result])

        ## Build the message using unified builder
        msg = build_message(AIMessage, content, usage;
            status = Int(resp.status),
            finish_reason = "stop",
            extras)

        ## Reporting
        verbose && @info _report_stats(msg, model_id)
    else
        msg = nothing
    end

    ## Select what to return
    output = finalize_outputs(prompt,
        conv_rendered,
        msg;
        conversation,
        return_all,
        dry_run,
        kwargs...)
    return output
end

## Error stubs for unsupported operations
function aiembed(prompt_schema::AbstractLangdockSchema, doc_or_docs, postprocess::Function = identity;
        kwargs...)
    error("LangdockSchema (Agent API) does not support embeddings. Use LangdockOpenAISchema instead, e.g., `aiembed(LangdockOpenAISchema(), \"text\"; model=\"text-embedding-ada-002\")` or use model=\"langdock-emb\".")
end

function aiclassify(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        kwargs...)
    error("Langdock Agent API does not directly support aiclassify. Use aigenerate with structured output via api_kwargs=(; output=Dict(:type => \"enum\", :enum => [...])) instead.")
end

function aiextract(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        kwargs...)
    error("Langdock Agent API does not directly support aiextract. Use aigenerate with structured output via api_kwargs=(; output=Dict(:type => \"object\", :schema => ...)) instead.")
end

function aiscan(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        kwargs...)
    error("Langdock Agent API does not support image scanning directly. Upload images as attachments via api_kwargs=(; attachment_ids=[...]) instead.")
end

function aiimage(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        kwargs...)
    error("Langdock Agent API does not support image generation. Use a different schema like OpenAISchema.")
end

function aitools(prompt_schema::AbstractLangdockSchema, prompt::ALLOWED_PROMPT_TYPE;
        kwargs...)
    error("Langdock Agent API handles tools internally via agent capabilities. Configure tools in the agent settings or use api_kwargs=(; assistant=Dict(:capabilities => Dict(:webSearch => true))) instead.")
end
