using PromptingTools: TestEchoLangdockSchema, render, LangdockSchema
using PromptingTools: AIMessage, SystemMessage, AbstractMessage
using PromptingTools: UserMessage, UserMessageWithImages, DataMessage
using PromptingTools: langdock_api, extract_usage, TokenUsage

@testset "render-Langdock" begin
    schema = LangdockSchema()

    # Given a schema and a vector of messages with handlebar variables, it should replace the variables
    messages = [
        SystemMessage("Act as a helpful AI assistant"),
        UserMessage("Hello, my name is {{name}}")
    ]
    conversation = render(schema, messages; name = "John")
    # System message should be prepended to first user message
    @test length(conversation.conversation) == 1
    @test conversation.conversation[1]["role"] == "user"
    @test occursin("System Instructions:", conversation.conversation[1]["content"])
    @test occursin("Hello, my name is John", conversation.conversation[1]["content"])

    # AI message does NOT replace variables
    messages = [
        UserMessage("Hello"),
        AIMessage("Hi, my name is {{name}}")
    ]
    conversation = render(schema, messages; name = "John")
    @test length(conversation.conversation) == 2
    @test conversation.conversation[1]["role"] == "user"
    @test conversation.conversation[2]["role"] == "assistant"
    # AIMessage keeps handlebar variables
    @test conversation.conversation[2]["content"] == "Hi, my name is {{name}}"

    # Multiple messages
    messages = [
        UserMessage("Hello"),
        AIMessage("Hi there"),
        UserMessage("How are you?"),
        AIMessage("I'm doing well!")
    ]
    conversation = render(schema, messages)
    @test length(conversation.conversation) == 4
    @test conversation.conversation[1]["role"] == "user"
    @test conversation.conversation[2]["role"] == "assistant"
    @test conversation.conversation[3]["role"] == "user"
    @test conversation.conversation[4]["role"] == "assistant"

    # Given an empty vector of messages, it throws an error.
    messages = AbstractMessage[]
    @test_throws AssertionError render(schema, messages)

    # Given a schema and a vector of messages with a DataMessage, it should skip it
    messages = [
        UserMessage("Hello"),
        DataMessage(; content = ones(3, 3)),
        AIMessage("Hi there")
    ]
    conversation = render(schema, messages)
    @test length(conversation.conversation) == 2
    @test conversation.conversation[1]["role"] == "user"
    @test conversation.conversation[2]["role"] == "assistant"

    # Test with dry_run=true on ai* functions requires api_kwargs
    @test aigenerate(schema, messages; dry_run = true,
        api_kwargs = (; assistant_id = "test-id")) == nothing
    result = aigenerate(schema, messages; dry_run = true, return_all = true,
        api_kwargs = (; assistant_id = "test-id"))
    @test result.conversation == conversation.conversation
end

@testset "langdock_api-Langdock" begin
    # Test that exactly one of assistant_id or assistant must be provided
    schema = LangdockSchema()
    messages = [Dict("role" => "user", "content" => "Hello")]

    # Neither provided - should error
    @test_throws AssertionError langdock_api(schema, messages)

    # Both provided - should error
    @test_throws AssertionError langdock_api(schema, messages;
        assistant_id = "test-id",
        assistant = Dict(:name => "Test", :instructions => "Test"))

    # max_steps validation
    @test_throws AssertionError langdock_api(schema, messages;
        assistant_id = "test-id", max_steps = 0)
    @test_throws AssertionError langdock_api(schema, messages;
        assistant_id = "test-id", max_steps = 21)

    # Test echo schema
    echo_schema = TestEchoLangdockSchema()
    messages = [Dict("role" => "user", "content" => "Hello")]

    resp = langdock_api(echo_schema, messages; assistant_id = "test-assistant-123")
    @test echo_schema.model_id == "test-assistant-123"
    @test echo_schema.inputs.assistant_id == "test-assistant-123"
    @test echo_schema.inputs.messages == messages
    @test resp.status == 200
    @test haskey(resp.response, :result)

    # Test with temporary assistant config
    echo_schema2 = TestEchoLangdockSchema()
    assistant_config = Dict(:name => "TestBot", :instructions => "Be helpful")
    resp2 = langdock_api(echo_schema2, messages; assistant = assistant_config)
    @test echo_schema2.model_id == "temp-assistant"
    @test echo_schema2.inputs.assistant == assistant_config
end

@testset "extract_usage-Langdock" begin
    schema = LangdockSchema()

    # Test with response containing text
    resp = Dict(
        :result => [
            Dict(
                :id => "msg-123",
                :role => "assistant",
                :content => [
                    Dict(:type => "text", :text => "Hello, how can I help you today?")
                ]
            )
        ]
    )

    usage = extract_usage(schema, resp; model_id = "langdock", elapsed = 1.5)
    @test usage isa TokenUsage
    @test usage.model_id == "langdock"
    @test usage.elapsed == 1.5
    @test usage.output_tokens > 0  # Estimated from content length
    @test usage.cost == 0.0  # Langdock pricing varies

    # Test with empty response
    empty_resp = Dict(:result => [])
    usage_empty = extract_usage(schema, empty_resp; model_id = "langdock")
    @test usage_empty.output_tokens >= 1  # At least 1
end

@testset "aigenerate-Langdock" begin
    # Test with TestEchoLangdockSchema
    echo_schema = TestEchoLangdockSchema()

    # Test basic generation with assistant_id
    msg = aigenerate(echo_schema, "Hello!"; api_kwargs = (; assistant_id = "test-agent"))
    @test msg isa AIMessage
    @test msg.content == "Test response"

    # Test that assistant_id or assistant is required
    @test_throws AssertionError aigenerate(echo_schema, "Hello!")

    # Test with temporary assistant
    msg2 = aigenerate(echo_schema, "Hello!";
        api_kwargs = (;
            assistant = Dict(:name => "Bot", :instructions => "Be helpful", :model => "gpt-4o")))
    @test msg2 isa AIMessage

    # Test with conversation history
    messages = [
        UserMessage("First question"),
        AIMessage("First answer"),
        UserMessage("Follow-up")
    ]
    msg3 = aigenerate(echo_schema, messages; api_kwargs = (; assistant_id = "test-agent"))
    @test msg3 isa AIMessage

    # Test return_all
    result = aigenerate(echo_schema, "Hello!";
        return_all = true,
        api_kwargs = (; assistant_id = "test-agent"))
    @test result isa Vector
    @test any(m -> m isa AIMessage, result)
end

@testset "unsupported-operations-Langdock" begin
    schema = LangdockSchema()

    # Test that unsupported operations throw informative errors
    @test_throws ErrorException aiembed(schema, "test", identity)
    @test_throws ErrorException aiclassify(schema, "test")
    @test_throws ErrorException aiextract(schema, "test"; return_type = String)
    @test_throws ErrorException aiscan(schema, "test")
    @test_throws ErrorException aiimage(schema, "test")
    @test_throws ErrorException aitools(schema, "test")
end

@testset "LangdockOpenAISchema" begin
    # Test that LangdockOpenAISchema exists and is an AbstractOpenAISchema
    @test PT.LangdockOpenAISchema() isa PT.AbstractOpenAISchema

    # Test that the embedding model is registered
    @test haskey(PT.MODEL_REGISTRY, "langdock-emb")

    # Test the embedding alias
    @test haskey(PT.MODEL_ALIASES, "ldockemb")
    @test PT.MODEL_ALIASES["ldockemb"] == "langdock-emb"

    # Test ModelSpec for embeddings
    spec = PT.MODEL_REGISTRY["langdock-emb"]
    @test spec.name == "langdock-emb"
    @test spec.schema isa PT.LangdockOpenAISchema
end

@testset "model-registration-Langdock" begin
    # Test that the model is registered
    @test haskey(PT.MODEL_REGISTRY, "langdock")

    # Test the alias
    @test haskey(PT.MODEL_ALIASES, "ldock")
    @test PT.MODEL_ALIASES["ldock"] == "langdock"

    # Test ModelSpec
    spec = PT.MODEL_REGISTRY["langdock"]
    @test spec.name == "langdock"
    @test spec.schema isa LangdockSchema
end
