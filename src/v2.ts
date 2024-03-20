import Anthropic, {ClientOptions} from "@anthropic-ai/sdk";
import {ToolDefinition} from "./types.js";
import {XMLParser} from "fast-xml-parser";
import {Stream} from "@anthropic-ai/sdk/streaming";
import {zodToJsonSchema} from "zod-to-json-schema";

type AnthropicMessage = Anthropic.MessageParam & {
    hidden?: boolean;
}

type AnthropicStreamEvent = |
    (Anthropic.Messages.MessageStreamEvent |
        {
            type: 'tool_invoke';
            tool: ToolDefinition;
        } |
        {
            type: 'tool_result';
            tool: ToolDefinition;
            result: any;
        } |
        {
            type: 'tool_error';
            tool: ToolDefinition;
            error: any;
        } |
        {
            type: 'tool_call';
            tool: ToolDefinition;
            parameters: any;
        } |
        {
            type: 'tool_results',
            tools: Record<string, ToolDefinition>;
            results: Record<string, any>;
            messages: Anthropic.MessageParam[];
        }) & {
    tool?: ToolDefinition;
}

export class AnthropicClientV2 {
    private readonly client: Anthropic;
    private readonly verbose: boolean;

    constructor(options: ClientOptions & {
        verbose?: boolean;
    } = {}) {
        this.client = new Anthropic(options);
        this.verbose = options.verbose || false;
    }

    public async stream(
        body: Anthropic.MessageCreateParams & {
            tools?: ToolDefinition[];
            tools_model?: Anthropic.MessageCreateParams['model'];
        },
        options?: Anthropic.RequestOptions
    ): Promise<Stream<AnthropicStreamEvent>> {
        const tools = body.tools || [];
        const tools_model = body.tools_model || body.model;
        const system = this.buildSystemMessage(tools, body.system);
        const stop_sequences = tools.length > 0 ? [...(body.stop_sequences || []), '</function_calls>'] : [];

        if (this.verbose) {
            console.log('Sending message:', body);
        }

        delete body.tools;
        delete body.tools_model;

        const stream = await this.client.messages.create({
            ...body,
            system,
            messages: [...body.messages],
            stop_sequences,
            stream: true,
        }, options);

        let content: Anthropic.ContentBlock[] = [];
        let consumed = false;

        const classThis = this;

        async function* iterator(): AsyncGenerator<AnthropicStreamEvent, any, undefined> {
            if (consumed) {
                throw new Error('Cannot iterate over a consumed stream, use `.tee()` to split the stream.');
            }

            consumed = true;
            let done = false;

            try {
                for await (const chunk of stream) {
                    yield* classThis.processChunk(chunk, content);
                }

                // const messages = classThis.updateMessages(body.messages, content);

                if (content[0].text.includes('<function_calls>')) {
                    const functionCallsArray = classThis.extractFunctionCalls(content[0].text);

                    if (functionCallsArray.length > 0) {
                        const invokeFunctions = classThis.invokeFunctions(functionCallsArray, tools, tools_model, body, options);
                        let result: {
                            messages: Anthropic.MessageParam[];
                            results: Record<string, any>
                        };

                        for await (const chunk of invokeFunctions) {
                            if (chunk.type === 'tool_error') {
                                yield chunk;
                                break;
                            } else if (chunk.type === 'tool_result') {
                                yield chunk;
                            } else if (chunk.type === 'tool_results') {
                                yield chunk;

                                result = {
                                    messages: chunk.messages,
                                    results: chunk.results
                                }
                            }
                        }

                        yield * classThis.sendFinalResponse(body, body.messages, options);
                    }
                }

                done = true;
            } catch (e) {
                if (e instanceof Error && e.name === 'AbortError') {
                    return;
                }

                throw e;
            } finally {
                if (!done) {
                    stream.controller.abort();
                }
            }
        }

        return new Stream(iterator, stream.controller);
    }

    private buildSystemMessage(tools: ToolDefinition[], systemMessage: string | undefined): string {
        let system = '';

        if (tools.length > 0) {
            system += `In this environment you have access to a set of tools you can use to answer the user's questions.
You will be given the name and description of each tool, and you just need to call the tool by name. Another model will handle calling the tool with the correct parameters. You MUST NOT write any parameters yourself.

You may call one or more tools like this, the tools will be called in the order they are listed, but the result of each tool will NOT be passed to the next tool:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<context>$ALL_NECESSARY_CONTEXT_TO_SEND_TO_THE_OTHER_MODEL</context>
</invoke>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<context>$ALL_NECESSARY_CONTEXT_TO_SEND_TO_THE_OTHER_MODEL</context>
</invoke>
...
</function_calls>

Here are the tools available:
<tools>\n`;

            for (const tool of tools) {
                system += `<tool_description>\n`;
                system += `<name>${tool.anthropic.tool_name}</name>\n`;
                system += `<description>${tool.anthropic.description}</description>\n`;
                system += '</tool_description>\n';
            }

            system += `</tools>\n`;
        }

        return system + (systemMessage || '');
    }

    private* processChunk(chunk: Anthropic.Messages.MessageStreamEvent, content: Anthropic.ContentBlock[]): Generator<AnthropicStreamEvent, void, unknown> {
        switch (chunk.type) {
            case 'message_start':
                content = [];
                yield chunk;
                break;
            case 'message_delta':
            case 'message_stop':
                yield chunk;
                break;
            case 'content_block_start':
                content[chunk.index] = {
                    text: chunk.content_block.text,
                    type: 'text'
                };
                yield chunk;
                break;
            case 'content_block_delta':
                content[chunk.index] = {
                    text: content[chunk.index].text + chunk.delta.text,
                    type: 'text'
                };
                yield chunk;
                break;
            case 'content_block_stop':
                yield chunk;
                break;
        }
    }

    private updateMessages(messages: Anthropic.MessageParam[], content: Anthropic.ContentBlock[]): Anthropic.MessageParam[] {
        let finalText = '';
        for (const c of content) {
            finalText += c.text;
        }

        return [
            ...messages,
            {
                role: 'assistant',
                content: finalText
            }
        ];
    }

    private extractFunctionCalls(text: string): {
        tool_name: string;
        context: string;
    }[] {
        if (!text.includes('<function_calls>')) {
            return [];
        }

        let functionCalls = text.split('<function_calls>')[1];
        functionCalls = `<function_calls>\n${functionCalls}\n</function_calls>`;

        const parsed = new XMLParser().parse(functionCalls);
        const calls = parsed.function_calls;

        return Array.isArray(calls) ? calls.map(call => ({
            tool_name: call.invoke.tool_name,
            context: call.invoke.context
        })) : [{
            tool_name: calls.invoke.tool_name,
            context: calls.invoke.context
        }];
    }

    private async* invokeFunctions(
        functionCallsArray: {
            tool_name: string;
            context: string;
        }[],
        tools: ToolDefinition[],
        tools_model: Anthropic.MessageCreateParams['model'],
        body: Anthropic.MessageCreateParams,
        options?: Anthropic.RequestOptions
    ): AsyncGenerator<AnthropicStreamEvent> {
        const results: Record<string, any> = {};
        const messages: Anthropic.MessageParam[] = [];
        let functionResults = "<function_results>\n";

        for (const func of functionCallsArray) {
            const tool = tools.find(tool => tool.anthropic.tool_name === func.tool_name);

            if (!tool) {
                throw new Error(`Tool ${func} not found`);
            }

            yield {
                type: 'tool_invoke',
                tool: tool
            };

            for await (const result of this.invokeFunction(tool, tools_model, func.context, options)) {
                if (result.error) {
                    yield {
                        type: 'tool_error',
                        tool: tool,
                        error: result.error
                    };
                    functionResults = `<function_results>\n<error>\n${JSON.stringify(result.error)}\n</error>\n`;
                    break;
                }

                if (result.data !== undefined) {
                    functionResults += `<function_result>\n<tool_name>${tool.anthropic.tool_name}</tool_name>\n<result>${JSON.stringify(result.data)}</result>\n</function_result>\n`;
                    results[tool.anthropic.tool_name] = result.data;

                    yield {
                        type: 'tool_result',
                        tool: tool,
                        result: result.data
                    };
                }
            }
        }

        functionResults += "</function_results>";

        if (this.verbose) {
            console.log('Function results:', functionResults);
        }

        messages.push({
            role: 'user',
            content: functionResults
        });

        yield {
            type: 'tool_results',
            tools: Object.fromEntries(tools.map(tool => [tool.anthropic.tool_name, tool])),
            results: results,
            messages: messages
        };
    }

    private async* invokeFunction(
        tool: ToolDefinition,
        tools_model: Anthropic.MessageCreateParams['model'],
        context: string,
        options?: Anthropic.RequestOptions
    ): AsyncGenerator<{
        data?: any;
        error?: any
    }, void, unknown> {
        const zodParameters = tool.anthropic.parameters;
        const toolStream = await this.client.messages.create({
            stream: true,
            model: tools_model,
            system: `You are a tool caller. A model before you has chosen to call a tool. You will be given the name and description of the tool, along with the user's conversation. You will return JSON parameters to call the tool with. You are allowed to interpret what the user's intent is, and write the parameters accordingly. If you cannot interpret the user's intent, you will simply return <error>{{ASK FOR MORE INFORMATION}}</error>. The messages you see below are the messages between the previous model and the user.
The name of the tool you MUST call is ${tool.anthropic.tool_name}, and the description is ${tool.anthropic.description}. You must ALWAYS return JSON. Wrap the JSON in <parameters> and </parameters>. The JSON MUST be valid JSON. Do not return anything else.
You will return the parameters in JSON format following this JSON schema accurately. You must NOT return a JSON schema, but a valid JSON object that follows this schema:
${JSON.stringify(zodToJsonSchema(tool.anthropic.parameters))}`,
            messages: [{
                role: 'user',
                content: context
            }],
            stop_sequences: ['</parameters>'],
            max_tokens: 4000
        }, options);

        let jsonParametersContent: Anthropic.ContentBlock[] = [];

        for await (const chunk of toolStream) {
            yield {};

            switch (chunk.type) {
                case 'content_block_start':
                    jsonParametersContent[chunk.index] = {
                        text: chunk.content_block.text,
                        type: 'text'
                    };
                    break;
                case 'content_block_delta':
                    jsonParametersContent[chunk.index] = {
                        text: jsonParametersContent[chunk.index].text + chunk.delta.text,
                        type: 'text'
                    };
                    break;
            }
        }

        const finalText = jsonParametersContent.map(c => c.text).join('');

        let jsonParameters: any;

        try {
            jsonParameters = JSON.parse(finalText.replace(/<parameters>(.*)<\/parameters>/, '$1'));
        } catch (e: any) {
            yield {error: e};
            return;
        }

        const parsedParameters = zodParameters.safeParse(jsonParameters);

        if (!parsedParameters.success) {
            if (this.verbose) {
                console.error((parsedParameters as any).error);
            }

            yield {error: (parsedParameters as any).error};
            return;
        }

        try {
            const result = await tool.function(parsedParameters.data);
            yield {data: result};
        } catch (e: any) {
            yield {error: e};
        }
    }

    private async* sendFinalResponse(
        body: Anthropic.MessageCreateParams,
        messages: Anthropic.MessageParam[],
        options?: Anthropic.RequestOptions
    ): AsyncGenerator<Anthropic.Messages.MessageStreamEvent, void, unknown> {
        const finalResponse = await this.client.messages.create({
            ...body,
            system: `You have finished calling the tools. You MUST NOT call any more tools. Just tell the user the results of the tools you called.\n${body.system}`,
            stop_sequences: body.stop_sequences,
            messages: messages,
            stream: true,
        }, options);

        for await (const chunk of finalResponse) {
            yield chunk;
        }
    }
}
