import Anthropic, {ClientOptions} from "@anthropic-ai/sdk";
import {MessageCreateParams, MessageParam} from "@anthropic-ai/sdk/resources/index";
import {RequestOptions} from "@anthropic-ai/sdk/core";
import {XMLBuilder, XMLParser} from "fast-xml-parser";
import {ZodSchema} from "zod";
import {zodToJsonSchema} from "zod-to-json-schema";
import MessageCreateParamsNonStreaming = MessageCreateParams.MessageCreateParamsNonStreaming;

type AsyncFunction = (args: any) => Promise<any>;

export type AnthropicToolDefinition = {
    tool_name: string;
    description: string;
    parameters: ZodSchema;
}

export type ToolDefinition = {
    anthropic: AnthropicToolDefinition;
    function: AsyncFunction;
}

export function getTextFromMessage(message: MessageParam) {
    if (typeof message.content === 'string') {
        return message.content || null;
    }

    const content = message.content;

    if (content.length === 0) {
        return null;
    }

    if (content[0].type === 'text') {
        return content[0].text || null;
    }

    return null;
}

export class AnthropicClient {
    private readonly client: Anthropic;
    private readonly tools: ToolDefinition[];

    constructor(options: ClientOptions & {
        tools?: ToolDefinition[];
    } = {}) {
        this.client = new Anthropic(options);
        this.tools = options.tools || [];
    }

    async invoke(body: MessageCreateParamsNonStreaming & {
        tools?: ToolDefinition[]
    }, options?: RequestOptions) {
        const tools = body.tools || [];
        const allTools = [...this.tools, ...tools];

        let system = allTools.length > 0 ? `In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>\n` : '';

        const builder = new XMLBuilder();
        const parser = new XMLParser();

        for (const tool of allTools) {
            system += `<tool_description>\n`;
            system += `<name>${tool.anthropic.tool_name}</name>\n`;
            system += `<description>${tool.anthropic.description}</description>\n`;
            system += `<parameters>\n`;

            const parametersAsJsonSchema = zodToJsonSchema(tool.anthropic.parameters);
            const parametersProperties = (parametersAsJsonSchema as any)?.properties;

            for (const key in parametersProperties) {
                const parameterData = parametersProperties[key] as any;

                system += `<parameter>\n`;
                system += `<name>${key}</name>\n`;
                system += `<type>${parameterData.type}</type>\n`;
                system += `<description>${parameterData.description}</description>\n`;

                if (parameterData.type === 'array' && parameterData.items) {
                    system += `<items>${builder.build(parameterData.items.properties)}</items>\n`;
                }

                if (parameterData.properties) {
                    system += `<properties>${builder.build(parameterData.properties)}</properties>\n`;
                }

                system += `</parameter>\n`;
            }

            system += `</parameters>\n`;
            system += `</tool_description>\n`;
        }

        if (allTools.length > 0) {
            system += `</tools>\n`;
            body.stop_sequences = [...(body.stop_sequences || []), '</function_calls>'];
        }

        body.system = system + (body.system || '');

        let response = await this.client.messages.create(body, options);

        body.messages = [
            ...body.messages,
            {
                role: 'assistant',
                content: response.content
            }
        ];

        if (getTextFromMessage(response) === null) {
            return {
                response,
                fullMessages: body.messages,
                filteredMessages: body.messages.filter(message => {
                    const text = getTextFromMessage(message);

                    if (text === null) {
                        return true;
                    }

                    return !text.includes('<function_calls>') && !text.includes('<function_results>');
                })
            };
        }

        let text = response.content[0].text;
        let first = true;
        const functions: {
            name: string;
            parameters: Record<string, string>
        }[] = [];

        while (text.includes('<function_calls>')) {
            let functionCalls = text.split('<function_calls>')[1];
            functionCalls = `<function_calls>\n${functionCalls}\n</function_calls>`;

            let parsed = parser.parse(functionCalls);

            const calls = parsed.function_calls;
            const invokeArray = Array.isArray(calls) ? calls : [calls];

            for (const invoke of invokeArray) {
                const toolName = invoke.invoke.tool_name;
                const parameters = invoke.invoke.parameters;
                const parameterValues: Record<string, string> = {};

                for (const parameterName in parameters) {
                    parameterValues[parameterName] = parameters[parameterName];
                }

                functions.push({
                    name: toolName,
                    parameters: parameterValues
                });
            }

            if (functions.length > 0) {
                let functionResults = "<function_results>\n";

                for (const func of functions) {
                    const tool = allTools.find(tool => tool.anthropic.tool_name === func.name);

                    if (!tool) {
                        throw new Error(`Tool ${func.name} not found`);
                    }

                    try {
                        const result = await tool.function(func.parameters);
                        functionResults += `<function_result>\n<tool_name>${func.name}</tool_name>\n<result>${result}</result>\n</function_result>\n`;
                    } catch (e: any) {
                        functionResults = `<function_results>\n<error>\n${e}\n</error>\n`;
                        break;
                    }
                }

                functionResults += "</function_results>";

                body.messages = [
                    ...body.messages,
                    ...(first ? [] : [{
                        role: 'assistant',
                        content: response.content[0].text,
                    } as const]),
                    {
                        role: 'user',
                        content: functionResults
                    }
                ];

                response = await this.client.messages.create(body, options);

                if (!response.content || response.content.length === 0 || !response.content[0].text) {
                    return {
                        response,
                        fullMessages: body.messages,
                        filteredMessages: body.messages.filter(message => {
                            const text = getTextFromMessage(message);

                            if (text === null) {
                                return true;
                            }

                            return !text.includes('<function_calls>') && !text.includes('<function_results>');
                        })
                    };
                }

                text = response.content[0].text;
            } else {
                if (!first) {
                    body.messages = [
                        ...body.messages,
                        {
                            role: 'assistant',
                            content: response.content[0].text,
                        }
                    ];
                }
            }

            first = false;
        }

        if (!first) {
            body.messages = [
                ...body.messages,
                {
                    role: 'assistant',
                    content: response.content[0].text,
                }
            ];
        }

        return {
            response,
            fullMessages: body.messages,
            filteredMessages: body.messages.filter(message => {
                const text = getTextFromMessage(message);

                if (text === null) {
                    return true;
                }

                return !text.includes('<function_calls>') && !text.includes('<function_results>');
            })
        }
    }
}
