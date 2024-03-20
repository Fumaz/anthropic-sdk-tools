import {z, ZodSchema} from "zod";

export type AsyncFunction<Parameters extends ZodSchema = ZodSchema> = (args: z.infer<Parameters>) => Promise<any>;

export type AnthropicToolDefinition<Parameters extends ZodSchema = ZodSchema> = {
    tool_name: string;
    description: string;
    parameters: Parameters;
}

export type ToolDefinition<Parameters extends ZodSchema = ZodSchema> = {
    anthropic: AnthropicToolDefinition<Parameters>;
    function: AsyncFunction<Parameters>;
}

export type ResponseFormat = 'json' | 'xml' | 'yaml' | 'csv' | 'tsv' | 'html' | 'markdown' | 'latex';
