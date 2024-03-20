import {MessageParam} from "@anthropic-ai/sdk/resources/index";

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
