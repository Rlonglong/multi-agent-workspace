
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');

async function callOpenAI(apiKey, systemPrompt, temperature, maxTokens, messages, res) {
    const openai = new OpenAI({ apiKey });

    const chatMessages = systemPrompt ? [{ role: 'system', content: systemPrompt }, ...messages] : messages;

    try {
        const stream = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo', // 可以根據需求調整模型
            messages: chatMessages,
            temperature,
            max_tokens: maxTokens,
            stream: true,
        });

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
                res.write(`data: ${JSON.stringify({ content })}\n\n`);
            }
        }
        res.write('event: end\n\n'); // 通知前端串流結束
    } catch (error) {
        console.error('Error calling OpenAI:', error);
        res.write(`event: error\ndata: ${JSON.stringify({ message: error.message })}\n\n`);
    }
}

async function callGemini(apiKey, systemPrompt, temperature, maxTokens, messages, res) {
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" }); // 可以根據需求調整模型

    // Gemini API expects history in a specific format
    const history = messages.map(msg => {
        if (msg.role === 'user') {
            return { role: 'user', parts: [{ text: msg.content }] };
        } else if (msg.role === 'assistant') {
            return { role: 'model', parts: [{ text: msg.content }] };
        }
        return null; // System prompt is handled separately or integrated into the first user message
    }).filter(Boolean);

    // If there's a system prompt, prepend it to the first user message or handle it as context
    // For simplicity, we'll try to integrate it into the initial prompt if present.
    // A more robust solution might involve sending it as a separate context if the API supports it
    // or ensuring the first message implicitly contains the system's instruction.
    // For now, we'll just pass the history as is and assume system prompt is handled by the user's initial setup.
    // If systemPrompt is critical, we might need to adjust the first message.
    let generationConfig = {
        temperature: temperature,
        maxOutputTokens: maxTokens,
    };

    // Start a new chat session
    const chat = model.startChat({
        history: history,
        generationConfig: generationConfig,
    });

    try {
        // The last message in 'messages' array is the current user's input
        const latestUserMessage = messages[messages.length - 1].content;
        const result = await chat.sendMessageStream(systemPrompt ? `${systemPrompt}
${latestUserMessage}` : latestUserMessage);

        for await (const chunk of result.stream) {
            const chunkText = chunk.text();
            if (chunkText) {
                res.write(`data: ${JSON.stringify({ content: chunkText })}\n\n`);
            }
        }
        res.write('event: end\n\n'); // 通知前端串流結束
    } catch (error) {
        console.error('Error calling Gemini:', error);
        res.write(`event: error\ndata: ${JSON.stringify({ message: error.message })}\n\n`);
    }
}

module.exports = {
    callOpenAI,
    callGemini,
};
