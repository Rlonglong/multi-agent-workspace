
const { callOpenAI, callGemini } = require('../services/llmService');
const { getOpenAIApiKey } = require('../utils/apiKeys');

async function chatHandler(req, res) {
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
    });

    const { model, systemPrompt, temperature, maxTokens, messages, geminiApiKey } = req.body;

    try {
        if (model === 'openai') {
            const openaiApiKey = getOpenAIApiKey();
            if (!openaiApiKey) {
                res.write(`event: error\ndata: {"message": "OpenAI API Key not configured on server."}\n\n`);
                return res.end();
            }
            await callOpenAI(openaiApiKey, systemPrompt, temperature, maxTokens, messages, res);
        } else if (model === 'gemini') {
            if (!geminiApiKey) {
                res.write(`event: error\ndata: {"message": "Gemini API Key is required."}\n\n`);
                return res.end();
            }
            await callGemini(geminiApiKey, systemPrompt, temperature, maxTokens, messages, res);
        } else {
            res.write(`event: error\ndata: {"message": "Unsupported LLM model."}\n\n`);
            return res.end();
        }
    } catch (error) {
        console.error('Error in chatHandler:', error);
        res.write(`event: error\ndata: {"message": "${error.message || "An unexpected error occurred."}"}\n\n`);
    } finally {
        res.end();
    }
}

module.exports = {
    chatHandler,
};
