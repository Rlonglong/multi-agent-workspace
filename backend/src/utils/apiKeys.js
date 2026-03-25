
require('dotenv').config();

function getOpenAIApiKey() {
    return process.env.OPENAI_API_KEY;
}

module.exports = {
    getOpenAIApiKey,
};
