
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const chatRoute = require('./routes/chatRoute');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', chatRoute);

// Basic root route
app.get('/', (req, res) => {
    res.send('LLM Chat Backend is running!');
});

// Error handling middleware (optional, for more specific error handling)
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
