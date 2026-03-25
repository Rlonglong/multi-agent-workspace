
const express = require('express');
const { chatHandler } = require('../controllers/chatController');

const router = express.Router();

router.post('/chat', chatHandler);

module.exports = router;
