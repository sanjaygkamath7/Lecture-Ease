const mongoose = require('mongoose');

const TranscriptSchema = new mongoose.Schema({
  filename: { type: String, required: true },
  transcript: { type: String, required: true },
  summary: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Transcript', TranscriptSchema);
