require("dotenv").config();
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const path = require("path");
const cors = require("cors");
const fs = require("fs"); 

const app = express();
const PORT = process.env.PORT || 5000;
const mongoose = require("mongoose");
const authRoutes = require("./routes/auth");
app.use(express.json());
// app.use(cors());
app.use(cors({
  origin: "https://lecturease.tech",
  credentials: true
}));


mongoose
  .connect(
    "mongodb+srv://sanjaykamath7:qs0oEBdwmJMyNq23@cluster0.0p5hw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    // { useNewUrlParser: true, useUnifiedTopology: true, serverSelectionTimeoutMS: 30000 }
    {serverSelectionTimeoutMS: 30000 }
  )
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("Error connecting to MongoDB:", err));

app.use("/api", authRoutes);


// const uploadPath =
//   "D:/Mini-Project/backend/uploads"; 
const uploadPath = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadPath)) {
  fs.mkdirSync(uploadPath, { recursive: true });
}

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadPath),
  filename: (req, file, cb) => {
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `audio-${uniqueSuffix}${path.extname(file.originalname)}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "audio/mpeg",
      "audio/mp3",
      "audio/wav",
      "audio/x-wav",
      "audio/mp4",
      "audio/aac",
    ];
    if (allowedMimeTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Invalid file type. Got: ${
            file.mimetype
          }. Accepted types: ${allowedMimeTypes.join(", ")}`
        )
      );
    }
  },
}).single("audio");

// Uploads file contents directly to AssemblyAI
async function uploadFileToAssemblyAI(filePath) {
  try {
    const audioFile = await fs.promises.readFile(filePath); // Read the file as binary data
    const response = await axios.post(
      "https://api.assemblyai.com/v2/upload",
      audioFile,
      {
        headers: {
          authorization: process.env.ASSEMBLYAI_API_KEY,
          "content-type": "application/octet-stream",
          "transfer-encoding": "chunked",
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      }
    );
    console.log("AssemblyAI upload successful:", response.data.upload_url);
    return response.data.upload_url;
  } catch (error) {
    console.error(
      "Error uploading file to AssemblyAI:",
      error.response?.data || error.message
    );
    throw new Error("Failed to upload file to AssemblyAI");
  }
}

//Wait for transcription to complete
async function getTranscription(transcriptId) {
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  for (let attempt = 0; attempt < 60; attempt++) {
    // Retry up to 60 times
    try {
      const response = await axios.get(
        `https://api.assemblyai.com/v2/transcript/${transcriptId}`,
        { headers: { authorization: process.env.ASSEMBLYAI_API_KEY } }
      );
      if (response.data.status === "completed") {
        return response.data;
      }
      if (response.data.status === "error") {
        throw new Error(response.data.error);
      }
      await delay(2000); // Wait for 2 seconds before retrying
    } catch (error) {
      console.error(
        `Error waiting for transcription (attempt ${attempt + 1}):`,
        error.message
      );
      throw new Error("Failed to fetch transcription status");
    }
  }
  throw new Error("Transcription timed out.");
}

// Endpoint: Upload and summarize audio
const Transcript = require("./models/Transcript"); // Import the model

app.post("/api/upload-and-summarize", async (req, res) => {
  try {
    // Handle file upload
    await new Promise((resolve, reject) => {
      upload(req, res, (err) => (err ? reject(err) : resolve()));
    });
    if (!req.file) throw new Error("No audio file received.");
    const audioFilePath = path.join(uploadPath, req.file.filename);
    console.log("File uploaded locally at:", audioFilePath);

    // Upload file to AssemblyAI
    const uploadUrl = await uploadFileToAssemblyAI(audioFilePath);

    // Request transcription
    const transcriptResponse = await axios.post(
      "https://api.assemblyai.com/v2/transcript",
      { audio_url: uploadUrl, language_detection: true },
      { headers: { authorization: process.env.ASSEMBLYAI_API_KEY } }
    );

    // Wait for transcription to complete
    const transcriptionResult = await getTranscription(
      transcriptResponse.data.id
    );
    const transcriptText = transcriptionResult.text;
    if (!transcriptText) throw new Error("No transcript text generated.");

    // Generate summary using HuggingFace
    const summaryResponse = await axios.post(
      "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
      {
        inputs: transcriptText.slice(0, 1024), // Limit input size for HuggingFace
        parameters: { max_length: 150, min_length: 30, do_sample: false },
      },
      {
        headers: { Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}` },
        timeout: 30000,
      }
    );
    const summary =
      summaryResponse.data[0]?.summary_text || "No summary generated.";

    console.log("Summary generated:", summary);

   
    const newTranscript = new Transcript({
      filename: req.file.filename,
      transcript: transcriptText,
      summary,
    });

    await newTranscript.save();
    console.log("Transcript and summary stored successfully in the database.");

    res.json({ success: true, summary, transcriptText });
  } catch (error) {
    console.error("Error processing audio file:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get("/health", (req, res) =>
  res.json({ status: "healthy", timestamp: new Date().toISOString() })
);

// Start server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
