import React, { useState, useEffect, useCallback } from "react";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { Link } from 'react-router-dom';
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import axios from "axios";
import "./FileUpload.css";
import "./Home.css";

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const ALLOWED_VIDEO_TYPES = ["video/mp4", "video/webm", "video/ogg"];

// const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

const FileUpload = () => {
  const [ffmpeg] = useState(() => new FFmpeg());
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [ffmpegLoaded, setFfmpegLoaded] = useState(false);
  const [conversionProgress, setConversionProgress] = useState(0);
  const [showChatbot, setShowChatbot] = useState(false);
  const [chatMessage, setChatMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [processingStatus, setProcessingStatus] = useState("");
  const [lectureSummary, setLectureSummary] = useState("");
  const [chatId] = useState(() => Math.random().toString(36).substring(7));

  // Load FFmpeg WebAssembly library
  useEffect(() => {
    const loadFFmpeg = async () => {
      try {
        await ffmpeg.load({
          coreURL: await toBlobURL(
            "https://unpkg.com/@ffmpeg/core@0.12.4/dist/umd/ffmpeg-core.js",
            "text/javascript"
          ),
          wasmURL: await toBlobURL(
            "https://unpkg.com/@ffmpeg/core@0.12.4/dist/umd/ffmpeg-core.wasm",
            "application/wasm"
          ),
        });
        setFfmpegLoaded(true);
      } catch (error) {
        console.error("Error loading FFmpeg:", error);
        setErrorMessage("Error loading audio processor. Please try again.");
      }
    };
    loadFFmpeg();
  }, [ffmpeg]);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const validateFile = useCallback((file) => {
    if (!file) {
      throw new Error("Please select a file to upload.");
    }

    if (!ALLOWED_VIDEO_TYPES.includes(file.type)) {
      throw new Error("Please select a valid video file (MP4, WebM, or OGG).");
    }

    if (file.size > MAX_FILE_SIZE) {
      throw new Error("File size exceeds 100MB limit.");
    }
  }, []);

  const handleFileChange = useCallback(
    (event) => {
      const selectedFile = event.target.files[0];
      try {
        validateFile(selectedFile);
        setFile(selectedFile);
        setErrorMessage("");
        setAudioUrl(null);
        setShowSuccess(false);
        setConversionProgress(0);
        setShowChatbot(false);
        setChatHistory([]);
        setLectureSummary(""); // Reset lecture summary when new file is selected
      } catch (error) {
        setErrorMessage(error.message);
        setFile(null);
      }
    },
    [validateFile]
  );

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatMessage.trim() || !lectureSummary) return;

    const userMessage = chatMessage.trim();

    // Add user message to chat history immediately
    setChatHistory((prev) => [...prev, { role: "user", content: userMessage }]);

    // Clear input field
    setChatMessage("");

    try {
      // Make the API request to the backend with context
      const response = await axios.post(
        "http://localhost:5001/api/chat",
        {
          userMessage,
          chatId,
          summary: lectureSummary,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      // Handle the response from the Gemini AI
      setChatHistory((prev) => [
        ...prev,
        {
          role: "bot",
          content: response.data.reply || "No response received from Gemini.",
        },
      ]);
    } catch (error) {
      console.error("Error in Gemini API:", error);
      const errorMessage =
        error.response?.data?.error ||
        "Sorry, I encountered an error while connecting to Gemini.";

      setChatHistory((prev) => [
        ...prev,
        {
          role: "bot",
          content: errorMessage,
        },
      ]);
    }
  };

  const extractAudio = async (videoFile) => {
    try {
      console.log("Starting audio extraction from video:", {
        fileName: videoFile.name,
        fileType: videoFile.type,
        fileSize: videoFile.size,
      });

      await ffmpeg.writeFile("input.mp4", await fetchFile(videoFile));
      console.log("Video file written successfully");

      const ffmpegVersion = await ffmpeg.exec(["-version"]);
      console.log("FFmpeg version:", ffmpegVersion);

      ffmpeg.on("progress", ({ progress, time }) => {
        console.log("Extraction progress:", {
          progress: Math.round(progress * 100),
          timeMs: time,
        });
        setConversionProgress(Math.round(progress * 100));
      });

      await ffmpeg.exec([
        "-i",
        "input.mp4",
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ab",
        "128k",
        "-ar",
        "44100",
        "-y",
        "output.mp3",
      ]);

      const data = await ffmpeg.readFile("output.mp3");
      const audioBlob = new Blob([data.buffer], { type: "audio/mpeg" });

      await ffmpeg.deleteFile("input.mp4");
      await ffmpeg.deleteFile("output.mp3");

      return audioBlob;
    } catch (error) {
      console.error("Audio extraction error:", {
        message: error.message,
        stack: error.stack,
      });
      throw new Error(`Failed to extract audio: ${error.message}`);
    }
  };

  const saveAudioToServer = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "extracted-audio.mp3");

    try {
      setProcessingStatus("Uploading to server...");
      const response = await axios.post(
        "http://localhost:5000/api/upload-and-summarize",
        // `${BASE_URL}/api/upload-and-summarize`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 60000,
          onUploadProgress: (progressEvent) => {
            const uploadProgress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setConversionProgress(uploadProgress);
          },
        }
      );

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      setAudioUrl(null);
      setLectureSummary(response.data.summary); // Store the summary
      return response.data;
    } catch (error) {
      if (error.code === "ECONNABORTED") {
        throw new Error(
          "Request timed out. Please try again with a shorter video."
        );
      }
      if (error.response) {
        throw new Error(
          `Server error: ${error.response.data.error || "Unknown server error"}`
        );
      } else if (error.request) {
        throw new Error(
          "No response from server. Please check your connection."
        );
      } else {
        throw new Error(`Upload failed: ${error.message}`);
      }
    } finally {
      setProcessingStatus("");
    }
  };

  const handleUpload = async () => {
    try {
      if (!file) {
        throw new Error("Please select a file to upload.");
      }

      if (!ffmpegLoaded) {
        throw new Error("Audio processor is not ready. Please wait.");
      }

      setUploading(true);
      setErrorMessage("");

      setProcessingStatus("Extracting...");
      const audioBlob = await extractAudio(file);

      setProcessingStatus("Processing audio...");
      const result = await saveAudioToServer(audioBlob);

      setShowSuccess(true);
      setShowChatbot(true);

      // Add the summary to the chat history
      setChatHistory([
        { role: "bot", content: "Summary of the lecture:" },
        { role: "bot", content: result.summary },
      ]);
    } catch (error) {
      console.error("Processing Error:", error);
      setErrorMessage(error.message);
    } finally {
      setUploading(false);
      setConversionProgress(0);
      setProcessingStatus("");
    }
  };
 

  return (
    <>
    <div className="home">
      <div className="container">
        <div className="lectureEase">
          <Link to="/">
            <h1 className="text-4xl font-bold mb-4">LectureEase</h1>
          </Link>
        </div>
        <div className="login">
          <Link to="/" className="bg-white text-blue-500 font-bold py-2 px-4 rounded">
            Logout
          </Link>
        </div>
      </div>
      <div className="left-panel">
        <div className="lecture-summary">
          <h2>Lecture Summary</h2>
          <h3>Upload your video to get an AI summary!</h3>
        </div>

        <div className="file-upload-container">
          <input
            type="file"
            accept="video/mp4, video/webm, video/ogg"
            onChange={handleFileChange}
          />
          {errorMessage && <p className="error-message">{errorMessage}</p>}
          {file && (
            <button
              className="upload-button"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Upload and Summarize"}
            </button>
          )}
        </div>

        {processingStatus && (
          <div className="processing-status">
            <p>{processingStatus}</p>
            <progress value={conversionProgress} max="100" />
          </div>
        )}

        {showSuccess && (
          <div className="success-message">
            <p>Summary Extracted Successfully!!<br></br>Proceed to the Chatbot</p>
          </div>
        )}
      </div>

      {showChatbot && (
        <div className="right-panel">
          <div className="chatbot-container">
            {/* <div className="summary-section">
              <h4>Lecture Summary:</h4>
              <p>{lectureSummary}</p>
            </div> */}
            <div className="chat-history">
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={msg.role}>
                  <p>{msg.content}</p>
                </div>
              ))}
            </div>
            <form
              onSubmit={handleChatSubmit}
              className="chat-input-container"
            >
              <input
                type="text"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="Ask questions about the lecture..."
              />
              <button type="submit">Send</button>
            </form>
          </div>
        </div>
      )}
    </div>
    
 </> 
  );
};

export default FileUpload;
