import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import FileUpload from './FileUpload';
import Login from './components/login';
import Signup from './components/signup';
import './index.css';

const App = () => {
  const [videoUrl, setVideoUrl] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100 flex">
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<FileUpload setVideoUrl={setVideoUrl} />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={< Signup />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
