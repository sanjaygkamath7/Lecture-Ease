
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Login.css'; 

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [message, setMessage] = useState('');
    const navigate = useNavigate();

    const handleLogin = async (e) => {
        e.preventDefault();

        try {
            const response = await axios.post(`${process.env.NEXT_PUBLIC_API_BASE_URL}/api/login`, { email, password });
            setMessage(response.data.message);
            localStorage.setItem('token', response.data.token); // Save token locally
            setTimeout(() => {
                navigate('/upload'); // Redirect to upload page after 2 seconds
            }, 2000);
        } catch (err) {
            setMessage(err.response?.data?.message || 'Invalid email or password');
        }
    };

    return (
        <div className="home">
            <div className="container">
                <div className="lectureEase">
                        <h1 className="text-4xl font-bold mb-4">LectureEase</h1>
                   
                </div>
            </div>


            <div className="login-container">
                <h2>Login</h2>
                <form onSubmit={handleLogin} className="login-form">
                    <input
                        type="email"
                        placeholder="Email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <button type="submit">Login</button>
                </form>
                <p className="message">{message}</p>
                <a href="/signup" className="signup-link">Don't have an account? Sign up</a>
            </div>
        </div>
    );
};
export default Login;
