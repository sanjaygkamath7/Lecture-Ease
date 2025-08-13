import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing in environment variables.")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model with the correct model name
model = genai.GenerativeModel('gemini-1.5-flash')

# Store active chats and their summaries
active_chats = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Add detailed logging
        print("Received request to /api/chat")
        
        # Check if request has JSON data
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({
                "error": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        print(f"Request data: {data}")
        
        # Validate required fields
        if not data:
            print("No data received")
            return jsonify({
                "error": "No data received"
            }), 400
        
        user_message = data.get('userMessage')
        chat_id = data.get('chatId', 'default')  # Provide default chat_id
        summary = data.get('summary', '')  # Provide default summary
        
        print(f"User message: {user_message}")
        print(f"Chat ID: {chat_id}")
        print(f"Summary length: {len(summary) if summary else 0}")
        
        if not user_message or not isinstance(user_message, str):
            print("Invalid user message")
            return jsonify({
                "error": "Invalid input. Please provide a valid message."
            }), 400

        # Initialize or get existing chat
        if chat_id not in active_chats:
            print(f"Creating new chat for ID: {chat_id}")
            chat = model.start_chat(history=[])
            active_chats[chat_id] = {
                'chat': chat,
                'summary': summary
            }
        else:
            print(f"Using existing chat for ID: {chat_id}")
            chat = active_chats[chat_id]['chat']
            # Update summary if provided
            if summary:
                active_chats[chat_id]['summary'] = summary

        # Get the current summary
        current_summary = active_chats[chat_id]['summary']

        # Construct prompt with context
        if current_summary and current_summary.strip():
            contextualized_prompt = f"""Hey! I just attended a lecture and here's what we covered:

{current_summary}

Now someone is asking me: "{user_message}"

Can you help me answer this based on what was discussed in the lecture? If they're asking about something that wasn't covered in the lecture, just let them know that topic wasn't discussed. Keep it natural and conversational - like I'm explaining it to a friend."""
        else:
            contextualized_prompt = f"""Someone is asking: "{user_message}"

Can you help me give them a good, natural response? Just be conversational and helpful."""

        print(f"Sending prompt to Gemini (length: {len(contextualized_prompt)})")

        # Generate response using Gemini
        response = chat.send_message(contextualized_prompt)
        
        print(f"Received response from Gemini (length: {len(response.text)})")

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get a response from the AI: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "api_key_configured": bool(GOOGLE_API_KEY)})

@app.route('/api/test', methods=['POST'])
def test_endpoint():
    try:
        data = request.get_json()
        return jsonify({
            "message": "Test endpoint working",
            "received_data": data,
            "api_key_configured": bool(GOOGLE_API_KEY)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"Starting Flask app...")
    print(f"API Key configured: {bool(GOOGLE_API_KEY)}")
    app.run(debug=True, host='0.0.0.0', port=5001)