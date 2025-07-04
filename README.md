# ChatBot

A simple command-line chatbot using DeepSeek and LangChain.

## Features
- Chat with an AI model using DeepSeek via LangChain
- Loads API key securely from a `.env` file

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your environment variables**
   - Create a `.env` file in the project root:
     ```env
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     ```

## Usage

Run the chatbot from your terminal:
```bash
python app.py
```
Type your message and press Enter. Type `exit` or `quit` to stop the chatbot.

## Files
- `app.py`: Main chatbot application
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not tracked by git)

## Notes
- Make sure your `.env` file is not committed to version control (see `.gitignore`).
- This project is for educational/demo purposes.
