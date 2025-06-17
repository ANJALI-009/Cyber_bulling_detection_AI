# Cyberbullying Detection AI

This project provides an AI-powered system for detecting cyberbullying and toxic content in text. Built with Streamlit and Hugging Face transformers, it offers an intuitive web interface for analyzing user-entered text.

## Features

- **Real-time Text Analysis**: Instantly analyze input text for harmful content.
- **Hate Speech Detection**: Identifies and flags hate speech.
- **Offensive Language Detection**: Detects offensive and inappropriate language.
- **Confidence Scores**: Provides scores for different categories (Hate, Offensive, Normal) to indicate the likelihood of harmful content.
- **Visual Feedback**: Uses progress bars and color-coded alerts to highlight analysis results.
- **Model Flexibility**: Attempts to load PyTorch model with a fallback to Flax if PyTorch fails.

## Setup

To set up and run this project locally, follow these steps:

1. **Clone the repository (if you haven't already):**

   ```bash
   git clone <repository_url>
   cd "Cyber bulling AI"
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3. **Enter the text** you want to analyze in the text area and click the "Analyze Text" button to see the results
   
4. **Enter the text** you want to analyze in the text area and click the "Analyze Text" button to see the results. 
