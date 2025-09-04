# Polyglot Quant AI 🚀

An all-in-one financial learning and analysis platform designed to empower users on their journey from novice to proficient trader. Built with **Streamlit** and powered by **Google's Gemini API**, this application provides a comprehensive ecosystem for financial education, portfolio backtesting, algorithmic trading simulation, market screening, and AI-driven document analysis, with a core focus on supporting Indian vernacular languages.

## 🎥 Video Demonstration

A full walkthrough of the application's five pillars and its core functionalities.

**[Watch Demo Video](https://www.youtube.com/watch?v=-HAKIyURp7I)**

## ✨ The Vision: A Complete Financial Journey

The financial world can be intimidating. **Polyglot Quant AI** was built to address this challenge by creating a single, integrated platform that supports a user's entire journey. The application is structured around five powerful, interconnected pillars, moving seamlessly from foundational knowledge to advanced, practical application.

A primary objective of this project, in keeping with the diverse landscape of India, was to break down language barriers in financial education. Leveraging the **Gemini API**, the platform offers extensive support for **22 scheduled Indian vernacular languages**, ensuring that knowledge is accessible to everyone.

## 🏛️ The Five Pillars of Polyglot Quant AI

### 1. 🎓 Interactive Learning Center

The journey begins with a solid foundation. This pillar provides a structured, interactive curriculum on the fundamentals of investing.

- **Interactive Quizzes**: Reinforce learning with quizzes after each module. The one-question-at-a-time format with instant, detailed feedback ensures a focused and effective learning experience.
- **Polyglot Support**: In line with the project's core vision, all educational content, including modules and quizzes, can be instantly translated into 22 Indian languages, making financial literacy truly accessible.

### 2. 📈 Portfolio Playground

This pillar allows users to move from theory to practice by backtesting long-term investment strategies against years of historical market data.

- **Advanced Optimization Models**: Test and compare classic portfolio strategies like Mean-Variance Optimization and Risk Parity.
- **Realistic Simulation**: The backtester accounts for real-world factors like transaction costs and slippage to provide more accurate performance insights.
- **Benchmark Comparison**: Visualize your portfolio's growth and risk metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown) against the Nifty 50 benchmark.
- **Head-to-Head Strategy Comparison**: Run multiple backtests with different parameters and overlay their equity curves on a single chart for direct comparison.

### 3. 🤖 Algorithmic Trading Playground

For those interested in active, rule-based trading, this pillar provides a sandbox to simulate popular technical trading strategies on individual stocks.

- **Diverse Strategy Library**: Test and analyze four common strategies: Moving Average Crossover, MACD Crossover, Bollinger Bands, and Mean Reversion.
- **Visual Trade Analysis**: The platform plots every single buy and sell signal directly on the stock's price chart, allowing you to visually inspect and understand exactly how and why the strategy performed as it did.
- **Detailed Performance Analytics**: Get a full breakdown of your strategy's performance, including total returns and risk-adjusted metrics, compared to a simple buy-and-hold approach.

### 4. 🔎 Dynamic Stock Screener

Discover trading and investment opportunities in real-time. The screener acts as a powerful funnel to find stocks that meet your specific technical criteria from a broad market universe.

- **Large Stock Universe**: Scans over 150 of India's top, liquid stocks from the Nifty 500.
- **Multi-Filter Capability**: Combine a variety of technical filters—such as Golden Cross, RSI Oversold/Overbought, High Relative Volume, and New 52-Week Highs—to create highly specific and powerful custom screens.
- **Instant, Detailed Results**: View key metrics for all stocks that match your scan in a clean, filterable table, allowing for quick analysis and decision-making.

### 5. 📄 FinSummarizer (RAG with Gemini)

This is the most advanced pillar, an intelligent tool that uses **Retrieval-Augmented Generation (RAG)** to have a natural language conversation with your financial documents.

- **AI-Powered Summaries**: Upload any financial document (PDFs of annual reports, RBI circulars, SEBI regulations), and the Gemini API provides an instant, high-level summary.
- **Chat with Your Documents**: Ask specific, detailed questions in plain English (e.g., "What were the company's main risks related to competition?") and get direct, relevant answers extracted from the text.
- **Multilingual Analysis**: Extend the platform's polyglot capabilities to your research. You can ask questions and receive answers about your documents in your preferred Indian language.

## 🛠️ Tech Stack & Key Libraries

This application was built using a modern, powerful stack of open-source libraries.

### Application Framework
- **Streamlit**: For building the interactive web application

### Data & Quantitative Analysis
- **Pandas & NumPy**: For data manipulation and numerical computation
- **yfinance**: For downloading historical financial market data
- **SciPy**: For scientific and technical computing, used in portfolio optimization
- **pandas-ta**: For calculating a wide range of technical analysis indicators
- **Matplotlib**: For creating detailed financial charts

### AI & Language Models (Google Gemini)
- **gemini-1.5-flash-latest**: Used for fast and efficient text translations and RAG-based Q&A
- **models/text-embedding-004**: A specialized model for converting document text into numerical vectors for the RAG system

### Document & Utility Libraries
- **PyPDF**: For extracting text from user-uploaded PDF documents

## ⚙️ Setup & Installation

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- **Python 3.9** or higher
- **Git**

### 2. Clone the Repository

Open your terminal and clone the repository:

git clone https://github.com/Abhinaba925/Polyglot-Quant-AI.git
cd Polyglot-Quant-AI


### 3. Create a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage project dependencies.

For Mac/Linux
python3 -m venv venv
source venv/bin/activate

For Windows
python -m venv venv
.\venv\Scripts\activate


### 4. Install Dependencies

Install all the required libraries from the `requirements.txt` file:

pip install -r requirements.txt


### 5. Add Your Gemini API Key

The translation and FinSummarizer features require a Google Gemini API key.

1. Get your free key from [Google AI Studio](https://aistudio.google.com/)
2. In the main project directory, create a new folder named `.streamlit`
3. Inside this `.streamlit` folder, create a file named `secrets.toml`
4. Add your API key to this file in the following format:

GEMINI_API_KEY = "YOUR_API_KEY_HERE"


### 6. Run the Streamlit Application

Once the setup is complete, run the following command in your terminal:

streamlit run app.py


The application will automatically open in your default web browser at `http://localhost:8501`.

## 🚀 Getting Started

1. **Start with the Learning Center** to build your foundational knowledge
2. **Practice with Portfolio Playground** to understand long-term investing
3. **Experiment with Algorithmic Trading** to explore active strategies
4. **Use the Stock Screener** to identify opportunities
5. **Leverage FinSummarizer** to analyze financial documents with AI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
