# Summarization and Querying Earning Calls Transcripts PDF with LLaMA2 and Streamlit

Earnings calls are crucial events in the financial world where publicly traded companies communicate their financial performance, strategies, and outlook to analysts, investors, and the public. These calls, often in the form of transcripts, contain a wealth of information that can significantly impact investment decisions and market trends. However, the sheer volume and complexity of these transcripts pose challenges for efficient analysis. Summarizing and querying earnings call transcripts address these challenges by distilling vast amounts of information into concise, meaningful insights.

The Earnings Call Summarizer project aims to streamline and enhance the analysis of earnings call transcripts by leveraging below key components.

**Lang Chain Integration:** LangChain enables LLM models to generate responses based on the most up-to-date information available online, and also simplifies the process of organizing large volumes of data so that it can be easily accessed by LLMs.

**LLaMa2 Framework:** LLaMa2 stands out as a framework for large-scale language model querying, providing sophisticated tools for context-aware searches. With its capabilities, the project ensures nuanced analysis of earning call transcripts, empowering users to glean deeper insights into the textual data and identify recurring themes and 

**Vector Storage with ChromoDB:** ChromoDB serves as a scalable vector storage solution, optimizing the storage and retrieval of vectors crucial for efficient summarization and querying. Its efficiency plays a pivotal role in managing and organizing data, contributing to the seamless extraction of relevant information from earning call transcripts.

**Streamlit App:** The Streamlit app provides a user-friendly interface, simplifying the interaction with the summarized earning call information. With minimal code, it enables the development of a dynamic web application, offering a seamless experience for users to visualize, query, and derive actionable insights from financial documents.

### Architecture

**Architecture**
![alt text](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/git-images/git-image-2.png)


# Getting Started

## Prerequisites
Follow these steps to set up and run the project on your local machine.

#### Usage
1. Install required Python packages from requirements.txt
2. Clone the repository: git clone [https://github.com/jvadlamudi2/EarningsCallSummarizer_LlaMA2.git]
3. Navigate to the project directory: cd EarningsCallSummarizer_LlaMA2
4. Run the Streamlit app:
    ```
    streamlit run app.py
    ```
The app will be accessible at http://localhost:8501 in your web browser.

