# Import necessary libraries.
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Set web page title and icon.
st.set_page_config(
    page_title="EarningsCallInsightHub",
    page_icon=":robot:"
)

# Set web page title and markdown.
st.title('Earnings Call Insight Hub: Intelligent Summarization and Querying Interface')
st.markdown(
    """
    Welcome to the EarningsInsight Hub! Have a question about your earnings call transcripts? Our intelligent bot is here to help. Simply ask your question in natural language, and the bot will provide you with insightful answers.
    """
)

# Define a function to get user input.
def get_input_text():
    input_text = st.text_input("Ask a question about your transcript:")
    return input_text

# Define to variables to use "sentence-transformers/all-MiniLM-L6-v2" embedding model from HuggingFace.
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

pdfLoader = DirectoryLoader('./data/')
documents = []
documents.extend(pdfLoader.load())
print(f'You have {len(documents)} document(s) in your data folder.')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
print(len(documents))
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db/")
db.persist()

# Define the Chroma vector store and function to generate embeddings.
db = Chroma(persist_directory="./chroma_db/", embedding_function=embeddings)

# Get user input.
user_input = get_input_text()

# Initialize the Azure OpenAI ChatGPT model.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
# model_basename = "llama-2-13b-chat.Q4_K_M.gguf"
# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Define the path of the Llamaccp model.
model_path = "/Users/jahnavi/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGUF/snapshots/4458acc949de0a9914c3eab623904d4fe999050a/llama-2-13b-chat.Q4_K_M.gguf"

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Initialize the llamaCpp model.
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=2048,
    verbose=False,
)

# Define the function to get the response.
if user_input:
    # Perform similarity search for the user input.
    docs = db.similarity_search(user_input)

    # Load the question answering chain.
    chain = load_qa_chain(llm, chain_type="stuff")

    # Get the response from llamaCpp model.
    response = chain.run(input_documents=docs, question=user_input)

    # Display the response.
    st.write(response)
