import streamlit as st
import google.generativeai as genai
import pypdf
import pandas as pd

# --- UI Styling ---
def apply_css():
    """Applies the dark theme CSS for a consistent look."""
    st.markdown("""<style>
        .stApp { background-color: #0f172a; }
        .stApp > header { background: transparent; }
        div.stBlock, div.st-emotion-cache-1y4p8pa {
            background-color: #1e293b; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #334155; }
        .stFileUploader > label { font-size: 1.2em; font-weight: bold; }
        </style>""", unsafe_allow_html=True)

# --- Language Constants ---
LANGUAGES = {
    "English": "en", "Assamese": "as", "Bengali": "bn", "Bodo": "brx", "Dogri": "doi",
    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Kashmiri": "ks", "Konkani": "gom",
    "Maithili": "mai", "Malayalam": "ml", "Manipuri (Meitei)": "mni", "Marathi": "mr",
    "Nepali": "ne", "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Santhali": "sat",
    "Sindhi": "sd", "Tamil": "ta", "Telugu": "te", "Urdu": "ur"
}

# --- RAG Core Functions ---

@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    """Splits a long text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_data(show_spinner="Creating document embeddings...")
def create_text_embeddings(_chunks):
    """Creates embeddings for text chunks using Gemini API."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return pd.DataFrame(_chunks, columns=['chunk_text']).assign(embeddings=result['embedding'])
    except Exception as e:
        st.error(f"Failed to create embeddings: {e}")
        return None

def find_best_passages(query, dataframe, top_n=3):
    """Finds the most relevant passages in the dataframe based on a query."""
    try:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        # Using dot product for similarity search
        dataframe['similarity'] = dataframe.embeddings.apply(lambda x: pd.Series(x).dot(query_embedding['embedding']))
        return dataframe.nlargest(top_n, 'similarity')['chunk_text'].tolist()
    except Exception as e:
        st.error(f"Error finding best passages: {e}")
        return []

def generate_rag_answer(query, passages, language):
    """Generates an answer using RAG from the provided passages in the specified language."""
    escaped_passages = "\n".join(f"- {p.replace('`', ' ')}" for p in passages)
    prompt = f"""
    You are a helpful financial analyst AI. Based ONLY on the following passages from a document, answer the user's query in the **{language}** language.

    PASSAGES:
    {escaped_passages}

    QUERY:
    {query}

    Answer (in {language}):
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        answer = model.generate_content(prompt)
        return answer.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."

# --- Main Render Function ---
def render_summarizer():
    apply_css()
    st.title("📄 FinSummarizer: RAG for Financial Documents")
    st.markdown("Upload a financial document (e.g., Annual Report, SEBI circular) to summarize it or ask specific questions in your chosen language.")

    # --- Session State Initialization ---
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = None
    if 'chunk_embeddings_df' not in st.session_state:
        st.session_state.chunk_embeddings_df = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # --- Controls in Sidebar ---
    with st.sidebar:
        st.header("📄 FinSummarizer Controls")
        selected_lang_name = st.selectbox(
            "Select Language for Analysis",
            options=list(LANGUAGES.keys()),
            key="summarizer_lang"
        )

    # --- PDF Upload and Processing ---
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    if uploaded_file:
        if st.button("Process Document", use_container_width=True, type="primary"):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            if st.session_state.pdf_text:
                text_chunks = split_text_into_chunks(st.session_state.pdf_text)
                st.session_state.chunk_embeddings_df = create_text_embeddings(text_chunks)
                st.session_state.messages = [] # Clear chat on new document
                st.success("Document processed successfully! You can now summarize or ask questions.")

    # --- Main Interaction Area ---
    if st.session_state.chunk_embeddings_df is not None:
        st.markdown("---")
        
        # --- Summarization Feature ---
        st.subheader(f"Summarize Document in {selected_lang_name}")
        if st.button("Generate Executive Summary", use_container_width=True):
            with st.spinner(f"Generating summary in {selected_lang_name}..."):
                summary_passages = find_best_passages("Provide a concise executive summary of the entire document.", st.session_state.chunk_embeddings_df, top_n=5)
                summary = generate_rag_answer("Provide a concise executive summary of the entire document.", summary_passages, selected_lang_name)
                st.markdown(summary)
        
        st.markdown("---")
        
        # --- Q&A Chat Feature ---
        st.subheader("Ask a Question")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask in any language... (answers will be in {selected_lang_name})"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner(f"Finding relevant information and generating answer in {selected_lang_name}..."):
                relevant_passages = find_best_passages(prompt, st.session_state.chunk_embeddings_df)
                if not relevant_passages:
                    response = "I could not find relevant information in the document to answer your question. Please try another query."
                else:
                    response = generate_rag_answer(prompt, relevant_passages, selected_lang_name)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    else:
        st.info("Please upload and process a PDF document to begin.")

