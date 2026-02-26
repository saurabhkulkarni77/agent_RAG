
import streamlit as st
import google.generativeai as genai
import streamlit_authenticator as stauth
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import tempfile
import os

# --- 1. RAG System Functions ---
def process_documents(uploaded_files, api_key):
    try:
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        
        # FIXED: Updated to 2026 stable model name
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        
        return Chroma.from_documents(documents=splits, embedding=embeddings)
    except Exception as e:
        st.error(f"Embedding Model Error: {e}")
        return None

# --- 2. Security Audit Engine ---
def run_security_audit(code):
    checks = {
        "Block Structure": "FUNCTION_BLOCK" in code.upper() and "END_FUNCTION_BLOCK" in code.upper(),
        "Input Clamping": "LIMIT" in code.upper(),
        "3-Way Handshake": "i_HMI_Confirm" in code,
        "Safety Interlock": "Global_Safety_DB" in code,
        "Memory (Static VAR)": "VAR" in code and "END_VAR" in code
    }
    return checks

# --- 3. Simplified Colab Authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Industrial Gateway Login")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pw == "admin123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Credentials")
    st.stop()

# --- 4. Main App Logic ---
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    st.sidebar.title("📚 Knowledge Base")
    uploaded_files = st.sidebar.file_uploader("Upload Siemens PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and "vectorstore" not in st.session_state:
        with st.sidebar:
            with st.spinner("Indexing Manuals..."):
                st.session_state.vectorstore = process_documents(uploaded_files, api_key)
                if st.session_state.vectorstore:
                    st.success("Knowledge Base Ready!")

    st.title("🤖 Siemens SCL FB Agent")
    st.markdown("Generates safety-validated Siemens S7-1500 logic.")
    
    req = st.text_area("Describe Function (e.g., Lead/Lag Pump Control):")

    if st.button("Generate Code"):
        context = ""
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(req, k=3)
            context = "\n".join([d.page_content for d in docs])
            st.info("Retrieved context from manuals.")

        with st.spinner("Writing SCL..."):
            genai.configure(api_key=api_key)
            # Using the latest workhorse model
            model = genai.GenerativeModel('gemini-2.5-flash') 
            
            prompt = f"""
            Act as a Senior Siemens PLC Developer. Generate a FUNCTION_BLOCK in SCL.
            CONTEXT FROM MANUALS: {context}
            REQUIREMENT: {req}
            
            RULES:
            1. Start with FUNCTION_BLOCK "FB_Generated_Logic"
            2. First line after BEGIN: IF NOT "Global_Safety_DB".All_Systems_OK THEN RETURN; END_IF;
            3. Use LIMIT() for analog signals.
            4. Logic must require i_AI_Req AND i_HMI_Confirm AND i_System_Ready.
            5. Output ONLY raw SCL code. No markdown.
            """
            
            response = model.generate_content(prompt)
            scl_code = response.text.replace("```scl", "").replace("```", "").strip()
            st.session_state.scl_code = scl_code

    if "scl_code" in st.session_state:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.code(st.session_state.scl_code, language="pascal")
            st.download_button("💾 Download .SCL", st.session_state.scl_code, "AI_FB.scl")
        with col2:
            st.subheader("🛡️ Audit")
            audit = run_security_audit(st.session_state.scl_code)
            for check, passed in audit.items():
                st.write(f"{'✅' if passed else '❌'} {check}")
else:
    st.warning("Please enter your Gemini API Key in the sidebar.")
