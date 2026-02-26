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
def process_documents(uploaded_files):
    """Processes uploaded PDFs and stores them in a temporary vector database."""
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    # Using an ephemeral in-memory Chroma instance
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- 2. Security & Authentication Setup ---
try:
    credentials = st.secrets["credentials"].to_dict()
    cookie = st.secrets["cookie"].to_dict()

    authenticator = stauth.Authenticate(
        credentials,
        cookie["name"],
        cookie["key"],
        int(cookie["expiry_days"])
    )
except Exception as e:
    st.error(f"Configuration Error: {e}. Check your Dashboard Secrets!")
    st.stop()

# Handle Login
try:
    result = authenticator.login(label='Login', location='main')
except TypeError:
    result = authenticator.login('main')

if isinstance(result, tuple):
    name, authentication_status, username = result
else:
    name = st.session_state.get("name")
    authentication_status = st.session_state.get("authentication_status")

# --- 3. Security Audit Engine ---
def run_security_audit(code):
    checks = {
        "Block Structure": "FUNCTION_BLOCK" in code.upper() and "END_FUNCTION_BLOCK" in code.upper(),
        "Input Clamping": "LIMIT" in code.upper(),
        "3-Way Handshake": "i_HMI_Confirm" in code,
        "Safety Interlock": "Global_Safety_DB" in code,
        "Memory (Static VAR)": "VAR" in code and "END_VAR" in code
    }
    return checks

# --- 4. Main App Logic ---
if authentication_status:
    # Sidebar Setup
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"User: {name}")
    
    st.sidebar.title("📚 Knowledge Base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Siemens Manuals (PDF)", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files and "vectorstore" not in st.session_state:
        with st.sidebar:
            with st.spinner("Indexing manuals..."):
                st.session_state.vectorstore = process_documents(uploaded_files)
                st.success("Knowledge Base Ready!")

    # Main UI
    st.title("🤖 Siemens SCL Agent + RAG")
    st.markdown("Generates production-ready **FBs** using AI, RAG context, and safety audits.")

    req = st.text_area("Describe the PLC Function (e.g., Lead/Lag Pump Control with Pressure Safety):")

    if st.button("Generate Function Block"):
        if not req:
            st.warning("Please enter a requirement first.")
        else:
            context = ""
            if "vectorstore" in st.session_state:
                docs = st.session_state.vectorstore.similarity_search(req, k=3)
                context = "\n".join([d.page_content for d in docs])
                st.sidebar.info("Context retrieved from manuals.")

            with st.spinner("Analyzing requirements and generating SCL..."):
                prompt = f"""
                Act as a Senior Siemens PLC Developer. Generate a complete Siemens S7-1500 FUNCTION_BLOCK in SCL.
                
                REFERENCE CONTEXT FROM UPLOADED MANUALS:
                {context if context else "No extra context. Use standard S7-1500 SCL practices."}

                USER REQUIREMENT: {req}

                STRICT RULES:
                1. Syntax: FUNCTION_BLOCK "FB_Generated_Logic"
                2. Input: i_AI_Req (Bool), i_HMI_Confirm (Bool), i_System_Ready (Bool), and process sensors.
                3. Output: q_Execute (Bool) and status indicators.
                4. Safety check: First line after BEGIN must be: IF NOT "Global_Safety_DB".All_Systems_OK THEN RETURN; END_IF;
                5. Use LIMIT(MN:=, IN:=, MX:=) for analog scaling.
                6. Logic: Trigger output only if i_AI_Req, i_HMI_Confirm, and i_System_Ready are TRUE.

                Output ONLY raw SCL text. No markdown formatting.
                """

                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                try:
                    response = model.generate_content(prompt)
                    scl_code = response.text.strip()
                    scl_code = scl_code.replace("```scl", "").replace("```", "").strip()
                    st.session_state.scl_code = scl_code
                except Exception as e:
                    st.error(f"Generation Error: {e}")

    # --- 5. Display Results ---
    if "scl_code" in st.session_state:
        scl_code = st.session_state.scl_code
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("📋 Copy-Paste SCL Code")
            st.code(scl_code, language="pascal")
            st.download_button(
                label="💾 Download .SCL File",
                data=scl_code,
                file_name="AI_FB_Block.scl",
                mime="text/plain"
            )

        with col2:
            st.subheader("🛡️ Security Audit")
            audit = run_security_audit(scl_code)
            for check, passed in audit.items():
                st.write(f"{'✅' if passed else '❌'} {check}")

            if all(audit.values()):
                st.success("Validated for TIA Portal")
            else:
                st.error("Audit Failed - Manual Review Required")

elif authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.info("Please log in to generate industrial logic.")
