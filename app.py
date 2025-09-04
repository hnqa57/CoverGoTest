import streamlit as st
from summarizer import summarize_text
from pdf_extractor import extract_text_from_pdf
from text_cleaner import clean_text
from QA_chatbot import ask_question
import base64

# Set page title and icon
st.set_page_config(
    page_title="PDF Summarizer & Q&A Chatbot",
    page_icon=":book:",  # Emoji icon or you can use an image path
    layout="centered",  # Optional: can be "centered" or "wide"
)


# Function to load and encode the logo
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Path to the logo
# # logo_path = "8943377.png"
# logo_base64 = get_base64_of_bin_file(logo_path)

# Custom CSS
st.markdown(
    f"""
    <style>
        .main {{
            background-color: #e6f0ff; /* Light Blue */
        }}
        .stTextInput > div > div > input {{
            border: 2px solid #004080; /* Dark Blue */
        }}
        .stButton>button {{
            background-color: #004080; /* Dark Blue */
            color: white;
            border-radius: 5px;
            border: 2px solid #004080; /* Dark Blue */
        }}
        .stButton>button:hover {{
            background-color: #003366; /* Darker Blue */
            border: 2px solid #003366; /* Darker Blue */
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #b3d9ff; /* Light Blue Background for header */
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .logo {{
            width: 100px;
        }}
    </style>
""",
    unsafe_allow_html=True,
)

# App title with logo
st.markdown(
    f"""
    <div class="header">
        <h2>PDF Text Summarization and Q&A Chatbot</h2>
   
    </div>
    <hr>
""",
    unsafe_allow_html=True,
)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text with error handling
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    # Check for OCR errors and provide guidance
    if raw_text.startswith("OCR not available") or raw_text.startswith("Error during"):
        st.error("⚠️ OCR Error Detected")
        st.warning(raw_text)

        st.markdown("""
        ### 🔧 To fix this issue:
        
        **1. Install Tesseract OCR with Chinese support:**
        
        📥 **Download Links:**
        - **Windows**: [Download Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
        - **During installation**: Check "Additional language data" for Chinese/multilingual support
        
        💻 **PowerShell Commands:**
        ```
        winget install -e --id UB-Mannheim.TesseractOCR
        ```
        or
        ```
        choco install tesseract -y
        ```
        
        **2. For your insurance form (Chinese/English):**
        - Tesseract needs `chi_sim`, `chi_tra`, and `eng` language packs
        - The app will automatically detect and use appropriate languages
        
        **3. After installation:**
        - Restart this app
        - Or specify Tesseract path below if it's not in PATH
        """)

        # Download link button
        st.markdown(
            """
        <a href="https://github.com/UB-Mannheim/tesseract/wiki" target="_blank">
            <button style="background-color:#004080;color:white;padding:10px;border:none;border-radius:5px;cursor:pointer;">
                📥 Download Tesseract OCR
            </button>
        </a>
        """,
            unsafe_allow_html=True,
        )

        # Allow user to specify Tesseract path
        tesseract_path = st.text_input(
            "Tesseract executable path (optional):",
            placeholder="C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        )

        if tesseract_path and st.button("Retry with custom path"):
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                with st.spinner("Retrying text extraction..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                st.success("✅ Retry successful!")
            except Exception as e:
                st.error(f"❌ Retry failed: {str(e)}")

        # Stop processing if OCR failed
        if raw_text.startswith("OCR not available") or raw_text.startswith(
            "Error during"
        ):
            st.stop()

    # Clean and display extracted text
    cleaned_text = clean_text(raw_text)

    # Display extracted text
    st.subheader("Extracted Text")
    if len(cleaned_text.strip()) < 50:
        st.warning(
            "⚠️ Very little text extracted. This might be an image-based PDF that requires OCR."
        )

    st.text_area("Extracted Text", cleaned_text, height=300)

    # Summarization section
    if st.button("Summarize"):
        summary = summarize_text(cleaned_text)
        st.subheader("Summary")
        st.success(summary)

    # Q&A section
    st.subheader("Ask Questions About the PDF")
    question = st.text_input("Enter your question:")
    if question:
        answer = ask_question(question, cleaned_text)
        st.subheader("Answer")
        st.info(answer)
