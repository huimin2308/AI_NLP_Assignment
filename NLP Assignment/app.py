import os
import time # Record response time

# Export chat history
import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle

# streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

# langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# import from other python file
import file_utils as fHandler
import llmInteraction as llmHandler
import text_preprocessing as preprocessHandler
from Firebase import FirebaseChatManager

# Create a firebase interaction instance
chat_manager = FirebaseChatManager()

# Current Application: No Login Feature, hence the user id is constant along all sessions
# For future improvement purpose: the database has preserved the place for future login implementation
USER_ID = "user123"

# Temp file path --> to keep the conversation activate
TEMP_UPLOAD_DIR = "./temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Avatar images
ASSISTANT_AVATAR = "https://img.freepik.com/free-vector/3d-ai-robot-character-chat-bot-wink-mascot-icon_107791-30020.jpg?semt=ais_hybrid&w=740"
USER = "https://img.freepik.com/free-vector/3d-cartoon-young-woman-smiling-circle-frame-character-illustration-vector-design_40876-3100.jpg?semt=ais_hybrid&w=740"

# ========================= Initializer for session state ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "full_document_text" not in st.session_state:
    st.session_state.full_document_text = ""
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "readonly_mode" not in st.session_state:
    st.session_state.readonly_mode = False
if "preprocess" not in st.session_state:
    st.session_state.preprocess = None
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "show_export_options" not in st.session_state:
    st.session_state.show_export_options = False
if "model" not in st.session_state:
    st.session_state.model = None
if "file_preview" not in st.session_state:
    st.session_state.file_preview = False

if "firstInput" not in st.session_state:
    st.session_state.firstInput = True
if "firstLoad" not in st.session_state:
    st.session_state.firstLoad = True

# ========================== Save File Uploaded (Temp) ==========================
def save_temp_file(file):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{file.name}"
    file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path, unique_filename

# ========================== Export Helper Function ==========================
def generate_pdf_from_messages(messages, title="Chat History", file_name=None, chat_id=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(name="TitleStyle", parent=styles['Title'], alignment=TA_CENTER, fontSize=18)
    meta_style = ParagraphStyle(name="MetaStyle", parent=styles['Normal'], spaceAfter=6)
    role_style = ParagraphStyle(name="RoleStyle", parent=styles['Heading4'], fontSize=12, spaceAfter=4)
    content_style = ParagraphStyle(name="ContentStyle", parent=styles['Normal'], fontSize=11, spaceAfter=10)

    elements = []

    # Title
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))

    # Metadata
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if file_name:
        elements.append(Paragraph(f"<b>File Name:</b> {file_name}", meta_style))
    if chat_id:
        elements.append(Paragraph(f"<b>Chat ID:</b> {chat_id}", meta_style))
    elements.append(Paragraph(f"<b>Exported On:</b> {current_time}", meta_style))
    elements.append(Spacer(1, 12))

    # Chat messages
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")

        elements.append(Paragraph(f"{role}:", role_style))
        elements.append(Paragraph(content.replace('\n', '<br/>'), content_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string format
    raise TypeError(f"Type {type(obj)} not serializable")

# ========================== Extracted Text Handler ==========================
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". "])
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v2') # Handle multilanguage
    # embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # Only for english
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    return vector_store

# Search similar chunks in vector store
def search_similar_chunks(query, vectorstore, top_k=10):
    # No vector store loaded, can't search
    if not vectorstore:
        return []  
    
    # Search vector store for the user's query 
    docs = vectorstore.similarity_search(query, k=top_k)

    # Return documents if relevant results are found, else return an empty list
    return [doc.page_content for doc in docs] if docs else []

# Handle those common term that related to whole file content
def detect_command(user_input):
    text = user_input.lower()
    if "summarize" in text or "summary" in text or "总结" in text or "meringkaskan" in text or "rumusan" in text:
        return "summarize"
    elif "translate" in text or "terjemah" in text or "翻译" in text or "write in" in text:
        return "translate"
    elif "conclusion" in text or "conclude" in text or "结论" in text or "kesimpulan" in text or "conclude" in text:
        return "conclusion"
    return None

# =============================== Actual App ===============================
# Set page configuration
st.set_page_config(page_title="Interact with your PDF") 
st.title("Chat with your PDF") 

# ========================== Side bar ==========================
with st.sidebar:
    with stylable_container(
    key="new_chat_button",
    css_styles="""
        button {
            background-color: #002664;
            color: white;
            border: none;
            font-size: 24px;     
            font-weight: bold;    
            padding: 1.2em 1.2em;  
            width: 80%;           
            height: 30px;        
            border-radius: 10px;  
        }

    """,
    ):
        if st.button("🗨️ New Chat"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.uploaded_file = None
            st.session_state.full_document_text = ""
            st.session_state.chat_id = None
            st.session_state.readonly_mode = False
            st.session_state.preprocess = None
            st.session_state.uploaded_file_path = None
            st.session_state.show_export_options = False
            st.session_state.model = None
            st.session_state.file_preview = False
    
    add_vertical_space()
    
    # button to trigger export options
    if st.button("📥 Export Chat History", type="tertiary"):
        st.session_state.show_export_options = not st.session_state.show_export_options

    # Once clicked, show options
    if st.session_state.show_export_options:
        st.warning("Before exporting the chat history, kindly click the above button again to ensure the export contents up-to-date.")
        export_format = st.selectbox("Export format", ["Text", "JSON", "PDF"], key="export_format")
        
        if st.session_state.messages:
            if export_format == "Text":
                export_content = "\n\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
                )
                buffer = io.BytesIO(export_content.encode("utf-8"))
                filename = "chat_history.txt"
                mime_type = "text/plain"

            elif export_format == "JSON":
                # Serialize messages with custom serializer
                export_content = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False, default=custom_serializer)
                # Convert to bytes
                buffer = io.BytesIO(export_content.encode("utf-8"))
                filename = "chat_history.json"
                mime_type = "application/json"

            elif export_format == "PDF":
                st.warning("PDF format only supports English characters; others may appear as blocks.")
                buffer = generate_pdf_from_messages(
                    st.session_state.messages,
                    title="Chat History",
                    file_name=st.session_state.get("uploaded_file"),
                    chat_id=st.session_state.get("chat_id")
                )
                filename = "chat_history.pdf"
                mime_type = "application/pdf"
            
            st.download_button(
                label="📥 Download Chat History",
                data=buffer,
                file_name=filename,
                mime=mime_type,
                key="download_button"
            )
        else:
            st.warning("No chat history available to export.")

    # Retrive the chat history
    user_chats = chat_manager.get_user_chats(user_id=USER_ID)

    # If it is first load (new session), deactivate all the chat that stored in the database
    if (st.session_state.firstLoad):
        chat_manager.deactivate_previous_chats(USER_ID)
        st.session_state.firstLoad = False

    if user_chats:
        st.markdown("### Previous Chats")
        for chat in user_chats:
            title = chat.get("chat_name", "Untitled Conversation")
            col1, col2 = st.columns([4, 1])

            if col1.button(title, key=f"load-{chat['id']}", use_container_width=True):
                st.session_state.messages = []
                st.session_state.messages = chat_manager.get_chat_messages(USER_ID, chat["id"])
                st.session_state.vectorstore = None
                st.session_state.uploaded_file = chat.get("file_name", "")
                st.session_state.full_document_text = ""
                st.session_state.chat_id = chat["id"]
                st.session_state.readonly_mode = False if chat["activeChat"] else True
                st.session_state.preprocess = chat.get("preprocess", "")
                st.session_state.uploaded_file_path = chat.get("temp_path", "")
                st.session_state.model = chat.get("model", "")
                # Reactivate the same session's chat
                if not st.session_state.readonly_mode and st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
                    raw_text = fHandler.extract_file(st.session_state.uploaded_file_path)
                    if st.session_state.preprocess == "None":
                        processed_text = raw_text
                    else:
                        remove_stop = st.session_state.preprocess in ["Remove stopwords", "Perform all preprocess tasks"]
                        remove_num = st.session_state.preprocess in ["Remove numbers", "Perform all preprocess tasks"]
                        processed_text = preprocessHandler.preprocess_text(raw_text, remove_stopwords=remove_stop, remove_numbers=remove_num)
                    chunk_text = get_chunks(processed_text)
                    vectorstore = get_vector_store(chunk_text)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.full_document_text = raw_text
                st.session_state.show_export_options = False
                st.session_state.file_preview = False

            if col2.button("🗑️", key=f"del-{chat['id']}", type="tertiary"):
                chat_manager.delete_chat(USER_ID, chat["id"])
                if st.session_state.chat_id == chat["id"]:
                    st.session_state.messages = []
                    st.session_state.vectorstore = None
                    st.session_state.uploaded_file = None
                    st.session_state.full_document_text = ""
                    st.session_state.chat_id = None
                    st.session_state.readonly_mode = False
                    st.session_state.preprocess = None
                    st.session_state.uploaded_file_path = None
                    st.session_state.model = None
                st.session_state.show_export_options = False
                st.session_state.file_preview = False
                st.success("Chat deleted successfully.")
                st.rerun()

# ========================== File Upload ==========================
# Display uploaded file if present
if st.session_state.get("uploaded_file"):
    if not st.session_state.readonly_mode and st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**Uploaded file:** {st.session_state.get('uploaded_file')}")
        
        if col2.button("Preview File"):
            st.session_state.file_preview = not st.session_state.file_preview
        
        if st.session_state.file_preview:
            with st.container(border=True):
                with open(st.session_state.uploaded_file_path, "rb") as f:
                    file_ext = st.session_state.uploaded_file_path.split(".")[-1].lower()
                    if file_ext == "pdf":
                        st.download_button("Download PDF", f, file_name=st.session_state.get("uploaded_file"))
                        st.info("Download to view detail.")

                        # Extract and store PDF pages
                        pdf_pages = fHandler.extract_pdf_images(st.session_state.get("uploaded_file_path"))
                        
                        # Display PDF if pages are available
                        if pdf_pages:
                            # PDF display controls
                            zoom_level = col1.slider(
                                "Zoom Level", 
                                min_value=100, 
                                max_value=1000, 
                                value=700, 
                                step=50,
                                key="zoom_slider"
                            )

                            # Display PDF pages
                            with st.container(height=300):
                                for page_image in pdf_pages:
                                    st.image(page_image, width=zoom_level)
                         
                    else:
                        text = f.read().decode("utf-8", errors="ignore")
                        placeholder = st.empty()
                        placeholder.write(text[:2000])

    else:
        st.markdown(f"**Uploaded file:** {st.session_state.get('uploaded_file')}")

    st.markdown(f"**Preprocessing Carried Out:** {st.session_state.get('preprocess')}")

    model_name = "DeepSeek R1" if st.session_state.get('model') == "deepseek/deepseek-r1:free" else "DeepSeek V3 0324"
    st.markdown(f"**LLM Model Selected:** {model_name}")

else:
    # If no file uploaded yet, show upload interface
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Hi! How can I help you today? Please upload a file, and select your preprocessing option." \
        " For your information, the remove stopwords option means the preprocessing step will include the stopword removal but without removing any numbers, while" \
        " the remove numbers option means the preprocessing step will include the number removal but without removing stopwords.")

    uploaded_pdf = st.file_uploader("Upload a file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    preproc = st.radio("Preprocessing options", ["None", "Remove stopwords", "Remove numbers", "Perform all preprocess tasks"])
    st.session_state.preprocess = preproc

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Select a LLM model for your conversation.")
    model = st.radio("Model", ["DeepSeek R1", "DeepSeek V3 0324"])
    st.session_state.model = "deepseek/deepseek-r1:free" if model == "DeepSeek R1" else "deepseek/deepseek-chat-v3-0324:free"

    if uploaded_pdf and st.button("Upload & Process Document"):
        with st.spinner("Uploading file..."):
            # Save the file - temp folder
            file_path, safe_filename = save_temp_file(uploaded_pdf)

            # Extract text and process it into chunks
            raw_text = fHandler.extract_file(uploaded_pdf)

            # Handle preprocessing
            if preproc == "None":
                # No preprocessing, use raw text 
                processed_text = raw_text
            else:
                remove_stop = preproc in ["Remove stopwords", "Perform all preprocess tasks"]
                remove_num = preproc in ["Remove numbers", "Perform all preprocess tasks"]
                processed_text = preprocessHandler.preprocess_text(raw_text, remove_stopwords=remove_stop, remove_numbers=remove_num)

            chunk_text = get_chunks(processed_text)
            vectorstore = get_vector_store(chunk_text)

            # Store state
            st.session_state.uploaded_file = uploaded_pdf.name
            st.session_state.vectorstore = vectorstore
            st.session_state.full_document_text = raw_text
            st.session_state.uploaded_file_path = file_path  

            # Create a chat session in the database
            chat_id = chat_manager.create_chat_session(USER_ID, chat_name="Chat with " + uploaded_pdf.name, file_name=uploaded_pdf.name, preprocess=preproc, temp_path=file_path, model=st.session_state.model)
            st.session_state.chat_id = chat_id
            st.session_state.readonly_mode = False

            st.success("File uploaded successfully.")

            # Display the extracted text content (first 2000 words)
            st.subheader("Extracted Raw Text from the file uploaded:")
            st.write("(Only shows the first 2000 characters)")
            with st.container(border=True):
                placeholder = st.empty()
                placeholder.write(raw_text[:2000])

            # Add chatbot greeting --> let the user know can start conversation
            st.session_state.firstInput = True
            st.session_state.messages.append({"role": "assistant", "content": "Type 'Hi' and I'll assist you."})
            chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", "Type 'Hi' and I'll assist you.")

# ========================== Display Chat History ==========================
# Show chat history
for message in st.session_state.messages:
    if(message["role"]=="assistant"):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(message["content"])
    elif(message["role"]=="user"):
        with st.chat_message("user", avatar=USER):
            st.markdown(message["content"])

# ========================== Chat Interaction ==========================
# Ask a question if file is uploaded
if st.session_state.get("uploaded_file") and not st.session_state.readonly_mode:
    user_input = st.chat_input("Ask a question based on file content")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=USER):
            st.markdown(user_input)
        chat_manager.add_message(USER_ID, st.session_state.chat_id, "user", user_input)

        # Skip the first input after uploading the file --> the first input is used to transit to the actual chat session
        if (st.session_state.firstInput):
            st.session_state.messages.append({"role": "assistant", "content": "Let's start conversation. Ask me about your file's content."})
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown("Let's start conversation. Ask me about your file's content.")
            st.session_state.firstInput = False
            chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", "Let's start conversation. Ask me about your file's content.")

        else: 
            command = detect_command(user_input)
            vectorstore = st.session_state.get("vectorstore")
            full_doc_text = st.session_state.get("full_document_text", "")

            # Handle those special commands 
            if command:
                if not full_doc_text:
                    response = "Sorry, the document doesn't contain enough content to perform this operation."
                    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", response)

                else:
                    if command == "summarize":
                        full_prompt = f"""The user asked: "{user_input}"\n\n
                        Below is the relevant content from the document:\n{full_doc_text}\n\n
                        Please provide a focused and accurate summary strictly based on the context above.
                        If the input refers to a specific section, limit your response to that part only.
                        If the user ask about the summarization on other languages, strictly follow user's request to provide response in that language.
                        If the user does not mention the response should be in which langauge, provide the response by using the
                        language that the user use. If mixied languages used, provide response using english.
                        """
                    elif command == "translate":
                        full_prompt = f"""The user asked: "{user_input}"\n\n
                        Below is the content from the document:\n{full_doc_text}\n\n
                        Translate the content above as instructed in the user's request, or
                        Answer the question based on user's request and strictly follow the document's content.
                        Do not invent content or translate irrelevant parts.
                        """
                    elif command == "conclusion":
                        full_prompt = f"""The user asked: "{user_input}"\n\n
                        Here is the relevant content from the document:\n{full_doc_text}\n\n
                        Provide a clear conclusion strictly based on the above context and user input.
                        If the user requests the conclusion on other languages, strictly follow user's request to provide conclusion in that language.
                        If the user does not mention the response should be in which langauge, provide the response by using the
                        language that the user use. If mixied languages used, provide response using english.
                        """
                    
                    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                        with st.spinner("Assistant is typing..."):
                            start_time = time.time()  # Start timer

                            stream_fn = llmHandler.load_llm_model(full_prompt, st.session_state.model)
                            response_placeholder = st.empty()
                            full_response = ""

                            for chunk in stream_fn():
                                full_response += chunk
                                response_placeholder.markdown(full_response)
                            
                            end_time = time.time()  # End timer
                            elapsed_time = end_time - start_time

                        # Display response time
                        st.markdown(f"⏱️ Response time: {elapsed_time:.2f} seconds")

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", full_response)
            
            # Normal Question and answer flow
            else:
                # Retrieve context from vector store
                retrieved_contexts = search_similar_chunks(user_input, vectorstore)

                if not retrieved_contexts:
                    response = "Sorry, I couldn't find anything relevant in the document."
                    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", response)

                else:
                    context = "\n\n".join(retrieved_contexts)
                    full_prompt = f"""Context:\n{context}\n\n
                    Question: {user_input}\n\n
                    Answer the question as detailed as possible strictly based on above context, make sure to provide all the details. If no
                    relevant answer is found in the provided context, reply: 'Sorry, I couldn't find anything related to that in the document.',
                    don't provide the wrong answer. The user may ask in different language or has typo, handle the question answering with the most
                    relevant answer, strictly provide the answer based on the provided context and user's input.
                    If the user does not mention the response should be in which langauge, provide the response by using the
                    language that the user use. If mixied languages used, provide response using english.
                    """

                    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                        with st.spinner("Assistant is typing..."):
                            start_time = time.time()  # Start timer

                            stream_fn = llmHandler.load_llm_model(full_prompt, st.session_state.model)
                            response_placeholder = st.empty()
                            full_response = ""

                            for chunk in stream_fn():
                                full_response += chunk
                                response_placeholder.markdown(full_response)
                            
                            end_time = time.time()  # End timer
                            elapsed_time = end_time - start_time

                        # Display response time
                        st.markdown(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                        
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    chat_manager.add_message(USER_ID, st.session_state.chat_id, "assistant", full_response)

else:
    user_input = None
    if st.session_state.messages:
        st.info("This is a past session. To chat with a document, start a new chat session.")