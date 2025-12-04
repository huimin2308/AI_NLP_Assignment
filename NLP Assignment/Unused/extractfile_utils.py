import pdfplumber
import docx2txt
import os

def create_file_object(file_path):
    """Convert a file path to a file-like object compatible with extract_file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Determine MIME type based on file extension
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain'
    }
    mime = mime_types.get(extension, 'application/octet-stream')
    
    # Create a file-like object
    class FileObject:
        def __init__(self, file_path, mime):
            self.file = open(file_path, 'rb')
            self.type = mime
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.file.close()
    
    return FileObject(file_path, mime)

def extract_file(file):
    file_type = file.type

    if file_type == "application/pdf":
        return extract_pdf_text(file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_doc_text(file)
    elif file_type == "text/plain":
        return extract_txt_text(file)
    else:
        return "Unsupported file type."

def extract_pdf_text(file):
    text = ""
    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    continue
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    return text

def extract_doc_text(file):
    return docx2txt.process(file.file)

def extract_txt_text(file):
    return file.file.read().decode("utf-8")