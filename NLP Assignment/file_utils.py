from PyPDF2 import PdfReader
import docx2txt
import mimetypes
import pdfplumber

def extract_file(file):
    # If the input is a string --> file path
    if isinstance(file, str):
        mime_type, _ = mimetypes.guess_type(file)
        with open(file, "rb") as f:
            return _extract_based_on_type(mime_type, f)
    else:
        # Streamlit UploadedFile
        return _extract_based_on_type(file.type, file)

def _extract_based_on_type(file_type, file_obj):
    # check whethe it is pdf file (using MIME type (Multipurpose Internet Mail Extensions) that identifies Portable Document Format (PDF) files)
    if file_type == "application/pdf":
        return extract_pdf_text(file_obj)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_doc_text(file_obj)
    elif file_type == "text/plain":
        return extract_txt_text(file_obj)
    else:
        return "Unsupported file type."

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            continue

    return text

def extract_doc_text(file):
    return docx2txt.process(file)

def extract_txt_text(file):
    return file.read().decode("utf-8")

def extract_pdf_images(pdf_path):
    """Extract images from PDF pages."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return [page.to_image().original for page in pdf.pages]
    except Exception as e:
        return []