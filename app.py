import streamlit as st
import fitz  # PyMuPDF for PDF processing
import os
import shutil
import tempfile
import atexit
from mistralai import Mistral
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
import base64
import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import traceback
import math
import numpy as np

# ✅ Set Up Temporary Directory
temp_dir = tempfile.mkdtemp()
def cleanup():
    shutil.rmtree(temp_dir, ignore_errors=True)
atexit.register(cleanup)

# ✅ API Keys (read from environment variables only)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ✅ Initialize Mistral Client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ✅ Helper function to validate image 
def is_valid_image(image_path):
    try:
        # Try to open and verify the image
        with Image.open(image_path) as img:
            img.verify()  # Verify the image
        return True
    except:
        return False

# ✅ Upload PDF & Process OCR for Selected Pages Only
def process_pdf_with_mistral(file_bytes, file_name, start_page, end_page):
    try:
        uploaded_pdf = mistral_client.files.upload(
            file={"file_name": file_name, "content": file_bytes},
            purpose="ocr"
        )

        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id).url

        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url
            },
            include_image_base64=True
        )

        # Filter pages based on user selection
        filtered_pages = [page for page in ocr_response.pages if start_page <= page.index + 1 <= end_page]
        ocr_response.pages = filtered_pages

        return ocr_response
    except Exception as e:
        st.error(f"Error processing PDF with Mistral OCR: {str(e)}")
        print(traceback.format_exc())
        return None

# ✅ Extract Text & Images (Mistral OCR)
def extract_text_and_images(ocr_data):
    extracted_data = []
    image_data = []

    for page in ocr_data.pages:
        page_number = page.index + 1  # Convert to 1-based index

        # Extract text
        text = page.markdown.strip() if hasattr(page, "markdown") else ""
        if text:
            extracted_data.append(Document(page_content=text, metadata={"page": page_number, "source_type": "pdf"}))

        # Extract images
        for img in getattr(page, "images", []):
            if hasattr(img, "image_base64") and img.image_base64:
                try:
                    # Decode Base64 Image Correctly
                    image_bytes = base64.b64decode(img.image_base64)
                    
                    # Verify image data is valid before saving
                    try:
                        # Test if it's a valid image by trying to open it
                        Image.open(io.BytesIO(image_bytes))
                        
                        # If we got here, it's valid
                        img_path = os.path.join(temp_dir, f"page_{page_number}_img_{len(image_data)}.png")
                        
                        # Save Image as Binary File
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Extract text from the image (if available)
                        img_text = img.text.strip() if hasattr(img, "text") else ""

                        # Store image in extracted data
                        extracted_data.append(Document(page_content=img_text, metadata={
                            "page": page_number, 
                            "image": img_path, 
                            "source_type": "pdf"
                        }))
                        image_data.append((page_number, img_path, img_text))
                    except Exception as img_verify_error:
                        print(f"⚠️ Ignoring invalid image data on Page {page_number}: {img_verify_error}")
                        
                except Exception as e:
                    print(f"⚠️ Error processing image on Page {page_number}: {e}")

    return extracted_data, image_data

# ✅ Simple hashing-based embeddings (dependency-free)
class SimpleHashEmbeddings:
    def __init__(self, dimension=512, ngram=3):
        self.dimension = dimension
        self.ngram = ngram

    def _hash(self, token: str) -> int:
        # Stable hash using md5
        import hashlib
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension

    def _text_to_vector(self, text: str):
        vec = np.zeros(self.dimension, dtype=np.float32)
        t = text.lower()
        # simple char n-grams
        for i in range(max(1, len(t) - self.ngram + 1)):
            ngram = t[i:i + self.ngram]
            idx = self._hash(ngram)
            vec[idx] += 1.0
        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts):
        return [self._text_to_vector(text or "") for text in texts]

    def embed_query(self, text):
        return self._text_to_vector(text or "")

    # Allow using this instance as a callable embedding function
    def __call__(self, text: str):
        return self.embed_query(text)

# ✅ Build RAG System (FAISS Vector Store)
def build_rag_system(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(extracted_data)

    # Lightweight local embeddings: hashing-based, no external deps
    embeddings = SimpleHashEmbeddings(dimension=512, ngram=3)
    vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever, vector_store

# ✅ Web Search with Image Scraping Functionality
def web_search_with_images(query):
    search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    search_results = search_tool.run(query)
    
    # Extract top search result URL
    try:
        # Parse search results to get the top URL (simplified)
        if "http" in search_results:
            url = search_results.split("http")[1].split(" ")[0]
            url = "http" + url.strip()
            
            # Scrape the webpage
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            text_content = soup.get_text()[:1000]  # Limit text length
            
            # Save a screenshot or extract an image
            img_path = None
            images = soup.find_all('img')
            if images and len(images) > 0:
                for potential_img in images[:3]:  # Try the first 3 images
                    img_src = potential_img.get('src')
                    if img_src:
                        if not img_src.startswith('http'):
                            if img_src.startswith('/'):
                                base_url = url.split('/')[0] + '//' + url.split('/')[2]
                                img_src = base_url + img_src
                            else:
                                img_src = url + '/' + img_src
                        
                        try:
                            img_response = requests.get(img_src, timeout=5)
                            if img_response.status_code == 200:
                                try:
                                    # Verify it's a valid image before saving
                                    Image.open(io.BytesIO(img_response.content))
                                    
                                    img_path = os.path.join(temp_dir, f"web_img_{len(os.listdir(temp_dir))}.png")
                                    with open(img_path, "wb") as f:
                                        f.write(img_response.content)
                                    
                                    # If we found a valid image, break the loop
                                    break
                                except:
                                    # Not a valid image, try the next one
                                    continue
                        except:
                            continue
            
            # Store in our document format
            web_doc = Document(
                page_content=text_content, 
                metadata={
                    "source_type": "web",
                    "url": url,
                    "image": img_path
                }
            )
            
            # Add to vector store if available
            if 'vector_store' in st.session_state and st.session_state.vector_store:
                # Vector store was created with HF embeddings; adding documents is fine
                st.session_state.vector_store.add_documents([web_doc])
            
            return f"Web search found: {text_content[:200]}... from {url}"
    except Exception as e:
        print(f"Error in web search: {e}")
        print(traceback.format_exc())
    
    return search_results

# ✅ Generate Answer (With Image & Text Citations)
def answer_with_rag(query, retriever, image_data):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    retrieved_docs = retriever.get_relevant_documents(query)

    citations = []
    numbered_citations = []
    citation_details = []

    for i, doc in enumerate(retrieved_docs):
        source_type = doc.metadata.get("source_type", "Unknown")
        citation_id = f"citation_{i+1}"
        
        if source_type == "pdf":
            page_num = doc.metadata.get("page", "Unknown")
            
            if "image" in doc.metadata:
                # Image citation (from PDF)
                image_path = doc.metadata["image"]
                if os.path.exists(image_path) and is_valid_image(image_path):
                    try:
                        with open(image_path, "rb") as img_file:
                            img_bytes = img_file.read()
                            img_b64 = base64.b64encode(img_bytes).decode()
                        
                        citation_details.append({
                            "id": citation_id,
                            "type": "image",
                            "source": f"PDF Page {page_num}",
                            "image_b64": img_b64
                        })
                        
                        citations.append(f"📜 **Page {page_num} (Image):**")
                        numbered_citations.append(f"[{i+1}] PDF Page {page_num} (Image)")
                    except Exception as e:
                        print(f"Error processing image citation: {e}")
                        citations.append(f"⚠️ Image processing error: {str(e)}")
                        numbered_citations.append(f"[{i+1}] PDF Page {page_num} (Image processing error)")
                else:
                    citations.append(f"⚠️ Image not found or invalid: {image_path}")
                    numbered_citations.append(f"[{i+1}] PDF Page {page_num} (Image not found)")
            else:
                # Text citation (from PDF)
                snippet = doc.page_content[:200] + "..."
                citation_details.append({
                    "id": citation_id,
                    "type": "text",
                    "source": f"PDF Page {page_num}",
                    "content": doc.page_content
                })
                
                citations.append(f"📜 **Page {page_num}** (Text): {snippet}")
                numbered_citations.append(f"[{i+1}] PDF Page {page_num}")
        
        elif source_type == "web":
            # Web content citation
            url = doc.metadata.get("url", "Unknown URL")
            snippet = doc.page_content[:200] + "..."
            
            citation_data = {
                "id": citation_id,
                "type": "web",
                "source": url,
                "content": doc.page_content
            }
            
            # Add image if available
            if "image" in doc.metadata and doc.metadata["image"]:
                image_path = doc.metadata["image"]
                if os.path.exists(image_path) and is_valid_image(image_path):
                    try:
                        with open(image_path, "rb") as img_file:
                            img_bytes = img_file.read()
                            img_b64 = base64.b64encode(img_bytes).decode()
                        citation_data["image_b64"] = img_b64
                    except Exception as e:
                        print(f"Error processing web image: {e}")
            
            citation_details.append(citation_data)
            citations.append(f"🌐 **Web Source**: {url[:50]}... - {snippet}")
            numbered_citations.append(f"[{i+1}] Web: {url[:30]}...")

    # Generate Answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    
    # Format prompt to include citations
    augmented_query = f"""
    Answer the following question based on the provided context.
    Question: {query}
    
    Make sure to include citation numbers [1], [2], etc. at the end of sentences where you use information from the sources.
    """
    
    response = qa_chain.run(augmented_query)

    return response, citations, citation_details, numbered_citations

# ✅ Setup Agentic RAG (Web Search + RAG)
def setup_agent(retriever):
    def rag_tool(q):
        answer, _, _, _ = answer_with_rag(q, retriever, [])
        return answer
    
    tools = [
        Tool(
            name="RAG Document Search",
            func=rag_tool,
            description="Searches the uploaded document for information. Use this when the question is about the document."
        ),
        Tool(
            name="Web Search",
            func=web_search_with_images,
            description="Searches the web for additional context or up-to-date information. Use this when information might not be in the document."
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools=tools,
        llm=ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )

    return agent

# ✅ Agent Answer (combines web search and document search)
def agent_answer(query, agent):
    with st.expander("Agent Reasoning", expanded=False):
        # Redirect stdout to capture agent's verbose output
        import io
        import sys
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            response = agent.run(input=f"""
            Answer the following question: {query}
            
            If the information is in the uploaded document, cite it with [1], [2], etc.
            If the information is from the web, provide the source URL.
            """)
        except Exception as e:
            response = f"Error during agent processing: {str(e)}"
        finally:
            sys.stdout = old_stdout
        
        # Display agent's thought process
        st.code(new_stdout.getvalue(), language="text")
    
    return response

# ✅ Custom CSS for Citations
def load_css():
    st.markdown("""
    <style>
    .citation-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .citation-text {
        margin-bottom: 10px;
    }
    .citation-image {
        max-width: 100%;
        height: auto;
        margin-top: 10px;
        border: 1px solid #ddd;
    }
    .citation-number {
        display: inline-block;
        background-color: #007bff;
        color: white;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        text-align: center;
        line-height: 25px;
        margin-right: 5px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

# ✅ Display image safely
def display_image_safely(image_path):
    try:
        if os.path.exists(image_path) and is_valid_image(image_path):
            st.image(image_path)
        else:
            st.warning(f"Unable to display image: invalid or missing file")
    except Exception as e:
        st.warning(f"Error displaying image: {str(e)}")

# ✅ Streamlit UI
def main():
    load_css()
    
    st.title("📄 MultiModal Agentic RAG with Interactive Citations")
    st.markdown("#### Upload a PDF, ask questions, and get answers grounded in both document content and web search")
    
    # Initialize session state
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'image_data' not in st.session_state:
        st.session_state.image_data = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'citation_details' not in st.session_state:
        st.session_state.citation_details = []

    # Tabs for different modes
    tab1, tab2 = st.tabs(["📄 Upload & Process PDF", "🔍 Ask Questions"])

    with tab1:
        uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
        if uploaded_file:
            try:
                pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                total_pages = len(pdf_doc)

                # Page Selection UI
                st.write(f"📄 **Total Pages in PDF:** {total_pages}")
                col1, col2 = st.columns(2)
                with col1:
                    start_page = st.number_input("📌 Start Page", min_value=1, max_value=total_pages, value=1)
                with col2:
                    end_page = st.number_input("📌 End Page", min_value=1, max_value=total_pages, value=min(5, total_pages))

                st.info(f"You've selected to process pages {start_page} to {end_page} (total: {end_page-start_page+1} pages)")

                if st.button("🔍 Process Selected Pages", use_container_width=True):
                    progress_bar = st.progress(0)
                    
                    try:
                        # Step 1: Process PDF with Mistral OCR
                        progress_bar.progress(10, text="Step 1/4: Uploading PDF to Mistral OCR...")
                        pdf_bytes = uploaded_file.getvalue()
                        ocr_data = process_pdf_with_mistral(pdf_bytes, uploaded_file.name, start_page, end_page)
                        
                        if ocr_data is None:
                            st.error("Failed to process PDF with Mistral OCR. Please try again.")
                            return
                        
                        # Step 2: Extract text and images
                        progress_bar.progress(40, text="Step 2/4: Extracting text and images...")
                        extracted_data, image_data = extract_text_and_images(ocr_data)
                        
                        if not extracted_data:
                            st.warning("No text or image data could be extracted from the PDF.")
                        
                        # Step 3: Build vector store and retriever
                        progress_bar.progress(70, text="Step 3/4: Building vector embeddings...")
                        retriever, vector_store = build_rag_system(extracted_data)
                        st.session_state.retriever = retriever
                        st.session_state.vector_store = vector_store
                        st.session_state.image_data = image_data
                        
                        # Step 4: Initialize agent
                        progress_bar.progress(90, text="Step 4/4: Setting up AI agent...")
                        st.session_state.agent = setup_agent(retriever)
                        
                        progress_bar.progress(100, text="Complete!")
                        st.success(f"✅ PDF processed successfully! Extracted {len(extracted_data)} text chunks and {len(image_data)} images.")
                        
                        # Display sample of extracted text
                        if extracted_data:
                            with st.expander("📄 Sample of extracted text"):
                                for i, doc in enumerate(extracted_data[:3]):
                                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page')}):**")
                                    st.text(doc.page_content[:200] + "...")
                        
                        # Display sample images
                        if image_data:
                            with st.expander("🖼️ Sample of extracted images"):
                                for i, (page_num, img_path, img_text) in enumerate(image_data[:3]):
                                    st.markdown(f"**Image from Page {page_num}:**")
                                    display_image_safely(img_path)
                                    if img_text:
                                        st.text(f"OCR Text: {img_text}")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        print(traceback.format_exc())
            except Exception as e:
                st.error(f"Error opening PDF file: {str(e)}")

    with tab2:
        if st.session_state.retriever is not None:
            st.markdown("### 🤖 Ask a question about the document")
            
            # Select mode
            answer_mode = st.radio(
                "Select answer mode:",
                ["Document RAG Only", "Document RAG + Web Search"],
                horizontal=True
            )
            
            # Query input
            query = st.text_input("🔍 Your question:")
            
            if query:
                try:
                    with st.spinner("🧠 Thinking..."):
                        if answer_mode == "Document RAG Only":
                            # RAG-only mode
                            response, citations, citation_details, numbered_citations = answer_with_rag(
                                query, st.session_state.retriever, st.session_state.image_data
                            )
                            st.session_state.citation_details = citation_details
                        else:
                            # Agent mode (RAG + Web Search)
                            agent_response = agent_answer(query, st.session_state.agent)
                            
                            # Then get citations from RAG for proper display
                            response, citations, citation_details, numbered_citations = answer_with_rag(
                                query, st.session_state.retriever, st.session_state.image_data
                            )
                            st.session_state.citation_details = citation_details
                            
                            # Use the agent's response as it includes web search
                            response = agent_response

                    # Display the answer
                    st.markdown("### 💡 Answer:")
                    st.write(response, unsafe_allow_html=True)
                    
                    # Add interactive citation numbers to the text
                    processed_response = response
                    for i in range(len(st.session_state.citation_details)):
                        citation_num = f"[{i+1}]"
                        if citation_num in processed_response:
                            processed_response = processed_response.replace(
                                citation_num, 
                                f'<span class="citation-number" onclick="showCitation{i+1}()">{i+1}</span>'
                            )
                    
                    # Display citations
                    st.markdown("### 📌 Citations & Sources:")
                    
                    # JavaScript for toggling citation visibility
                    js_code = "<script>\n"
                    
                    for i, citation in enumerate(st.session_state.citation_details):
                        citation_id = f"citation_{i+1}"
                        
                        # Begin citation container
                        st.markdown(f'<div id="{citation_id}" class="citation-box">', unsafe_allow_html=True)
                        
                        # Display citation source info
                        if citation.get("type") == "web":
                            st.markdown(f'<h4>Web Source [{i+1}]:</h4>', unsafe_allow_html=True)
                            st.markdown(f'<a href="{citation.get("source")}" target="_blank">{citation.get("source")}</a>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h4>Source [{i+1}]: {citation.get("source")}</h4>', unsafe_allow_html=True)
                        
                        # Display citation content
                        if citation.get("content"):
                            st.markdown('<div class="citation-text">', unsafe_allow_html=True)
                            st.write(citation.get("content"))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display citation image if available
                        if "image_b64" in citation:
                            try:
                                image_bytes = base64.b64decode(citation["image_b64"])
                                # Verify it's a valid image
                                Image.open(io.BytesIO(image_bytes))
                                st.markdown(f'<img src="data:image/png;base64,{citation["image_b64"]}" class="citation-image">', unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"Unable to display image in citation: {str(e)}")
                        
                        # End citation container
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add JavaScript toggle function
                        js_code += f"""
                        function showCitation{i+1}() {{
                            var citations = document.querySelectorAll('.citation-box');
                            for (var i = 0; i < citations.length; i++) {{
                                citations[i].style.display = 'none';
                            }}
                            document.getElementById('{citation_id}').style.display = 'block';
                            
                            // Scroll to the citation
                            document.getElementById('{citation_id}').scrollIntoView({{behavior: 'smooth'}});
                        }}
                        """
                    
                    js_code += "</script>"
                    st.markdown(js_code, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    print(traceback.format_exc())
        else:
            st.warning("⚠️ Please upload and process a PDF document first in the 'Upload & Process PDF' tab.")

if __name__ == "__main__":
    main()
