import os
import asyncio
import base64
import fitz  # PyMuPDF
from typing import List, Dict
from dotenv import load_dotenv
import streamlit as st
from supabase import create_client, Client
from google import genai
from pypdf import PdfReader

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 5))

# Page config
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-first responsive CSS
st.markdown("""
<style>
    /* Mobile-first base styles */
    .main .block-container {
        padding: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Responsive breakpoints */
    @media (min-width: 768px) {
        .main .block-container {
            padding: 2rem !important;
        }
    }
    
    /* Full-width buttons on mobile */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Chat input styling */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: white;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }
    
    @media (min-width: 768px) {
        .stChatInput {
            position: relative;
            border-top: none;
            padding: 0;
        }
    }
    
    /* Source chunk highlight */
    .source-chunk {
        background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
        overflow-x: auto;
        word-wrap: break-word;
    }
    
    /* Source reference badge */
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* PDF viewer container */
    .pdf-container {
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        overflow: hidden;
        background: #f8f9fa;
    }
    
    /* Page navigation */
    .page-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Status indicator */
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat message styling */
    .stChatMessage {
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Responsive columns on mobile */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }
        
        .row-widget {
            flex-direction: column !important;
        }
    }
</style>
""", unsafe_allow_html=True)


class GeminiRAG:
    """RAG system using Gemini embeddings and Supabase vector storage"""
    
    def __init__(self):
        # Fix for Streamlit thread loop issue with google-genai
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        if not SUPABASE_URL or not SUPABASE_KEY:
            st.error("⚠️ Set SUPABASE_URL and SUPABASE_KEY in .env")
            st.stop()
        if not GEMINI_API_KEY:
            st.error("⚠️ Set GEMINI_API_KEY in .env")
            st.stop()
            
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.genai = genai.Client(api_key=GEMINI_API_KEY)
    
    def split_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence/paragraph boundary
            if end < len(text):
                for sep in ['\n\n', '\n', '. ', '! ', '? ', ', ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:
                        chunk = chunk[:last_sep + len(sep)]
                        break
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = start + len(chunk) - overlap
            if start >= len(text) - overlap:
                break
        
        return chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using Gemini (768 dimensions)"""
        # Use text-embedding-004 with explicit dimensionality
        from google.genai import types
        response = self.genai.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    
    def clear_all_chunks(self):
        """Delete all existing chunks"""
        self.supabase.table('document_chunks').delete().neq('id', 0).execute()
    
    def get_chunk_count(self) -> int:
        """Get number of stored chunks"""
        result = self.supabase.table('document_chunks').select('id', count='exact').execute()
        return result.count or 0
    
    def get_max_chunk_index(self) -> int:
        """Get maximum chunk index for appending"""
        result = self.supabase.table('document_chunks').select('chunk_index').order(
            'chunk_index', desc=True
        ).limit(1).execute()
        if result.data:
            return result.data[0]['chunk_index']
        return -1
    
    def process_pdf(self, pdf_path: str, append: bool = False) -> int:
        """Process PDF and store embeddings in Supabase"""
        
        if not append:
            self.clear_all_chunks()
            start_index = 0
        else:
            start_index = self.get_max_chunk_index() + 1
        
        # Load PDF using pypdf
        reader = PdfReader(pdf_path)
        
        chunks_data = []
        chunk_index = start_index
        
        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            page_chunks = self.split_text(page_text)
            
            for chunk_text in page_chunks:
                # Find position in page
                start_char = page_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else page_text.find(chunk_text)
                end_char = start_char + len(chunk_text) if start_char != -1 else len(chunk_text)
                
                chunks_data.append({
                    'text': chunk_text,
                    'index': chunk_index,
                    'page': page_num,
                    'start': max(0, start_char),
                    'end': end_char
                })
                chunk_index += 1
        
        # Create embeddings and store
        total = len(chunks_data)
        progress = st.progress(0)
        status = st.empty()
        
        for i, chunk in enumerate(chunks_data):
            embedding = self.create_embedding(chunk['text'])
            
            self.supabase.table('document_chunks').insert({
                'chunk_text': chunk['text'],
                'chunk_index': chunk['index'],
                'page_number': chunk['page'],
                'start_char': chunk['start'],
                'end_char': chunk['end'],
                'embedding': embedding
            }).execute()
            
            progress.progress((i + 1) / total)
            status.text(f"Processing: {i + 1}/{total} chunks")
        
        progress.empty()
        status.empty()
        
        return total
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Search for similar chunks"""
        query_embedding = self.create_embedding(query)
        
        result = self.supabase.rpc('match_chunks', {
            'query_embedding': query_embedding,
            'match_threshold': 0.3,
            'match_count': top_k
        }).execute()
        
        return result.data or []
    
    def generate(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer with citations"""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] (Page {chunk['page_number']}): {chunk['chunk_text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer based ONLY on this context. Cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Provide a clear, comprehensive answer with citations."""
        
        response = self.genai.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text


def render_pdf_page(pdf_path: str, page_num: int, highlight_text: str = None) -> bytes:
    """Render PDF page with optional highlighting"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    if highlight_text and len(highlight_text) > 20:
        # Search for text to highlight
        search_text = highlight_text[:100]
        instances = page.search_for(search_text)
        for inst in instances[:3]:
            highlight = page.add_highlight_annot(inst)
            highlight.set_colors(stroke=[1, 1, 0])
            highlight.update()
    
    # Render at 2x for quality
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total pages in PDF"""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def main():
    # Initialize session state
    if "rag" not in st.session_state:
        st.session_state.rag = GeminiRAG()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "highlight" not in st.session_state:
        st.session_state.highlight = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    
    # PDF path
    pdf_path = os.path.join(os.path.dirname(__file__), "uploaded.pdf")
    
    # Header
    st.title("📄 PDF RAG Assistant")
    st.caption("Ask questions • Get answers with sources • View highlighted chunks")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document")
        
        # Upload section
        uploaded = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        if uploaded:
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("✅ Uploaded!")
            st.session_state.processed = False
            st.rerun()
        
        # Check if PDF exists
        pdf_exists = os.path.exists(pdf_path)
        
        if pdf_exists:
            st.markdown(f'<span class="status-pill status-success">✓ uploaded.pdf loaded</span>', unsafe_allow_html=True)
            
            # Get chunk count
            chunk_count = st.session_state.rag.get_chunk_count()
            if chunk_count > 0:
                st.session_state.processed = True
            
            st.markdown(f'<span class="status-pill status-info">📊 {chunk_count} chunks in DB</span>', unsafe_allow_html=True)
            
            # Process buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Process", use_container_width=True, help="Replace all embeddings"):
                    with st.spinner("Processing..."):
                        count = st.session_state.rag.process_pdf(pdf_path, append=False)
                    st.success(f"✅ {count} chunks!")
                    st.session_state.processed = True
                    st.rerun()
            
            with col2:
                if st.button("➕ Append", use_container_width=True, help="Add to existing"):
                    with st.spinner("Appending..."):
                        count = st.session_state.rag.process_pdf(pdf_path, append=True)
                    st.success(f"✅ +{count} chunks!")
                    st.rerun()
        else:
            st.warning("⚠️ Upload a PDF to start")
        
        # Settings
        st.markdown("---")
        st.caption(f"**Model:** Gemini Embedding (768-dim)")
        st.caption(f"**Chunks:** {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    
    # Main content
    if not pdf_exists:
        st.info("👈 Upload a PDF file to get started")
        return
    
    if not st.session_state.processed:
        st.warning("👈 Click **Process** to create embeddings")
    
    # Detect mobile
    is_mobile = st.checkbox("📱 Mobile view", value=False, help="Single column layout")
    
    if is_mobile:
        # Mobile layout - single column
        render_chat_section(pdf_path)
        st.markdown("---")
        render_sources_section()
        st.markdown("---")
        render_pdf_section(pdf_path)
    else:
        # Desktop layout - two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_chat_section(pdf_path)
            render_sources_section()
        
        with col2:
            render_pdf_section(pdf_path)


def render_chat_section(pdf_path: str):
    """Render chat interface"""
    st.subheader("💬 Chat")
    
    # Messages
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input
    query = st.chat_input("Ask about the PDF...")
    
    if query and st.session_state.processed:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.spinner("🔍 Searching & generating..."):
            chunks = st.session_state.rag.search(query)
            
            if chunks:
                answer = st.session_state.rag.generate(query, chunks)
                st.session_state.sources = chunks
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No relevant information found in the document."
                })
                st.session_state.sources = []
        
        st.rerun()
    elif query and not st.session_state.processed:
        st.warning("Process the PDF first!")


def render_sources_section():
    """Render source references"""
    if not st.session_state.sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, chunk in enumerate(st.session_state.sources, 1):
        similarity = chunk.get('similarity', 0)
        page = chunk['page_number']
        
        with st.expander(f"[{i}] Page {page} • {similarity:.0%} match", expanded=False):
            st.markdown(f'<div class="source-chunk">{chunk["chunk_text"]}</div>', unsafe_allow_html=True)
            
            if st.button(f"📖 View Page {page}", key=f"view_{i}", use_container_width=True):
                st.session_state.page = page
                st.session_state.highlight = chunk["chunk_text"]
                st.rerun()


def render_pdf_section(pdf_path: str):
    """Render PDF viewer with navigation"""
    if not os.path.exists(pdf_path):
        return
    
    total_pages = get_pdf_page_count(pdf_path)
    
    st.subheader(f"📄 PDF Viewer")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.page <= 1):
            st.session_state.page -= 1
            st.session_state.highlight = None
            st.rerun()
    
    with col2:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.page,
            label_visibility="collapsed"
        )
        if new_page != st.session_state.page:
            st.session_state.page = new_page
            st.session_state.highlight = None
            st.rerun()
    
    with col3:
        if st.button("Next ▶", use_container_width=True, disabled=st.session_state.page >= total_pages):
            st.session_state.page += 1
            st.session_state.highlight = None
            st.rerun()
    
    st.caption(f"Page {st.session_state.page} of {total_pages}")
    
    # Render PDF
    try:
        img_bytes = render_pdf_page(pdf_path, st.session_state.page, st.session_state.highlight)
        st.image(img_bytes, use_container_width=True)
        
        if st.session_state.highlight:
            st.info("🔍 Text highlighted from source reference")
    except Exception as e:
        # Fallback to iframe
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={st.session_state.page}" '
            f'width="100%" height="600px" style="border:1px solid #ddd;border-radius:8px;"></iframe>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
