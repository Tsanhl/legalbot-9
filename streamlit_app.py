"""
Legal AI - Streamlit Application
AI-powered legal research assistant with OSCOLA citations
"""
import streamlit as st
import json
import base64
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

# Import services
from knowledge_base import load_law_resource_index, get_knowledge_base_summary
from gemini_service import (
    initialize_knowledge_base, 
    send_message_with_docs, 
    reset_session,
    encode_file_to_base64
)

# RAG Service for document content retrieval
try:
    from rag_service import get_rag_service, RAGService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for legal styling with proper edge effects (NOT sticking to edges)
st.markdown("""
<style>
/* Import Google-like fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Product+Sans:wght@400;700&display=swap');

/* Google AI Studio-inspired Clean Theme */
:root {
    --bg-color: #ffffff;
    --sidebar-bg: #f9fafe; /* Very light gray/blue tint */
    --text-primary: #1f1f1f;
    --text-secondary: #3c4043; /* Darker gray for better visibility */
    --accent-blue: #1a73e8;
    --border-color: #e0e0e0;
    --card-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    --hover-bg: #f1f3f4;
}

/* Force full opacity for sidebar elements to prevent fading when busy */
section[data-testid="stSidebar"] {
    opacity: 1 !important;
}

section[data-testid="stSidebar"] * {
    transition: none !important; /* Remove fade transition */
}

/* Ensure text is always dark in sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* Specific fix for file uploader "ghosting" */
[data-testid="stFileUploader"], 
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div {
    opacity: 1 !important;
    color: var(--text-primary) !important;
}

/* Prevent blur/darken overlay on main content */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container {
    opacity: 1 !important;
    filter: none !important;
    backdrop-filter: none !important;
}

/* Remove any modal overlay effects */
[data-testid="stModal"],
.stModal {
    background: transparent !important;
    backdrop-filter: none !important;
}

/* Ensure main area never gets dimmed */
section[data-testid="stMain"] {
    opacity: 1 !important;
    filter: none !important;
}

/* Global Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3 {
    font-family: 'Product Sans', 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Sidebar Styling - Light Clean Look */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] > div:first-child {
    background-color: var(--sidebar-bg);
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p, 
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

/* Input Fields - Google Style */
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 8px;
    color: var(--text-primary);
    padding: 0.75rem;
    transition: all 0.2s;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(26,115,232,0.2);
}

/* Buttons - Primary & Secondary */
.stButton button {
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.2s;
    border: none;
    box-shadow: none;
}

/* Force Primary Buttons to Google Blue */
div.stButton > button[kind="primary"] {
    background-color: #1a73e8 !important;
    color: white !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #1557b0 !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.3) !important;
}

/* Secondary Buttons */
div.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #1a73e8 !important;
    border: 1px solid #dadce0 !important;
}

div.stButton > button[kind="secondary"]:hover {
    background-color: #f1f3f4 !important;
    border-color: #1a73e8 !important;
}

/* Vertically center buttons in sidebar columns */
section[data-testid="stSidebar"] [data-testid="column"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

section[data-testid="stSidebar"] [data-testid="column"] > div {
    width: 100%;
}

/* File Uploader - Specific Fix for Black Text */
[data-testid="stFileUploader"] {
    padding: 1rem;
    border: 1px dashed #dadce0;
    border-radius: 8px;
    background: white;
}

[data-testid="stFileUploader"] section {
    background-color: #f8f9fa !important;
}

/* Fix font size and family for uploader text - ALL SAME */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 0.875rem !important; /* 14px - same as sidebar labels */
    font-weight: 400 !important; /* Normal weight for all */
    line-height: 1.5 !important;
    color: #202124 !important;
}

/* Make "Browse files" button slightly different for visibility */
[data-testid="stFileUploader"] button {
    color: #202124 !important;
    border-color: #dadce0 !important;
    background-color: #ffffff !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important; /* Slightly bolder for button */
}

/* Custom Lists (React Style) */
.custom-list-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.blue-dot {
    width: 0.375rem;
    height: 0.375rem;
    background-color: var(--accent-blue);
    border-radius: 9999px;
    flex-shrink: 0;
}

/* Chips for Suggestions */
.suggestion-chip {
    padding: 0.75rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: var(--text-primary);
    display: block; /* Ensure full width block */
    text-decoration: none;
}

.suggestion-chip:hover {
    background-color: #e8f0fe;
}

/* Project Cards - Clean & Minimal */
.project-card {
    background-color: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.project-card:hover {
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transform: translateY(-1px);
}

.project-card.active {
    background-color: #e8f0fe; /* Light blue selection */
    border-color: var(--accent-blue);
    color: var(--accent-blue);
}

/* Chat Messages */
.chat-message {
    padding: 1rem 0;
}

.chat-bubble {
    padding: 16px 20px;
    border-radius: 18px;
    line-height: 1.5;
    font-size: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.chat-bubble.user {
    background-color: #e8f0fe; /* Google Blue Tint */
    color: #1a73e8;
    border-bottom-right-radius: 4px;
}

.chat-bubble.assistant {
    background-color: #ffffff;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}

/* Sidebar Section Headers */
.sidebar-section {
    font-size: 13px; /* Slightly larger for readability */
    font-weight: 600;
    color: var(--text-primary) !important; /* Force dark color */
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Capabilities & Tips Boxes - Google Style Cards */
.big-box {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #dadce0;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #bdc1c6;
}

/* Modal/Overlay Fixes */
div[data-baseweb="modal"], div[class*="backdrop"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# Constants
MAX_PROJECTS = 10

# Initialize session state
def init_session_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = [{
            'id': str(uuid.uuid4()),
            'name': 'Default Project',
            'messages': [],
            'documents': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'cross_memory': False
        }]
    
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = st.session_state.projects[0]['id']
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
        st.session_state.kb_count = 0
        st.session_state.kb_categories = []
    
    if 'active_citation' not in st.session_state:
        st.session_state.active_citation = None
    
    if 'input_value' not in st.session_state:
        st.session_state.input_value = ''
    
    if 'renaming_project_id' not in st.session_state:
        st.session_state.renaming_project_id = None
    
    if 'rag_indexing' not in st.session_state:
        st.session_state.rag_indexing = False
    
    if 'rag_stats' not in st.session_state:
        st.session_state.rag_stats = None
    
    if 'rag_indexed' not in st.session_state:
        st.session_state.rag_indexed = False
    
    if 'rag_chunk_count' not in st.session_state:
        st.session_state.rag_chunk_count = 0
    
    if 'auto_index_triggered' not in st.session_state:
        st.session_state.auto_index_triggered = False

def get_current_project() -> Optional[Dict]:
    """Get the current project"""
    for p in st.session_state.projects:
        if p['id'] == st.session_state.current_project_id:
            return p
    return None

def create_new_project(name: str = None) -> Dict:
    """Create a new project"""
    return {
        'id': str(uuid.uuid4()),
        'name': name or f"Project {datetime.now().strftime('%Y-%m-%d')}",
        'messages': [],
        'documents': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'cross_memory': False
    }

def parse_citations(text: str) -> str:
    """Parse citation JSON and convert to HTML buttons"""
    pattern = r'\[\[\{.*?\}\]\]'
    
    def replace_citation(match):
        try:
            json_str = match.group(0)[2:-2]  # Remove [[ and ]]
            citation = json.loads(json_str)
            ref = citation.get('ref', 'Citation')
            # Format in proper OSCOLA style - just the reference in brackets
            return f'({ref})'
        except:
            return match.group(0)
    
    return re.sub(pattern, replace_citation, text)

def render_message(message: Dict, is_user: bool):
    """Render a chat message"""
    bubble_class = "user" if is_user else "assistant"
    
    # Clean text (remove ** and * markdown)
    text = message.get('text', '')
    text = text.replace('**', '').replace('*', '')
    
    # Parse citations
    text_with_citations = parse_citations(text)
    
    if is_user:
        # User message with label
        st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-bubble user">
                <div class="chat-role user">You</div>
                <div class="chat-text">{text_with_citations}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message - no label, just clean response
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-bubble assistant">
                <div class="chat-text">{text_with_citations}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Load knowledge base on startup
    if not st.session_state.knowledge_base_loaded:
        index = load_law_resource_index()
        if index:
            st.session_state.knowledge_base_loaded = True
            st.session_state.kb_count = index.totalFiles
            st.session_state.kb_categories = index.categories
            initialize_knowledge_base()
    
    # ===== SIDEBAR =====
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <span style="color: #1a73e8; font-size: 1.25rem;">⚖️</span>
            <h1 style="color: #202124; font-family: 'Product Sans', sans-serif;">Legal AI</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration Section
        st.markdown('<div class="sidebar-section">⚙️ Configuration</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Gemini API Key (Optional)",
            value=st.session_state.api_key,
            type="password",
            placeholder="Enter Key or use Default...",
            help="Leave empty to use the default system key."
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Projects Section - Header and New button on same line
        col_header, col_new = st.columns([3, 1])
        with col_header:
            st.markdown(f'<div class="sidebar-section" style="display: flex; align-items: center; height: 38px; margin: 0;">📁 Projects ({len(st.session_state.projects)}/{MAX_PROJECTS})</div>', unsafe_allow_html=True)
        with col_new:
            if st.button("New", disabled=len(st.session_state.projects) >= MAX_PROJECTS, key="new_project_btn", use_container_width=True):
                new_project = create_new_project()
                st.session_state.projects.insert(0, new_project)
                st.session_state.current_project_id = new_project['id']
                st.rerun()
        
        # Project list with rename functionality
        for project in st.session_state.projects:
            is_active = project['id'] == st.session_state.current_project_id
            is_renaming = st.session_state.renaming_project_id == project['id']
            
            col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
            
            with col1:
                if is_renaming:
                    # Show text input when renaming
                    new_name = st.text_input(
                        "New name",
                        value=project['name'],
                        key=f"rename_{project['id']}",
                        label_visibility="collapsed"
                    )
                else:
                    # Show project button
                    if st.button(
                        project['name'],
                        key=f"proj_{project['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_project_id = project['id']
                        st.rerun()
            
            with col2:
                if is_renaming:
                    # Save button when renaming
                    if st.button("✓", key=f"save_{project['id']}", help="Save"):
                        new_name = st.session_state.get(f"rename_{project['id']}", project['name'])
                        if new_name.strip():
                            project['name'] = new_name.strip()
                        st.session_state.renaming_project_id = None
                        st.rerun()
                else:
                    # Rename button (pencil icon)
                    if st.button("✎", key=f"rename_btn_{project['id']}", help="Rename"):
                        st.session_state.renaming_project_id = project['id']
                        st.rerun()
            
            with col3:
                # Cross memory toggle
                icon = "🔗" if project.get('cross_memory') else "⛓️"
                if st.button(icon, key=f"mem_{project['id']}", help="Toggle cross-memory"):
                    project['cross_memory'] = not project.get('cross_memory', False)
                    st.rerun()
            
            with col4:
                # Delete button
                if len(st.session_state.projects) > 1:
                    if st.button("✕", key=f"del_{project['id']}", help="Delete project"):
                        st.session_state.projects = [p for p in st.session_state.projects if p['id'] != project['id']]
                        if st.session_state.current_project_id == project['id']:
                            st.session_state.current_project_id = st.session_state.projects[0]['id']
                        reset_session(project['id'])
                        st.rerun()
        
        st.caption("Double-click project name or click ✎ to rename. 🔗 = share memory across projects.")
        
        st.markdown("---")
        
        # Research Materials Section
        st.markdown('<div class="sidebar-section">📚 Research Materials</div>', unsafe_allow_html=True)
        
        # Link input
        link_url = st.text_input("Add Web Reference (URL)", placeholder="https://...")
        if st.button("Add URL", use_container_width=True):
            if link_url:
                current_project = get_current_project()
                if current_project:
                    url = link_url if link_url.startswith('http') else f'https://{link_url}'
                    current_project['documents'].append({
                        'id': str(uuid.uuid4()),
                        'type': 'link',
                        'name': url,
                        'mimeType': 'text/uri-list',
                        'data': url,
                        'size': 0
                    })
                    st.rerun()
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=['pdf', 'txt', 'md', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            current_project = get_current_project()
            if current_project:
                files_added = False
                for file in uploaded_files:
                    # Check if file already added
                    existing_names = [d['name'] for d in current_project['documents']]
                    if file.name not in existing_names:
                        content = file.read()
                        current_project['documents'].append({
                            'id': str(uuid.uuid4()),
                            'type': 'file',
                            'name': file.name,
                            'mimeType': file.type or 'application/octet-stream',
                            'data': encode_file_to_base64(content),
                            'size': len(content)
                        })
                        files_added = True
                # Only rerun after processing all files
                if files_added:
                    st.rerun()
        
        st.markdown("---")
        
        # ===== KNOWLEDGE BASE ACTIVE SECTION =====
        # This section handles auto-indexing for Streamlit Cloud deployment
        if RAG_AVAILABLE:
            try:
                rag_service = get_rag_service()
                stats = rag_service.get_stats()
                
                # Check if we need to auto-index (first deployment or empty database)
                resources_path = os.path.join(os.path.dirname(__file__), 'Law resouces  copy 2')
                
                # Auto-index on first startup if database is empty
                if stats['total_chunks'] == 0 and not st.session_state.auto_index_triggered and os.path.exists(resources_path):
                    st.session_state.auto_index_triggered = True
                    st.session_state.rag_indexing = True
                    st.rerun()
                
                # Show indexing progress if currently indexing
                if st.session_state.rag_indexing:
                    st.markdown('<div class="sidebar-section">📚 Knowledge Base</div>', unsafe_allow_html=True)
                    st.info("⏳ Auto-indexing law documents... Please wait.")
                    
                    with st.spinner("Indexing documents... This may take a few minutes on first startup."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(count, filename):
                            progress_bar.progress(min(count / 500, 1.0))  # Estimate ~500 files
                            status_text.text(f"Processing: {filename[:40]}...")
                        
                        try:
                            result = rag_service.index_documents(resources_path, progress_callback)
                            st.session_state.rag_stats = result
                            st.session_state.rag_indexed = True
                            st.session_state.rag_chunk_count = result['chunks']
                            st.session_state.rag_indexing = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Indexing error: {str(e)}")
                            st.session_state.rag_indexing = False
                else:
                    # Show Knowledge Base Active status
                    st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
                    
                    if stats['total_chunks'] > 0:
                        st.success(f"✅ {stats['total_chunks']} text chunks indexed")
                        st.caption("The AI can now search inside your law documents!")
                    else:
                        st.caption("No documents added. AI will use knowledge base and Google Search.")
                        
                        # Show manual index button if no documents indexed
                        if os.path.exists(resources_path):
                            if st.button("🔄 Index Law Documents", use_container_width=True, help="Extract and index text from all law resources"):
                                st.session_state.rag_indexing = True
                                st.rerun()
                
            except Exception as e:
                st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
                st.caption("No documents added. AI will use knowledge base and Google Search.")
        else:
            # RAG not available - show basic Knowledge Base status  
            st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
            if st.session_state.knowledge_base_loaded:
                st.caption("AI will use knowledge base and Google Search.")
            else:
                st.caption("No documents added. AI will use knowledge base and Google Search.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0;">
            <div class="ai-badge">AI</div>
            <div>
                <div style="font-size: 0.875rem; font-weight: 500; color: #202124;">Gemini 3 Pro</div>
                <div style="font-size: 0.75rem; color: #5f6368;">""" + ("Custom Key Active" if st.session_state.api_key else "Default Key Active") + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN AREA =====
    current_project = get_current_project()
    
    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("### 📖 Legal Research Workspace")
    with col2:
        if st.button("Clear", type="secondary"):
            if current_project:
                current_project['messages'] = []
                reset_session(current_project['id'])
                st.rerun()
    
    st.markdown("---")
    
    # Chat area
    if current_project:
        messages = current_project.get('messages', [])
        
        # Check if there are any messages - if yes, show chat only
        if len(messages) > 0:
            # Display existing messages - NO BOXES
            for msg in messages:
                is_user = msg.get('role') == 'user'
                render_message(msg, is_user)
        else:
            # EMPTY STATE - Show welcome screen with boxes
            st.markdown("""
            <div style="text-align: center; max-width: 40rem; margin: 3rem auto; padding: 2rem;">
                <div style="font-size: 4rem; color: #dadce0; margin-bottom: 1rem;">📚</div>
                <h2 style="font-family: 'Product Sans', sans-serif; font-size: 2rem; color: #202124; margin-bottom: 0.5rem;">Legal AI</h2>
                <p style="color: #5f6368; font-size: 1rem; margin-bottom: 2rem;">AI-powered legal research assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Knowledge Base Status
            col1, col2, col3 = st.columns([1, 2, 1])
            if st.session_state.knowledge_base_loaded:
                with col2:
                    st.success("✅ Knowledge Base Active")
            
            # Centered content - BIGGER BOXES with DARKER TEXT
            with col2:
                st.markdown('<p style="color: #202124; font-size: 1.25rem; font-weight: 500; text-align: center; margin: 2rem 0;">Just ask your question</p>', unsafe_allow_html=True)
                
                # Capabilities box - React Style (Blue Dots)
                st.markdown("""
                <div style="background: white; border: 1px solid #dadce0; border-radius: 0.75rem; padding: 2rem; margin: 1.5rem 0; text-align: left; box-shadow: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);">
                    <h4 style="font-size: 0.75rem; font-weight: 700; color: #5f6368; text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.5px;">Capabilities</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div class="custom-list-item"><div class="blue-dot"></div>Essay Writing</div>
                        <div class="custom-list-item"><div class="blue-dot"></div>Problem Questions</div>
                        <div class="custom-list-item"><div class="blue-dot"></div>Legal Advice & Strategy</div>
                        <div class="custom-list-item"><div class="blue-dot"></div>General Queries</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tips box - React Style (Chips)
                st.markdown("""
                <div style="background: white; border: 1px solid #dadce0; border-radius: 0.75rem; padding: 2rem; margin: 1.5rem 0; text-align: left; box-shadow: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);">
                    <h4 style="font-size: 0.75rem; font-weight: 700; color: #5f6368; margin-bottom: 1rem; letter-spacing: 0.5px; text-transform: uppercase; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #eab308; font-size: 1rem;">✨</span> Try Asking
                    </h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <div class="suggestion-chip">"What are the key elements of a valid contract under English law?"</div>
                        <div class="suggestion-chip">"Explain the duty of care in negligence under UK tort law"</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area - Docked at bottom (st.chat_input)
    if prompt := st.chat_input("Ask for an Essay, Case Analysis, or Client Advice..."):
        if current_project:
            # Add user message immediately
            user_message = {
                'id': str(uuid.uuid4()),
                'role': 'user',
                'text': prompt,
                'timestamp': datetime.now().isoformat()
            }
            current_project['messages'].append(user_message)
            
            # Display user message immediately
            render_message(user_message, is_user=True)
            
            # Get API key
            api_key = st.session_state.api_key or os.environ.get('GEMINI_API_KEY', '')
            
            if not api_key:
                st.error("Please enter a Gemini API key in the sidebar configuration.")
            else:
                # Show thinking indicator
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("""
                <div class="chat-message assistant">
                    <div class="chat-bubble assistant" style="display: flex; align-items: center; gap: 8px;">
                        <div style="display: flex; gap: 4px;">
                            <span style="animation: pulse 1s infinite; opacity: 0.6;">●</span>
                            <span style="animation: pulse 1s infinite 0.2s; opacity: 0.6;">●</span>
                            <span style="animation: pulse 1s infinite 0.4s; opacity: 0.6;">●</span>
                        </div>
                        <span style="color: #5f6368; font-style: italic;">Thinking...</span>
                    </div>
                </div>
                <style>
                @keyframes pulse {
                    0%, 100% { opacity: 0.3; }
                    50% { opacity: 1; }
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Stream the response for faster display
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Use streaming for faster response
                    stream = send_message_with_docs(
                        api_key,
                        prompt,
                        current_project.get('documents', []),
                        current_project['id'],
                        stream=True
                    )
                    
                    # Clear thinking indicator once we start getting response
                    first_chunk = True
                    
                    # Stream the response chunks
                    for chunk in stream:
                        if hasattr(chunk, 'text'):
                            if first_chunk:
                                thinking_placeholder.empty()
                                first_chunk = False
                            
                            full_response += chunk.text
                            # Clean and display progressively
                            display_text = full_response.replace('**', '').replace('*', '')
                            response_placeholder.markdown(f"""
                            <div class="chat-message assistant">
                                <div class="chat-bubble assistant">
                                    <div class="chat-text">{display_text}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Clear placeholders
                    thinking_placeholder.empty()
                    response_placeholder.empty()
                    
                    # Add assistant message
                    assistant_message = {
                        'id': str(uuid.uuid4()),
                        'role': 'assistant',
                        'text': full_response,
                        'timestamp': datetime.now().isoformat(),
                        'grounding_urls': []
                    }
                    current_project['messages'].append(assistant_message)
                    
                except Exception as e:
                    thinking_placeholder.empty()
                    response_placeholder.empty()
                    # Add error message
                    error_message = {
                        'id': str(uuid.uuid4()),
                        'role': 'assistant',
                        'text': f"I encountered an error: {str(e)}",
                        'timestamp': datetime.now().isoformat(),
                        'is_error': True
                    }
                    current_project['messages'].append(error_message)
            
            st.rerun()

if __name__ == "__main__":
    main()

