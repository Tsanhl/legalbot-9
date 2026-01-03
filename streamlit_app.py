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
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
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

/* Chat Role & Text */
.chat-role {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
    color: #5f6368;
}

.chat-role.user { color: #1a73e8; }
.chat-role.assistant { color: #5f6368; }

.chat-text {
    white-space: pre-wrap;
    word-break: break-word;
}

/* Ghost Button for Editing */
.ghost-btn {
    border: none !important;
    background: transparent !important;
    color: #dadce0 !important;
    font-size: 14px !important;
    cursor: pointer;
    padding: 2px 8px !important;
    border-radius: 4px !important;
}

.ghost-btn:hover {
    color: #1a73e8 !important;
    background: #e8f0fe !important;
}

/* Sidebar Section Headers */
.sidebar-section {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary) !important;
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

/* ULTRA-HARD RESET: Prevent any overlay/blur on top of main/right areas */
[data-testid="stAppViewContainer"], 
[data-testid="stMain"], 
.main, 
.block-container, 
body, 
html {
    opacity: 1 !important;
    filter: none !important;
    backdrop-filter: none !important;
}

/* Kill all overlays, modals, or glass backgrounds */
div[data-baseweb="modal"], 
div[class*="backdrop"], 
div[role="dialog"], 
div[data-testid="stFileUploaderOverlay"], 
div[aria-modal="true"] {
    opacity: 0 !important;
    pointer-events: none !important;
    filter: none !important;
    backdrop-filter: none !important;
    background: transparent !important;
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
        
    if 'editing_message_id' not in st.session_state:
        st.session_state.editing_message_id = None
        
    if 'stop_generation' not in st.session_state:
        st.session_state.stop_generation = False
        
    if 'auto_submit_prompt' not in st.session_state:
        st.session_state.auto_submit_prompt = None

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
            return f'<span class="citation-btn" title="Click to view source">[{ref}]</span>'
        except:
            return match.group(0)
    
    return re.sub(pattern, replace_citation, text)

def render_message(message: Dict, is_user: bool):
    """Render a chat message"""
    bubble_class = "user" if is_user else "assistant"
    role_label = "You" if is_user else "LexCitator AI"
    msg_id = message.get('id')
    
    # Handle Editing State
    if is_user and st.session_state.editing_message_id == msg_id:
        with st.container():
            st.markdown(f'<div class="chat-role user">{role_label} (Editing)</div>', unsafe_allow_html=True)
            new_text = st.text_area("Edit your question", value=message.get('text', ''), key=f"edit_input_{msg_id}", label_visibility="collapsed")
            col1, col2, _ = st.columns([1, 1, 4])
            if col1.button("Save", key=f"save_{msg_id}"):
                message['text'] = new_text
                # Remove all messages AFTER this one
                current_project = get_current_project()
                if current_project:
                    try:
                        idx = current_project['messages'].index(message)
                        current_project['messages'] = current_project['messages'][:idx+1]
                    except ValueError:
                        pass
                st.session_state.editing_message_id = None
                st.session_state.auto_submit_prompt = new_text
                st.rerun()
            if col2.button("Cancel", key=f"cancel_{msg_id}"):
                st.session_state.editing_message_id = None
                st.rerun()
        return

    # Normal Rendering
    text = message.get('text', '')
    text = text.replace('**', '').replace('*', '')
    text_with_citations = parse_citations(text)
    
    # Style for edit button
    edit_btn_html = ""
    if is_user:
        # We use a column layout to place the edit button nicely
        col1, col2 = st.columns([15, 1])
        with col1:
            st.markdown(f"""
            <div class="chat-message {bubble_class}">
                <div class="chat-bubble {bubble_class}">
                    <div class="chat-role {bubble_class}">{role_label}</div>
                    <div class="chat-text">{text_with_citations}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✎", key=f"edit_btn_{msg_id}", help="Edit this message", type="secondary"):
                st.session_state.editing_message_id = msg_id
                st.rerun()
    else:
        st.markdown(f"""
        <div class="chat-message {bubble_class}">
            <div class="chat-bubble {bubble_class}">
                <div class="chat-role {bubble_class}">{role_label}</div>
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
        col_header, col_new = st.columns([2.5, 1])
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
        
        # File upload with proper state management
        if 'uploaded_file_ids' not in st.session_state:
            st.session_state.uploaded_file_ids = set()
        
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=['pdf', 'txt', 'md', 'csv'],
            accept_multiple_files=True,
            key="file_uploader",
            help="Maximum 200MB per file. Supports: PDF, TXT, MD, CSV"
        )
        
        if uploaded_files:
            current_project = get_current_project()
            if current_project:
                files_added = False
                for file in uploaded_files:
                    # Create a unique identifier for this file
                    file_id = f"{file.name}_{file.size}"
                    
                    # Check if file already processed in this session
                    if file_id not in st.session_state.uploaded_file_ids:
                        # Check if file already exists in project by name
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
                            st.session_state.uploaded_file_ids.add(file_id)
                            files_added = True
                
                if files_added:
                    st.rerun()
        
        # Knowledge Base Status
        if st.session_state.knowledge_base_loaded:
            st.markdown("""
            <div class="kb-status">
                <span class="kb-status-dot"></span>
                <span style="color: #86efac; font-weight: 500; margin-left: 0.5rem;">📚 Knowledge Base Active</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⏳ Loading Knowledge Base...")
        
        # Document list removal (requested to avoid duplication with uploader)
        pass
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0;">
            <div class="ai-badge">AI</div>
            <div>
                <div style="font-size: 0.875rem; font-weight: 500; color: #202124;">Gemini</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN AREA =====
    current_project = get_current_project()
    
    if current_project:
        # 1. Header & Quick Actions
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown("### 📖 Legal Research Workspace")
        with col2:
            if st.button("Clear", type="secondary"):
                current_project['messages'] = []
                reset_session(current_project['id'])
                st.rerun()
        st.markdown("---")

        # 2. Capture Input (Early in script to update state)
        if st.session_state.auto_submit_prompt:
            prompt = st.session_state.auto_submit_prompt
            st.session_state.auto_submit_prompt = None
        else:
            prompt = st.chat_input("Ask for an Essay, Case Analysis, or Client Advice...")

        # 3. Workspace Rendering Mode Selection
        messages = current_project.get('messages', [])
        
        # Determine if we show the Welcome screen
        # MUST only show if no history AND no active prompt is being sent
        show_welcome = not messages and not prompt
        
        if show_welcome:
            # === WELCOME SCREEN (The 3 elements) ===
            st.markdown("""
            <div style="text-align: center; max-width: 50rem; margin: 0 auto; padding: 2rem 2rem 0 2rem;">
                <div style="font-size: 3.5rem; color: #dadce0; margin-bottom: 0.5rem;">📚</div>
                <h2 style="font-family: 'Product Sans', sans-serif; font-size: 2.25rem; color: #202124; margin-bottom: 0.5rem; font-weight: 700;">Legal AI</h2>
                <p style="color: #5f6368; font-size: 1.125rem; margin-bottom: 1rem;">Your distinguished legal scholar & academic writing expert</p>
                <div style="color: #202124; font-size: 1.25rem; font-weight: 500; margin: 1rem 0 1.5rem 0;">Just ask your question</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.knowledge_base_loaded:
                # Use a specific container for the boxes to ensure spacing from chat input
                with st.container():
                    col_cap, col_try = st.columns(2)
                    with col_cap:
                        st.markdown("""
                        <div style="background: white; border: 1px solid #dadce0; border-radius: 1rem; padding: 1.5rem; box-shadow: var(--card-shadow); height: 280px; display: flex; flex-direction: column;">
                            <h4 style="font-size: 0.8rem; font-weight: 700; color: #5f6368; text-transform: uppercase; margin-bottom: 1.25rem; letter-spacing: 0.5px;">Capabilities</h4>
                            <div class="custom-list-item"><div class="blue-dot"></div>Comprehensive Essay Writing</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>Legal Problem Question Analysis</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>Client Advice & Litigation Strategy</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>General Legal Queries</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_try:
                        st.markdown("""
                        <div style="background: white; border: 1px solid #dadce0; border-radius: 1rem; padding: 1.5rem; box-shadow: var(--card-shadow); height: 280px; display: flex; flex-direction: column;">
                            <h4 style="font-size: 0.8rem; font-weight: 700; color: #5f6368; margin-bottom: 1.25rem; letter-spacing: 0.5px; text-transform: uppercase;">✨ Try Asking</h4>
                            <div class="suggestion-chip" style="margin-bottom: 8px;">"What are the key elements of a valid contract under English law?"</div>
                            <div class="suggestion-chip">"Critically analyze the duty of care in UK tort law"</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Large spacer at the bottom to prevent overlap with the chat input bar
                st.markdown('<div style="margin-bottom: 150px;"></div>', unsafe_allow_html=True)
        else:
            # === CHAT WORKSPACE ===
            # If a new prompt was just sent, add it to the message list before rendering
            if prompt and (not messages or messages[-1]['text'] != prompt):
                new_user_msg = {
                    'id': str(uuid.uuid4()),
                    'role': 'user',
                    'text': prompt,
                    'timestamp': datetime.now().isoformat()
                }
                current_project['messages'].append(new_user_msg)
                messages = current_project['messages'] # Update local ref

            # 1. Render History Loop
            chat_placeholder = st.container()
            with chat_placeholder:
                for msg in messages:
                    render_message(msg, msg['role'] == 'user')
            
            # 2. Handle AI Generation (if this run was triggered by a prompt)
            if prompt:
                # Thinking State (Pic 2 style)
                thinking_placeholder = st.empty()
                with thinking_placeholder.container():
                    st.markdown("""
                    <div class="chat-message assistant">
                        <div class="chat-bubble assistant">
                            <div class="chat-role assistant">LexCitator AI</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div class="thinking-spinner"></div>
                                <div class="chat-text" style="color: #5f6368; font-style: italic;">Thinking...</div>
                            </div>
                        </div>
                    </div>
                    <style>
                    .thinking-spinner {
                        width: 14px;
                        height: 14px;
                        border: 2px solid #dadce0;
                        border-top: 2px solid #1a73e8;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    }
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Stop Button UI
                    _, stop_col = st.columns([5, 1])
                    if stop_col.button("Stop ⏹", key="stop_btn_active"):
                        st.info("Generation halted.")
                        st.rerun()

                # Call Service
                api_key = st.session_state.api_key or os.environ.get('GEMINI_API_KEY', '')
                if not api_key:
                    st.error("Please enter a Gemini API key in the sidebar.")
                else:
                    try:
                        history = messages[:-1] # All messages except the newest prompt
                        stream = send_message_with_docs(
                            api_key, prompt, current_project.get('documents', []),
                            current_project['id'], history=history, stream=True
                        )
                        
                        # Once stream starts, remove thinking indicator
                        thinking_placeholder.empty()
                        
                        # Streaming response
                        response_text = ""
                        message_placeholder = st.empty()
                        for chunk in stream:
                            if chunk.text:
                                response_text += chunk.text
                                display_text = response_text.replace('**', '').replace('*', '')
                                message_placeholder.markdown(f"""
                                <div class="chat-message assistant">
                                    <div class="chat-bubble assistant">
                                        <div class="chat-role assistant">LexCitator AI</div>
                                        <div class="chat-text">{display_text}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Save Final
                        current_project['messages'].append({
                            'id': str(uuid.uuid4()), 'role': 'assistant',
                            'text': response_text, 'timestamp': datetime.now().isoformat(),
                            'grounding_urls': []
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error communicating with AI: {str(e)}")

if __name__ == "__main__":
    main()
