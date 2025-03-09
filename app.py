import streamlit as st
from src.models.rag_model import RAGModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="UDST Policy Assistant",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Theme Variables */
    :root {
        --primary: #2970ff;
        --primary-light: #5b92ff;
        --secondary: #10a37f;
        --background: #f9fafb;
        --chat-background: #ffffff;
        --user-message-background: #f3f4f6;
        --assistant-message-background: #ffffff;
        --text: #111827;
        --text-light: #6b7280;
        --border: #e5e7eb;
        --shadow: rgba(0, 0, 0, 0.08);
        --radius: 12px;
        --radius-sm: 8px;
        --max-width: 900px;
    }

    /* Global Styles */
    .stApp {
        background: var(--background) !important;
    }

    .main {
        max-width: var(--max-width);
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    /* Chat Container */
    .chat-container {
        max-width: var(--max-width);
        margin: 0 auto;
        padding-bottom: 100px;
        background: transparent;
        border: none;
        box-shadow: none;
        min-height: auto;
    }

    /* Empty State */
    .empty-state {
        padding: 3rem 1rem;
        color: var(--text-light);
        font-size: 1.2rem;
        text-align: center;
        background: white;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 2px 4px var(--shadow);
    }
    
    /* Message Groups */
    .message-group {
        background: white;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        border-radius: var(--radius);
        box-shadow: 0 2px 4px var(--shadow);
    }

    .user-group {
        background: var(--user-message-background);
    }

    .assistant-group {
        background: var(--assistant-message-background);
    }
    
    /* Messages */
    .message {
        max-width: var(--max-width);
        margin: 0 auto;
        padding: 1.5rem;
        display: flex;
        gap: 1.5rem;
        line-height: 1.6;
        font-size: 1rem;
        color: var(--text);
        animation: messageSlide 0.3s ease-out;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        box-shadow: 0 2px 4px var(--shadow);
    }
    
    .user-avatar {
        background: var(--primary);
        color: white;
    }
    
    .assistant-avatar {
        background: var(--secondary);
        color: white;
    }
    
    .content {
        flex: 1;
        padding: 0.25rem 0;
        overflow-wrap: break-word;
    }
    
    /* Source Documents */
    .source-document {
        margin-top: 1.25rem;
        padding: 1rem 1.25rem;
        background: var(--user-message-background);
        border-radius: var(--radius-sm);
        font-size: 0.925rem;
        border-left: 3px solid var(--primary);
        animation: sourceSlide 0.5s ease-out;
    }
    
    .source-document p {
        margin: 0 0 0.75rem 0;
        color: var(--text);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Input Area */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: linear-gradient(180deg, transparent, var(--background) 20%);
        z-index: 1000;
    }
    
    .input-container {
        max-width: var(--max-width);
        margin: 0 auto;
        background: white;
        border-radius: var(--radius);
        box-shadow: 0 0 20px var(--shadow);
        padding: 0.75rem;
        border: 1px solid var(--border);
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 0.75rem;
        align-items: center;
    }

    /* Clear Button */
    .clear-button {
        min-width: 100px;
        height: 100%;
    }
    
    .clear-button button {
        height: 42px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 1.5rem !important;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1rem !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        color: var(--text) !important;
        background: var(--chat-background) !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: var(--primary-light) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(41, 112, 255, 0.1) !important;
    }

    /* Button Styling */
    .stButton > button {
        background: var(--user-message-background) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1.5rem !important;
        font-size: 0.925rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: var(--border) !important;
        border-color: var(--text-light) !important;
    }
    
    /* Alerts */
    .stAlert {
        background: var(--user-message-background) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.25rem !important;
        color: var(--text) !important;
        font-size: 0.925rem !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }

    /* Animations */
    @keyframes messageSlide {
        from { 
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes sourceSlide {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Welcome Screen */
    .welcome-container {
        max-width: var(--max-width);
        margin: 4rem auto;
        text-align: center;
        animation: fadeIn 0.5s ease-out;
    }

    .welcome-title {
        font-size: 2.5rem;
        color: var(--text);
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .welcome-subtitle {
        font-size: 1.1rem;
        color: var(--text-light);
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question' not in st.session_state:
    st.session_state.question = ""


def initialize_model():
    """Initialize the RAG model."""
    try:
        rag_model = RAGModel()
        if rag_model.initialize():
            st.session_state.rag_model = rag_model
            return True
        return False
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False


def display_chat_history():
    """Display the chat history with source documents."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
                <div class="message-group user-group">
                    <div class="message">
                        <div class="avatar user-avatar">👤</div>
                        <div class="content">{content}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-group assistant-group">
                    <div class="message">
                        <div class="avatar assistant-avatar">🤖</div>
                        <div class="content">{content}</div>
                    </div>
            """, unsafe_allow_html=True)
            
            if "sources" in message and message["sources"]:
                st.markdown("""
                    <div class="source-document">
                        <p>📚 Sources Referenced</p>
                """, unsafe_allow_html=True)
                for source in message["sources"]:
                    st.markdown(f"• {source}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)


def handle_question():
    """Handle the question submission."""
    if st.session_state.question:
        question = st.session_state.question
        st.session_state.question = ""
        
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        try:
            # Get the answer
            answer, sources = st.session_state.rag_model.get_answer_with_sources(question)
            
            if answer:
                # Add assistant's response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error("I couldn't generate an answer. Please try rephrasing your question.")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            st.error("An error occurred while processing your question. Please try again.")


def main():
    # Main content area
    if st.session_state.rag_model is None:
        st.markdown("""
            <div class="welcome-container">
                <div class="welcome-title">🎓 UDST Policy Assistant</div>
                <div class="welcome-subtitle">
                    Your AI-powered guide to understanding UDST policies.<br>
                    Get instant, accurate answers from official documentation.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize Assistant", use_container_width=True):
                with st.spinner("Initializing AI assistant..."):
                    if initialize_model():
                        st.success("✨ Ready to help you with UDST policies!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize the assistant. Please try again.")
    else:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown("""
                <div class="empty-state">
                    👋 Hi! I'm ready to help you with UDST policies.<br>
                    Ask me anything!
                </div>
            """, unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)
    
        # Input area
        st.markdown('<div class="input-area"><div class="input-container">', unsafe_allow_html=True)
        st.text_input(
            "",
            key="question",
            on_change=handle_question,
            placeholder="Ask me anything about UDST policies..."
        )
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div></div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main() 