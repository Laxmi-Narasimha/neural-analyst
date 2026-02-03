"""Streamlit frontend for AI Data Adequacy Agent."""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="AI Data Adequacy Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'validation_step' not in st.session_state:
    st.session_state.validation_step = 'initial'
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'results' not in st.session_state:
    st.session_state.results = None

def check_api_health():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_files_and_start_validation(goal: str, domain: str, uploaded_files):
    """Upload files and start validation process."""
    try:
        files = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                files.append(("files", (uploaded_file.name, uploaded_file.read(), uploaded_file.type)))
        
        data = {
            "goal": goal,
            "domain": domain
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/validate",
            data=data,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def continue_validation(session_id: str, answers: Dict[str, str]):
    """Continue validation with user answers."""
    try:
        payload = {
            "session_id": session_id,
            "answers": answers
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/validate/continue",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def render_readiness_badge(readiness_level: str, score: float):
    """Render readiness level badge."""
    colors = {
        "READY": "#28a745",
        "PARTIALLY_READY": "#ffc107", 
        "UNSAFE": "#fd7e14",
        "BLOCKED": "#dc3545"
    }
    
    color = colors.get(readiness_level, "#6c757d")
    
    st.markdown(f"""
    <div style="display: inline-block; background-color: {color}; color: white; 
                padding: 0.5rem 1rem; border-radius: 1rem; font-weight: bold; margin: 1rem 0;">
        {readiness_level.replace('_', ' ')} ({score:.1%})
    </div>
    """, unsafe_allow_html=True)

def render_questions_form(questions: List[Dict]):
    """Render the clarifying questions form."""
    st.header("📋 Clarifying Questions")
    st.write("Please answer these questions to help us better understand your requirements:")
    
    answers = {}
    
    with st.form("questions_form"):
        for i, question in enumerate(questions):
            st.subheader(f"Question {i+1}")
            st.write(f"**Priority:** {question.get('priority', 'medium').title()}")
            st.write(f"**Category:** {question.get('failure_mode', 'GENERAL')}")
            st.write(question['text'])
            
            if question.get('expected_evidence'):
                with st.expander("What we're looking for"):
                    st.write(question['expected_evidence'])
            
            answer = st.text_area(
                f"Your answer:",
                key=f"answer_{question['id']}",
                height=100,
                placeholder="Please provide as much detail as possible..."
            )
            
            answers[question['id']] = answer
        
        submit_button = st.form_submit_button("Submit Answers", type="primary")
        
        if submit_button:
            # Validate that at least critical questions are answered
            critical_questions = [q for q in questions if q.get('priority') == 'critical']
            unanswered_critical = []
            
            for q in critical_questions:
                if not answers.get(q['id'], '').strip():
                    unanswered_critical.append(q['text'][:50] + "...")
            
            if unanswered_critical:
                st.error("Please answer all critical questions before proceeding:")
                for q in unanswered_critical:
                    st.write(f"- {q}")
            else:
                return answers
    
    return None

def render_results_dashboard(results: Dict[str, Any]):
    """Render the validation results dashboard."""
    st.header("📊 Validation Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_readiness_badge(results['readiness_level'], results['composite_score'])
    
    with col2:
        st.metric("Composite Score", f"{results['composite_score']:.1%}")
    
    with col3:
        st.metric("Files Processed", results['technical_details']['files_processed'])
    
    with col4:
        st.metric("Processing Time", results['technical_details']['processing_time'])
    
    # Executive Summary
    st.subheader("📝 Executive Summary")
    st.markdown(results['executive_summary'])
    
    # Top Recommendations
    if results.get('top_recommendations'):
        st.subheader("⚠️ Top Recommendations")
        
        for i, rec in enumerate(results['top_recommendations'][:5], 1):
            priority_colors = {
                "critical": "🚨",
                "high": "⚠️", 
                "medium": "📋",
                "low": "💡"
            }
            
            icon = priority_colors.get(rec.get('priority', 'medium'), "📋")
            
            with st.expander(f"{icon} {rec.get('issue', 'Issue')} ({rec.get('priority', 'medium').title()})"):
                st.write(f"**Category:** {rec.get('category', 'general')}")
                st.write(f"**Details:** {rec.get('details', 'No details available')}")
                st.write(f"**Impact:** {rec.get('impact', 'Unknown impact')}")
                
                if rec.get('suggested_actions'):
                    st.write("**Suggested Actions:**")
                    for action in rec['suggested_actions']:
                        st.write(f"- {action}")
    
    # Next Steps
    if results.get('next_steps'):
        st.subheader("🎯 Next Steps")
        for step in results['next_steps']:
            st.write(f"- {step}")
    
    # Technical Details
    with st.expander("🔧 Technical Details"):
        tech_details = results['technical_details']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Processing Statistics:**")
            st.write(f"- Namespace: `{tech_details['namespace']}`")
            st.write(f"- Chunks Created: {tech_details['chunks_created']}")
            st.write(f"- LLM Calls Used: {tech_details['llm_calls_used']}")
        
        with col2:
            st.write("**Session Information:**")
            st.write(f"- Session ID: `{results['session_id']}`")
            st.write(f"- Processing Time: {tech_details['processing_time']}")
    
    # Download Reports
    st.subheader("📥 Download Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Markdown Report", type="secondary"):
            st.info("Report download functionality would be implemented here")
    
    with col2:
        if st.button("Download JSON Report", type="secondary"):
            st.info("JSON report download functionality would be implemented here")
    
    # Raw Results (for debugging)
    with st.expander("🔍 Raw Results (Debug)"):
        st.json(results)

def main():
    """Main Streamlit application."""
    st.title("🤖 AI Data Adequacy Agent")
    st.markdown("**Comprehensive data validation system for AI assistants**")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to the API server. Please ensure the backend is running on http://localhost:8000")
        st.info("Run: `python -m uvicorn app.main:app --reload` in the backend directory")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Domain selection
        domains = {
            "general": "General Purpose",
            "automotive": "Automotive Industry",
            "manufacturing": "Manufacturing",
            "real_estate": "Real Estate"
        }
        
        selected_domain = st.selectbox(
            "Select Domain:",
            options=list(domains.keys()),
            format_func=lambda x: domains[x],
            index=0
        )
        
        # Session management
        st.header("📊 Session Info")
        if st.session_state.session_id:
            st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
            st.write(f"**Step:** {st.session_state.validation_step}")
            
            if st.button("Reset Session", type="secondary"):
                st.session_state.session_id = None
                st.session_state.validation_step = 'initial'
                st.session_state.questions = []
                st.session_state.results = None
                st.experimental_rerun()
        
        # Help section
        with st.expander("ℹ️ Help"):
            st.write("""
            **How to use this tool:**
            
            1. **Describe your goal** - What should your AI assistant do?
            2. **Upload files** - Provide your knowledge base documents
            3. **Answer questions** - Help us understand your requirements
            4. **Review results** - Get detailed analysis and recommendations
            
            **Supported file types:**
            - PDF documents
            - Word documents (.docx)
            - Text files (.txt)
            - CSV files
            - Excel files (.xlsx)
            """)
    
    # Main content based on validation step
    if st.session_state.validation_step == 'initial':
        # Initial setup
        st.header("🚀 Start Validation")
        
        with st.form("initial_setup"):
            goal = st.text_area(
                "Describe your AI assistant goal:",
                height=150,
                placeholder="Example: I want to create an AI assistant that helps customers find the right car based on their needs, budget, and preferences. It should provide accurate pricing, availability, and detailed specifications.",
                help="Be as specific as possible about what your AI assistant should do"
            )
            
            uploaded_files = st.file_uploader(
                "Upload your knowledge base files:",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
                help="Upload documents that contain the information your AI assistant will use"
            )
            
            submit_button = st.form_submit_button("Start Validation", type="primary")
            
            if submit_button:
                if not goal.strip():
                    st.error("Please provide a description of your AI assistant goal.")
                elif not uploaded_files:
                    st.error("Please upload at least one file.")
                else:
                    with st.spinner("Starting validation process..."):
                        result = upload_files_and_start_validation(goal, selected_domain, uploaded_files)
                        
                        if result and result.get('success'):
                            st.session_state.session_id = result['session_id']
                            st.session_state.questions = result['questions']
                            st.session_state.validation_step = 'questions'
                            st.experimental_rerun()
    
    elif st.session_state.validation_step == 'questions':
        # Questions phase
        answers = render_questions_form(st.session_state.questions)
        
        if answers:
            with st.spinner("Processing your answers and running validation..."):
                result = continue_validation(st.session_state.session_id, answers)
                
                if result and result.get('success'):
                    st.session_state.results = result
                    st.session_state.validation_step = 'results'
                    st.experimental_rerun()
                else:
                    st.error("Validation failed. Please try again.")
    
    elif st.session_state.validation_step == 'results':
        # Results phase
        if st.session_state.results:
            render_results_dashboard(st.session_state.results)
            
            # Option to start new validation
            st.markdown("---")
            if st.button("Start New Validation", type="primary"):
                st.session_state.session_id = None
                st.session_state.validation_step = 'initial'
                st.session_state.questions = []
                st.session_state.results = None
                st.experimental_rerun()

if __name__ == "__main__":
    main()
