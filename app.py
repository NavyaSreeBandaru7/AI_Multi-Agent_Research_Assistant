import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import uuid
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Import our main modules
from main import MultiAgentOrchestrator, AIConfig
from research_agent import *

# Configure Streamlit page
st.set_page_config(
    page_title="AI Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-good { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .code-block {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
        st.session_state.orchestrator_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_response_time': 0.0,
            'last_update': datetime.now()
        }
    
    if 'config' not in st.session_state:
        st.session_state.config = AIConfig()

@st.cache_resource
def get_orchestrator():
    """Get or create the orchestrator instance"""
    config = AIConfig()
    orchestrator = MultiAgentOrchestrator(config)
    return orchestrator

# Async wrapper for Streamlit
def run_async(coro):
    """Run async function in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        initialize_session_state()
        self.orchestrator = get_orchestrator()
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AI Multi-Agent Research Assistant</h1>
            <p>Advanced AI system with NLP, Generative AI, and Multi-Agent Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_main_content()
        
        # Initialize orchestrator if not done
        if not st.session_state.orchestrator_initialized:
            self.initialize_orchestrator()
    
    def render_sidebar(self):
        """Render the sidebar with controls and system status"""
        with st.sidebar:
            st.header("🛠️ Control Panel")
            
            # System Status
            st.subheader("System Status")
            self.render_system_status()
            
            # Configuration
            st.subheader("⚙️ Configuration")
            self.render_configuration()
            
            # Analytics
            st.subheader("📊 Analytics")
            self.render_analytics()
            
            # Agent Management
            st.subheader("🤖 Agent Management")
            self.render_agent_management()
    
    def render_system_status(self):
        """Render system status indicators"""
        if st.session_state.orchestrator_initialized:
            try:
                status = run_async(self.orchestrator.get_system_status())
                
                # Overall status
                overall_health = status.get('performance_summary', {}).get('overall_system_health', 'unknown')
                status_color = {
                    'excellent': 'status-good',
                    'good': 'status-good',
                    'fair': 'status-warning',
                    'poor': 'status-error'
                }.get(overall_health, 'status-warning')
                
                st.markdown(f"""
                <div class="metric-card">
                    <div><span class="status-indicator {status_color}"></span>System Health: {overall_health.title()}</div>
                    <div>Session ID: {st.session_state.session_id[:8]}...</div>
                    <div>Uptime: {self.format_uptime(status.get('timestamp'))}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Agent status
                agents = status.get('agents', {})
                for agent_name, agent_info in agents.items():
                    agent_status = agent_info.get('status', 'unknown')
                    success_rate = agent_info.get('performance', {}).get('success_rate', 0)
                    
                    color = 'status-good' if agent_status == 'operational' else 'status-error'
                    
                    st.markdown(f"""
                    <div class="agent-card">
                        <span class="status-indicator {color}"></span>
                        {agent_name.upper()}: {success_rate:.1f}% success
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to get system status: {e}")
        else:
            st.warning("System initializing...")
    
    def render_configuration(self):
        """Render configuration controls"""
        with st.expander("Model Settings"):
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=st.session_state.config.temperature,
                step=0.1,
                help="Controls randomness in generation"
            )
            
            max_tokens = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=1024,
                value=st.session_state.config.max_new_tokens,
                step=50,
                help="Maximum tokens to generate"
            )
            
            top_p = st.slider(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config.top_p,
                step=0.05,
                help="Nucleus sampling parameter"
            )
            
            # Update config
            st.session_state.config.temperature = temperature
            st.session_state.config.max_new_tokens = max_tokens
            st.session_state.config.top_p = top_p
        
        with st.expander("System Settings"):
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=st.session_state.config.batch_size,
                help="Processing batch size"
            )
            
            max_concurrent = st.slider(
                "Max Concurrent Tasks",
                min_value=1,
                max_value=10,
                value=st.session_state.config.max_concurrent_tasks,
                help="Maximum concurrent agent tasks"
            )
            
            st.session_state.config.batch_size = batch_size
            st.session_state.config.max_concurrent_tasks = max_concurrent
        
        # Reset button
        if st.button("🔄 Reset to Defaults"):
            st.session_state.config = AIConfig()
            st.experimental_rerun()
    
    def render_analytics(self):
        """Render analytics and metrics"""
        metrics = st.session_state.system_metrics
        
        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Requests", metrics['total_requests'])
            st.metric("Success Rate", f"{self.calculate_success_rate():.1f}%")
        
        with col2:
            st.metric("Avg Response", f"{metrics['avg_response_time']:.2f}s")
            st.metric("Active Session", "🟢" if st.session_state.orchestrator_initialized else "🔴")
        
        # Performance chart
        if st.checkbox("Show Performance Chart"):
            self.render_performance_chart()
    
    def render_agent_management(self):
        """Render agent management controls"""
        if st.session_state.orchestrator_initialized:
            st.success("✅ All agents initialized")
            
            # Agent capabilities
            with st.expander("Agent Capabilities"):
                agent_info = {
                    "NLU Agent": ["intent_classification", "entity_extraction", "sentiment_analysis", "emotion_detection"],
                    "Generative Agent": ["text_generation", "summarization", "question_answering", "creative_writing"],
                    "Research Agent": ["web_scraping", "knowledge_synthesis", "fact_checking", "trend_analysis"]
                }
                
                for agent, capabilities in agent_info.items():
                    st.write(f"**{agent}:**")
                    for cap in capabilities:
                        st.write(f"  • {cap.replace('_', ' ').title()}")
        else:
            if st.button("🚀 Initialize Agents"):
                self.initialize_orchestrator()
    
    def render_main_content(self):
        """Render main content area"""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Analytics Dashboard", "🔍 Research Tools", "⚙️ System Monitor"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_research_tools()
        
        with tab4:
            self.render_system_monitor()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.header("💬 AI Assistant Chat")
        
        # Chat history
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                self.render_message(message)
        
        # Input form
        with st.form("chat_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask me anything...",
                    placeholder="Enter your question or request here",
                    height=100,
                    key="user_input"
                )
            
            with col2:
                task_type = st.selectbox(
                    "Task Type",
                    ["auto_detect", "question_answering", "research_tasks", "content_generation", 
                     "text_analysis", "fact_checking", "summarization", "creative_writing"],
                    help="Select specific task type or use auto-detect"
                )
            
            with col3:
                complexity = st.selectbox(
                    "Complexity",
                    ["low", "medium", "high"],
                    index=1,
                    help="Expected complexity level"
                )
            
            submitted = st.form_submit_button("🚀 Send", use_container_width=True)
        
        # Process input
        if submitted and user_input.strip():
            if not st.session_state.orchestrator_initialized:
                st.error("Please initialize the system first!")
                return
            
            # Add user message to history
            user_message = {
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now(),
                'task_type': task_type,
                'complexity': complexity
            }
            st.session_state.chat_history.append(user_message)
            
            # Process with orchestrator
            with st.spinner("🤖 AI agents are processing your request..."):
                start_time = time.time()
                
                try:
                    request = {
                        'text': user_input,
                        'task_type': task_type,
                        'complexity': complexity
                    }
                    
                    result = run_async(
                        self.orchestrator.process_request(request, st.session_state.session_id)
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Add assistant response to history
                    assistant_message = {
                        'role': 'assistant',
                        'content': result,
                        'timestamp': datetime.now(),
                        'response_time': response_time,
                        'agents_used': result.get('agents_used', [])
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Update metrics
                    self.update_metrics(response_time, 'error' not in result)
                    
                    # Force refresh
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error processing request: {e}")
                    error_message = {
                        'role': 'assistant',
                        'content': {'error': str(e)},
                        'timestamp': datetime.now(),
                        'response_time': time.time() - start_time
                    }
                    st.session_state.chat_history.append(error_message)
    
    def render_message(self, message: Dict[str, Any]):
        """Render a chat message"""
        role = message['role']
        content = message['content']
        timestamp = message['timestamp']
        
        if role == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong> <small>({timestamp.strftime('%H:%M:%S')})</small><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        
        else:  # assistant
            agents_used = message.get('agents_used', [])
            response_time = message.get('response_time', 0)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant</strong> 
                <small>({timestamp.strftime('%H:%M:%S')}) - {response_time:.2f}s</small>
                {f"<br><small>Agents: {', '.join(agents_used)}</small>" if agents_used else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Display content based on type
            if isinstance(content, dict):
                if 'error' in content:
                    st.error(f"Error: {content['error']}")
                else:
                    self.render_structured_response(content)
            else:
                st.write(content)
    
    def render_structured_response(self, response: Dict[str, Any]):
        """Render structured AI response"""
        strategy = response.get('strategy', 'unknown')
        final_output = response.get('final_output', {})
        
        # Show strategy used
        st.info(f"🔄 Processing Strategy: {strategy.title()}")
        
        # Show final output
        if isinstance(final_output, dict):
            # Handle different output types
            if 'comprehensive_answer' in final_output:
                self.render_comprehensive_answer(final_output['comprehensive_answer'])
            elif 'generated_text' in final_output:
                st.write(final_output['generated_text'])
            elif 'summary' in final_output:
                st.write(final_output['summary'])
            else:
                # Generic structured display
                with st.expander("📋 Detailed Results", expanded=True):
                    for key, value in final_output.items():
                        if key not in ['timestamp', 'metadata']:
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
        else:
            st.write(str(final_output))
        
        # Show agent results
        if 'results' in response:
            with st.expander("🔍 Agent Details"):
                for agent_name, agent_result in response['results'].items():
                    st.subheader(f"{agent_name.upper()} Agent")
                    if isinstance(agent_result, dict) and 'error' not in agent_result:
                        st.success("✅ Completed successfully")
                        st.json(agent_result)
                    else:
                        st.error(f"❌ Error: {agent_result.get('error', 'Unknown error')}")
    
    def render_comprehensive_answer(self, answer: Dict[str, Any]):
        """Render comprehensive answer format"""
        st.markdown("### 📝 Comprehensive Response")
        
        # Main response
        response = answer.get('comprehensive_response', '')
        if response:
            st.write(response)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{answer.get('confidence_score', 0):.1%}")
        with col2:
            st.metric("Sources", answer.get('sources_consulted', 0))
        with col3:
            st.metric("Depth", answer.get('analysis_depth', 'N/A'))
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.chat_history:
            st.info("No conversation data available yet. Start chatting to see analytics!")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
            st.metric("Total Queries", total_messages)
        
        with col2:
            avg_response_time = np.mean([
                m.get('response_time', 0) for m in st.session_state.chat_history 
                if m['role'] == 'assistant' and 'response_time' in m
            ])
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col3:
            success_rate = self.calculate_success_rate()
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            active_session_time = datetime.now() - st.session_state.chat_history[0]['timestamp'] if st.session_state.chat_history else timedelta(0)
            st.metric("Session Duration", str(active_session_time).split('.')[0])
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_response_time_chart()
        
        with col2:
            self.render_agent_usage_chart()
        
        # Task type distribution
        self.render_task_distribution_chart()
        
        # Conversation timeline
        self.render_conversation_timeline()
    
    def render_response_time_chart(self):
        """Render response time chart"""
        st.subheader("⏱️ Response Times")
        
        response_times = []
        timestamps = []
        
        for message in st.session_state.chat_history:
            if message['role'] == 'assistant' and 'response_time' in message:
                response_times.append(message['response_time'])
                timestamps.append(message['timestamp'])
        
        if response_times:
            df = pd.DataFrame({
                'Timestamp': timestamps,
                'Response Time (s)': response_times
            })
            
            fig = px.line(df, x='Timestamp', y='Response Time (s)', 
                         title='Response Time Over Time')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No response time data available")
    
    def render_agent_usage_chart(self):
        """Render agent usage chart"""
        st.subheader("🤖 Agent Usage")
        
        agent_usage = {}
        for message in st.session_state.chat_history:
            if message['role'] == 'assistant' and 'agents_used' in message:
                for agent in message['agents_used']:
                    agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        if agent_usage:
            df = pd.DataFrame(list(agent_usage.items()), columns=['Agent', 'Usage Count'])
            fig = px.pie(df, values='Usage Count', names='Agent', title='Agent Usage Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent usage data available")
    
    def render_task_distribution_chart(self):
        """Render task type distribution"""
        st.subheader("📋 Task Type Distribution")
        
        task_counts = {}
        for message in st.session_state.chat_history:
            if message['role'] == 'user' and 'task_type' in message:
                task_type = message['task_type']
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        if task_counts:
            df = pd.DataFrame(list(task_counts.items()), columns=['Task Type', 'Count'])
            fig = px.bar(df, x='Task Type', y='Count', title='Task Type Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No task distribution data available")
    
    def render_conversation_timeline(self):
        """Render conversation timeline"""
        st.subheader("📅 Conversation Timeline")
        
        if st.session_state.chat_history:
            timeline_data = []
            for i, message in enumerate(st.session_state.chat_history):
                timeline_data.append({
                    'Message': i + 1,
                    'Role': message['role'],
                    'Timestamp': message['timestamp'],
                    'Content Length': len(str(message['content']))
                })
            
            df = pd.DataFrame(timeline_data)
            fig = px.scatter(df, x='Timestamp', y='Message', color='Role', 
                           size='Content Length', title='Conversation Timeline')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_research_tools(self):
        """Render research tools interface"""
        st.header("🔍 Advanced Research Tools")
        
        if not st.session_state.orchestrator_initialized:
            st.warning("Please initialize the system first!")
            return
        
        # Research task selector
        research_type = st.selectbox(
            "Research Type",
            ["web_research", "knowledge_synthesis", "fact_checking", "trend_analysis"],
            help="Select the type of research to perform"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_area(
                "Research Query",
                placeholder="Enter your research question or topic...",
                height=100
            )
        
        with col2:
            depth = st.selectbox("Research Depth", ["shallow", "medium", "deep"])
            sources = st.multiselect(
                "Preferred Sources",
                ["academic", "news", "government", "social", "specialized"],
                default=["academic", "news"]
            )
        
        if st.button("🔍 Start Research", type="primary"):
            if query.strip():
                with st.spinner("🔬 Conducting research..."):
                    try:
                        research_request = {
                            'type': research_type,
                            'query': query,
                            'depth': depth,
                            'sources': sources
                        }
                        
                        # Get research agent
                        research_agent = self.orchestrator.agents.get('research')
                        if research_agent:
                            result = run_async(research_agent.process(research_request))
                            self.render_research_results(result)
                        else:
                            st.error("Research agent not available")
                    
                    except Exception as e:
                        st.error(f"Research failed: {e}")
            else:
                st.warning("Please enter a research query")
        
        # Fact-checking tool
        st.subheader("✅ Fact Checker")
        with st.expander("Fact Check Claims"):
            claim = st.text_input("Enter claim to fact-check:")
            if st.button("Check Fact") and claim:
                with st.spinner("Fact-checking..."):
                    try:
                        fact_check_request = {
                            'type': 'fact_checking',
                            'claim': claim,
                            'preferred_sources': ['academic', 'government']
                        }
                        
                        research_agent = self.orchestrator.agents.get('research')
                        if research_agent:
                            result = run_async(research_agent.process(fact_check_request))
                            self.render_fact_check_results(result)
                    except Exception as e:
                        st.error(f"Fact-check failed: {e}")
    
    def render_research_results(self, result: Dict[str, Any]):
        """Render research results"""
        st.success("✅ Research completed!")
        
        # Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Findings", len(result.get('findings', [])))
        with col2:
            st.metric("Quality Score", f"{result.get('quality_score', 0):.2f}")
        with col3:
            st.metric("Sources", len(set([f.get('source', '') for f in result.get('findings', [])])))
        
        # Findings
        findings = result.get('findings', [])
        for i, finding in enumerate(findings):
            with st.expander(f"Finding {i+1}: {finding.get('strategy', 'N/A').title()}", expanded=i==0):
                if 'content_summary' in finding:
                    st.write("**Summary:**", finding['content_summary'])
                
                if 'key_points' in finding:
                    st.write("**Key Points:**")
                    for point in finding['key_points']:
                        st.write(f"• {point}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'relevance_score' in finding:
                        st.metric("Relevance", f"{finding['relevance_score']:.2%}")
                with col2:
                    if 'credibility_score' in finding:
                        st.metric("Credibility", f"{finding['credibility_score']:.2%}")
                
                if 'source' in finding:
                    st.write(f"**Source:** {finding['source']}")
    
    def render_fact_check_results(self, result: Dict[str, Any]):
        """Render fact-check results"""
        findings = result.get('findings', [])
        if not findings:
            st.warning("No fact-check results available")
            return
        
        fact_check = findings[0]  # First finding should be fact-check result
        
        verdict = fact_check.get('verdict', 'unknown')
        confidence = fact_check.get('confidence', 0)
        
        # Color code verdict
        verdict_colors = {
            'likely_true': '🟢',
            'likely_false': '🔴',
            'disputed': '🟡',
            'insufficient_evidence': '⚫'
        }
        
        st.markdown(f"""
        ### {verdict_colors.get(verdict, '❓')} Fact Check Result: {verdict.replace('_', ' ').title()}
        **Confidence:** {confidence:.1%}
        """)
        
        # Evidence
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Supporting Evidence")
            supporting = fact_check.get('supporting_evidence', [])
            if supporting:
                for evidence in supporting:
                    st.write(f"• {evidence.get('content', 'No content')}")
            else:
                st.write("No supporting evidence found")
        
        with col2:
            st.subheader("Contradicting Evidence")
            contradicting = fact_check.get('contradicting_evidence', [])
            if contradicting:
                for evidence in contradicting:
                    st.write(f"• {evidence.get('content', 'No content')}")
            else:
                st.write("No contradicting evidence found")
    
    def render_system_monitor(self):
        """Render system monitoring interface"""
        st.header("⚙️ System Monitor")
        
        if not st.session_state.orchestrator_initialized:
            st.warning("System not initialized")
            return
        
        try:
            status = run_async(self.orchestrator.get_system_status())
            
            # System overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Orchestrator Status", status.get('orchestrator_status', 'unknown').title())
            with col2:
                agents = status.get('agents', {})
                active_agents = sum(1 for a in agents.values() if a.get('status') == 'operational')
                st.metric("Active Agents", f"{active_agents}/{len(agents)}")
            with col3:
                memory_info = status.get('memory_manager', {})
                total_memories = memory_info.get('short_term_sessions', 0) + memory_info.get('long_term_entries', 0)
                st.metric("Total Memories", total_memories)
            
            # Agent details
            st.subheader("🤖 Agent Status")
            for agent_name, agent_info in agents.items():
                with st.expander(f"{agent_name.upper()} Agent"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Status:** {agent_info.get('status', 'unknown')}")
                        st.write(f"**Success Rate:** {agent_info.get('performance', {}).get('success_rate', 0):.1f}%")
                        st.write(f"**Total Requests:** {agent_info.get('performance', {}).get('total_requests', 0)}")
                    
                    with col2:
                        st.write(f"**Avg Response Time:** {agent_info.get('performance', {}).get('average_response_time', 0):.2f}s")
                        capabilities = agent_info.get('capabilities', [])
                        st.write(f"**Capabilities:** {len(capabilities)}")
                        with st.expander("View Capabilities"):
                            for cap in capabilities:
                                st.write(f"• {cap.replace('_', ' ').title()}")
            
            # Memory manager status
            st.subheader("🧠 Memory Manager")
            memory_info = status.get('memory_manager', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Short-term Sessions", memory_info.get('short_term_sessions', 0))
            with col2:
                st.metric("Long-term Entries", memory_info.get('long_term_entries', 0))
            with col3:
                st.metric("Semantic Concepts", memory_info.get('semantic_concepts', 0))
            
            # Performance metrics
            st.subheader("📈 Performance Metrics")
            performance = status.get('performance_summary', {})
            
            if 'agent_performance' in performance:
                agent_perf = performance['agent_performance']
                
                # Create performance comparison chart
                perf_data = []
                for agent, metrics in agent_perf.items():
                    perf_data.append({
                        'Agent': agent.upper(),
                        'Success Rate': metrics.get('success_rate', 0),
                        'Avg Response Time': metrics.get('average_response_time', 0),
                        'Total Requests': metrics.get('total_requests', 0)
                    })
                
                if perf_data:
                    df = pd.DataFrame(perf_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(df, x='Agent', y='Success Rate', 
                                   title='Agent Success Rates', 
                                   color='Success Rate',
                                   color_continuous_scale='RdYlGn')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df, x='Agent', y='Avg Response Time', 
                                   title='Average Response Times',
                                   color='Avg Response Time',
                                   color_continuous_scale='RdYlBu_r')
                        st.plotly_chart(fig, use_container_width=True)
            
            # System logs
            with st.expander("📋 System Logs"):
                st.text_area("Recent Logs", 
                           "System operational\nAll agents initialized\nMemory manager active", 
                           height=150, disabled=True)
        
        except Exception as e:
            st.error(f"Failed to get system status: {e}")
    
    def render_performance_chart(self):
        """Render performance chart in sidebar"""
        chat_history = st.session_state.chat_history
        if not chat_history:
            st.info("No data to display")
            return
        
        # Extract response times
        response_times = []
        for msg in chat_history:
            if msg['role'] == 'assistant' and 'response_time' in msg:
                response_times.append(msg['response_time'])
        
        if response_times:
            # Simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=response_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#667eea')
            ))
            fig.update_layout(
                title="Response Times",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def initialize_orchestrator(self):
        """Initialize the orchestrator system"""
        with st.spinner("🚀 Initializing AI agents..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize orchestrator
                status_text.text("Initializing orchestrator...")
                progress_bar.progress(25)
                
                run_async(self.orchestrator.initialize())
                
                status_text.text("Loading NLU models...")
                progress_bar.progress(50)
                time.sleep(1)  # Simulate loading time
                
                status_text.text("Loading generative models...")
                progress_bar.progress(75)
                time.sleep(1)
                
                status_text.text("Finalizing setup...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                st.session_state.orchestrator_initialized = True
                
                status_text.empty()
                progress_bar.empty()
                
                st.success("✅ System initialized successfully!")
                st.balloons()
                
                # Force refresh
                time.sleep(1)
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                logger.error(f"Orchestrator initialization failed: {e}")
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate from chat history"""
        assistant_messages = [m for m in st.session_state.chat_history if m['role'] == 'assistant']
        if not assistant_messages:
            return 100.0
        
        successful = sum(1 for m in assistant_messages 
                        if isinstance(m.get('content'), dict) and 'error' not in m['content'])
        
        return (successful / len(assistant_messages)) * 100
    
    def update_metrics(self, response_time: float, success: bool):
        """Update system metrics"""
        metrics = st.session_state.system_metrics
        
        metrics['total_requests'] += 1
        if success:
            metrics['successful_requests'] += 1
        
        # Update average response time
        current_avg = metrics['avg_response_time']
        total_requests = metrics['total_requests']
        metrics['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        metrics['last_update'] = datetime.now()
    
    @staticmethod
    def format_uptime(timestamp_str: str) -> str:
        """Format uptime from timestamp"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            uptime = datetime.now() - timestamp.replace(tzinfo=None)
            
            hours, remainder = divmod(uptime.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            
            return f"{int(hours)}h {int(minutes)}m"
        except:
            return "Unknown"

# Additional utility functions and components
class AdvancedComponents:
    """Advanced UI components for professional features"""
    
    @staticmethod
    def render_code_editor(language: str = "python", theme: str = "monokai"):
        """Render advanced code editor"""
        st.subheader("💻 Code Generation & Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            code_prompt = st.text_area(
                "Describe the code you want to generate:",
                placeholder="E.g., Create a function that calculates fibonacci numbers",
                height=100
            )
        
        with col2:
            selected_language = st.selectbox(
                "Language",
                ["python", "javascript", "java", "cpp", "go", "rust"],
                index=0
            )
            
            complexity_level = st.selectbox(
                "Complexity",
                ["beginner", "intermediate", "advanced"],
                index=1
            )
        
        if st.button("🚀 Generate Code", type="primary"):
            if code_prompt.strip():
                with st.spinner("Generating code..."):
                    # This would integrate with the generative agent
                    generated_code = f"""
# Generated {selected_language.title()} code for: {code_prompt}

def example_function():
    \"\"\"
    This is a placeholder for the generated code.
    In the actual implementation, this would be generated
    by the AI generative agent based on the prompt.
    \"\"\"
    return "Generated code will appear here"

# Example usage
result = example_function()
print(result)
"""
                    
                    st.code(generated_code, language=selected_language)
                    
                    # Code analysis
                    with st.expander("📊 Code Analysis"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Lines of Code", len(generated_code.split('\n')))
                        with col2:
                            st.metric("Complexity", complexity_level.title())
                        with col3:
                            st.metric("Language", selected_language.title())
    
    @staticmethod
    def render_data_visualization_tool():
        """Render data visualization tool"""
        st.subheader("📊 Data Visualization Generator")
        
        # Sample data options
        data_type = st.selectbox(
            "Data Type",
            ["Sample Sales Data", "Performance Metrics", "User Analytics", "Upload Custom Data"]
        )
        
        if data_type == "Upload Custom Data":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file to create visualizations"
            )
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
        else:
            # Generate sample data
            if data_type == "Sample Sales Data":
                dates = pd.date_range('2024-01-01', periods=12, freq='M')
                df = pd.DataFrame({
                    'Month': dates,
                    'Sales': np.random.randint(10000, 50000, 12),
                    'Profit': np.random.randint(2000, 10000, 12),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 12)
                })
            elif data_type == "Performance Metrics":
                df = pd.DataFrame({
                    'Agent': ['NLU', 'Generative', 'Research'],
                    'Success_Rate': [95.2, 87.5, 92.1],
                    'Avg_Response_Time': [1.2, 2.8, 4.5],
                    'Total_Requests': [150, 120, 85]
                })
            else:  # User Analytics
                df = pd.DataFrame({
                    'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
                    'Active_Users': np.random.randint(100, 1000, 30),
                    'Sessions': np.random.randint(200, 2000, 30),
                    'Bounce_Rate': np.random.uniform(0.2, 0.8, 30)
                })
        
        if 'df' in locals():
            # Visualization options
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_column = st.selectbox("X-axis", df.columns)
            with col2:
                y_column = st.selectbox("Y-axis", [col for col in df.columns if col != x_column])
            
            # Generate visualization
            try:
                if chart_type == "Line Chart":
                    fig = px.line(df, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=y_column, title=f"Distribution of {y_column}")
                elif chart_type == "Box Plot":
                    fig = px.box(df, y=y_column, title=f"Box Plot of {y_column}")
                else:  # Heatmap
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        correlation_matrix = numeric_df.corr()
                        fig = px.imshow(correlation_matrix, title="Correlation Heatmap")
                    else:
                        st.warning("No numeric columns available for heatmap")
                        fig = None
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    if st.button("📥 Download Chart"):
                        st.info("Chart download functionality would be implemented here")
            
            except Exception as e:
                st.error(f"Error creating visualization: {e}")

def main():
    """Main application entry point"""
    try:
        app = StreamlitApp()
        app.run()
        
        # Additional features in sidebar
        with st.sidebar:
            st.markdown("---")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔄 Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()
            
            if st.button("📊 Export Analytics"):
                st.info("Analytics export functionality would be implemented here")
            
            if st.button("💾 Save Session"):
                st.info("Session save functionality would be implemented here")
            
            # Advanced tools
            with st.expander("🛠️ Advanced Tools"):
                AdvancedComponents.render_code_editor()
                AdvancedComponents.render_data_visualization_tool()
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; font-size: 0.8em;'>
                🤖 AI Multi-Agent Research Assistant v2.0<br>
                Built with advanced NLP, GenAI & Multi-Agent Architecture
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Streamlit app error: {e}")

if __name__ == "__main__":
    # Configure logging for the web app
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
