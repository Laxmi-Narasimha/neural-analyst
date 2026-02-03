"""Data Adequacy Manager - central brain coordinating all agents with ReAct loop."""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from .agents.qgen import QuestionGenerationAgent
from .agents.ingestion import DataIngestionAgent
from .agents.quality import QualityAnalysisAgent
from .agents.validation import ValidationResultsAgent
from .config import config

logger = logging.getLogger(__name__)


class DataAdequacyManager:
    """
    Data Adequacy Manager - Central coordination system (formerly OrchestratorAgent).
    
    Responsibilities:
    - Initialize and manage session state
    - Sequence agent calls (QGn → DI → QA → VR)
    - Implement ReAct loop (Reason, Act, Observe)
    - Handle user interactions and clarifying questions
    - Enforce cost control and retry logic
    - Maintain session memory and context
    """
    
    def __init__(self):
        self.qgen_agent = QuestionGenerationAgent()
        self.ingestion_agent = DataIngestionAgent()
        self.quality_agent = QualityAnalysisAgent()
        self.validation_agent = ValidationResultsAgent()
        
        self.session_state = {}
        self.llm_call_count = 0
        self.max_llm_calls = config.MAX_LLM_CALLS_PER_SESSION
        
    async def run_validation(self, 
                           user_goal: str,
                           domain: str = "general",
                           files: List[str] = None,
                           pinecone_namespace: str = None,
                           user_answers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Main validation pipeline coordinating all agents.
        
        Args:
            user_goal: User's description of AI assistant goal
            domain: Domain context (automotive, manufacturing, real_estate, general)
            files: List of file paths to process
            pinecone_namespace: Optional existing namespace
            user_answers: Optional answers to clarifying questions
            
        Returns:
            Complete validation results with reports and recommendations
        """
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting validation session {session_id}")
        
        # Initialize session state
        self.session_state = {
            "session_id": session_id,
            "user_goal": user_goal,
            "domain": domain,
            "files": files or [],
            "namespace": pinecone_namespace or f"session_{session_id}",
            "start_time": datetime.now().isoformat(),
            "llm_calls": 0,
            "agent_results": {},
            "current_step": "initialization"
        }
        
        try:
            # Step 0: Generate clarifying questions (if no answers provided)
            if not user_answers:
                logger.info("Step 0: Generating clarifying questions...")
                self.session_state["current_step"] = "question_generation"
                
                questions_result = await self._reason_and_act(
                    "generate_clarifying_questions",
                    user_goal=user_goal,
                    domain=domain,
                    files=files
                )
                
                if not questions_result["success"]:
                    return self._create_error_result("Question generation failed", questions_result)
                
                # Return questions for user interaction
                return {
                    "success": True,
                    "session_id": session_id,
                    "step": "clarifying_questions",
                    "questions": questions_result["questions"],
                    "estimated_time": questions_result.get("estimated_time_minutes", 10),
                    "next_action": "Please answer the questions and call continue_validation"
                }
            
            # Continue with validation using provided answers
            return await self._continue_validation_pipeline(user_answers)
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            return self._create_error_result(f"Pipeline execution failed: {str(e)}")
    
    async def continue_validation(self, user_answers: Dict[str, str]) -> Dict[str, Any]:
        """Continue validation after receiving user answers to clarifying questions."""
        if not self.session_state:
            return self._create_error_result("No active session found")
        
        return await self._continue_validation_pipeline(user_answers)
    
    async def _continue_validation_pipeline(self, user_answers: Dict[str, str]) -> Dict[str, Any]:
        """Continue the validation pipeline with user answers."""
        try:
            # Step 1: Process user answers and refine goal
            logger.info("Step 1: Processing user answers...")
            self.session_state["current_step"] = "goal_refinement"
            
            refined_goal = self._refine_goal_with_answers(
                self.session_state["user_goal"],
                self.session_state["domain"],
                user_answers
            )
            
            self.session_state["refined_goal"] = refined_goal
            
            # Step 2: Data ingestion
            logger.info("Step 2: Starting data ingestion...")
            self.session_state["current_step"] = "data_ingestion"
            
            ingestion_result = await self._reason_and_act(
                "ingest_data",
                files=self.session_state["files"],
                namespace=self.session_state["namespace"],
                goal=refined_goal
            )
            
            if not ingestion_result["success"]:
                return self._create_error_result("Data ingestion failed", ingestion_result)
            
            self.session_state["agent_results"]["ingestion"] = ingestion_result
            
            # Early exit check
            if ingestion_result["stats"]["chunks_created"] == 0:
                return self._create_error_result(
                    "No content was successfully processed. Please check your files and try again.",
                    {"ingestion_result": ingestion_result}
                )
            
            # Step 3: Quality analysis
            logger.info("Step 3: Running quality analysis...")
            self.session_state["current_step"] = "quality_analysis"
            
            quality_result = await self._reason_and_act(
                "analyze_quality",
                ingest_summary=ingestion_result,
                goal=refined_goal,
                namespace=self.session_state["namespace"]
            )
            
            if not quality_result["success"]:
                return self._create_error_result("Quality analysis failed", quality_result)
            
            self.session_state["agent_results"]["quality"] = quality_result
            
            # Step 4: Generate validation report
            logger.info("Step 4: Generating validation report...")
            self.session_state["current_step"] = "report_generation"
            
            report_result = await self._reason_and_act(
                "generate_report",
                quality_report=quality_result,
                goal=refined_goal,
                session_id=self.session_state["session_id"]
            )
            
            if not report_result["success"]:
                return self._create_error_result("Report generation failed", report_result)
            
            self.session_state["agent_results"]["validation"] = report_result
            self.session_state["current_step"] = "completed"
            
            # Compile final results
            final_results = {
                "success": True,
                "session_id": self.session_state["session_id"],
                "readiness_level": report_result["readiness_level"],
                "composite_score": report_result["composite_score"],
                "executive_summary": report_result["summary"],
                "markdown_report": report_result["markdown_result"],
                "json_report": report_result["json_report"],
                "top_recommendations": report_result["recommendations"],
                "next_steps": report_result["next_steps"],
                "session_summary": self._generate_session_summary(),
                "technical_details": {
                    "namespace": self.session_state["namespace"],
                    "files_processed": len(self.session_state["files"]),
                    "chunks_created": ingestion_result["stats"]["chunks_created"],
                    "llm_calls_used": self.llm_call_count,
                    "processing_time": self._calculate_processing_time()
                }
            }
            
            logger.info(f"Validation completed successfully for session {self.session_state['session_id']}")
            return final_results
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            return self._create_error_result(f"Pipeline execution failed: {str(e)}")
    
    async def _reason_and_act(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        ReAct loop implementation: Reason about action, Act, Observe results.
        
        Args:
            action: Action to perform
            **kwargs: Action-specific parameters
            
        Returns:
            Action result with observation
        """
        # Reason: Check if we should perform this action
        if not self._should_perform_action(action):
            return {"success": False, "error": f"Action {action} not allowed in current state"}
        
        # Check LLM call limits
        if self.llm_call_count >= self.max_llm_calls:
            return {"success": False, "error": "LLM call limit exceeded"}
        
        # Act: Perform the action
        try:
            if action == "generate_clarifying_questions":
                result = await self.qgen_agent.generate_questions(
                    kwargs["user_goal"],
                    kwargs["domain"],
                    sample_text=None,
                    existing_files=[f.split('/')[-1] for f in kwargs.get("files", [])]
                )
                self.llm_call_count += 1
                
            elif action == "ingest_data":
                result = await self.ingestion_agent.ingest(
                    kwargs["files"],
                    kwargs["namespace"],
                    kwargs["goal"]
                )
                self.llm_call_count += 2  # Embedding calls
                
            elif action == "analyze_quality":
                result = await self.quality_agent.run_all_checks(
                    kwargs["ingest_summary"],
                    kwargs["goal"],
                    kwargs["namespace"]
                )
                self.llm_call_count += 5  # Multiple analysis calls
                
            elif action == "generate_report":
                result = self.validation_agent.compile_report(
                    kwargs["quality_report"],
                    kwargs["goal"],
                    kwargs["session_id"]
                )
                # No LLM calls for report generation
                
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
            
            # Observe: Log and validate results
            self._observe_action_result(action, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Action {action} failed: {str(e)}")
            return {"success": False, "error": f"Action execution failed: {str(e)}"}
    
    def _should_perform_action(self, action: str) -> bool:
        """Reason about whether an action should be performed."""
        current_step = self.session_state.get("current_step")
        
        # Define valid action sequences
        valid_sequences = {
            "initialization": ["generate_clarifying_questions"],
            "question_generation": ["ingest_data"],
            "goal_refinement": ["ingest_data"],
            "data_ingestion": ["analyze_quality"],
            "quality_analysis": ["generate_report"],
            "report_generation": []
        }
        
        allowed_actions = valid_sequences.get(current_step, [])
        return action in allowed_actions or current_step == "completed"
    
    def _observe_action_result(self, action: str, result: Dict[str, Any]):
        """Observe and log action results."""
        success = result.get("success", False)
        
        logger.info(f"Action '{action}' completed with success={success}")
        
        if not success:
            error = result.get("error", "Unknown error")
            logger.warning(f"Action '{action}' failed: {error}")
        
        # Update session state with observations
        self.session_state["llm_calls"] = self.llm_call_count
        
        # Add to action history
        if "action_history" not in self.session_state:
            self.session_state["action_history"] = []
        
        self.session_state["action_history"].append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "llm_calls_used": 1 if "llm" in action else 0
        })
    
    def _refine_goal_with_answers(self, user_goal: str, domain: str, user_answers: Dict[str, str]) -> Dict[str, Any]:
        """Refine user goal with clarifying question answers."""
        return {
            "original_goal": user_goal,
            "domain": domain,
            "description": user_goal,
            "user_answers": user_answers,
            "refined_requirements": self._extract_requirements_from_answers(user_answers),
            "success_criteria": self._generate_success_criteria_from_answers(user_answers, domain)
        }
    
    def _extract_requirements_from_answers(self, user_answers: Dict[str, str]) -> List[str]:
        """Extract specific requirements from user answers."""
        requirements = []
        
        for question_id, answer in user_answers.items():
            if answer and len(answer.strip()) > 5:
                # Simple extraction - could be enhanced with NLP
                if any(keyword in answer.lower() for keyword in ["must", "required", "need", "essential"]):
                    requirements.append(f"User requirement: {answer[:100]}")
        
        return requirements
    
    def _generate_success_criteria_from_answers(self, user_answers: Dict[str, str], domain: str) -> List[str]:
        """Generate success criteria based on user answers."""
        criteria = [
            "System can answer user questions accurately",
            "Information is current and relevant",
            "No critical data gaps exist"
        ]
        
        # Add domain-specific criteria
        domain_criteria = {
            "automotive": ["Vehicle information is complete and accurate"],
            "manufacturing": ["Product specifications are comprehensive"],
            "real_estate": ["Property details are accurate and current"],
            "general": ["Content coverage meets user expectations"]
        }
        
        if domain in domain_criteria:
            criteria.extend(domain_criteria[domain])
        
        return criteria
    
    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate summary of the entire validation session."""
        results = self.session_state.get("agent_results", {})
        
        return {
            "session_id": self.session_state["session_id"],
            "total_processing_time": self._calculate_processing_time(),
            "files_processed": len(self.session_state["files"]),
            "llm_calls_used": self.llm_call_count,
            "steps_completed": len(self.session_state.get("action_history", [])),
            "final_step": self.session_state.get("current_step", "unknown"),
            "agent_performance": {
                "ingestion": "success" if results.get("ingestion", {}).get("success") else "failed",
                "quality_analysis": "success" if results.get("quality", {}).get("success") else "failed",
                "report_generation": "success" if results.get("validation", {}).get("success") else "failed"
            }
        }
    
    def _calculate_processing_time(self) -> str:
        """Calculate total processing time."""
        if "start_time" not in self.session_state:
            return "unknown"
        
        start_time = datetime.fromisoformat(self.session_state["start_time"])
        processing_time = datetime.now() - start_time
        
        total_seconds = int(processing_time.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _create_error_result(self, message: str, additional_data: Dict = None) -> Dict[str, Any]:
        """Create standardized error result."""
        result = {
            "success": False,
            "error": message,
            "session_id": self.session_state.get("session_id", "unknown"),
            "current_step": self.session_state.get("current_step", "unknown"),
            "llm_calls_used": self.llm_call_count,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_data:
            result["additional_info"] = additional_data
        
        return result
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "session_id": self.session_state.get("session_id"),
            "current_step": self.session_state.get("current_step"),
            "llm_calls_used": self.llm_call_count,
            "max_llm_calls": self.max_llm_calls,
            "processing_time": self._calculate_processing_time(),
            "files_count": len(self.session_state.get("files", [])),
            "namespace": self.session_state.get("namespace")
        }
    
    async def test_connections(self) -> Dict[str, Any]:
        """Test connections to external services."""
        return await self.ingestion_agent.test_connection()


# Convenience functions
async def start_validation(user_goal: str, domain: str = "general", files: List[str] = None) -> Dict[str, Any]:
    """Start a new validation session."""
    manager = DataAdequacyManager()
    return await manager.run_validation(user_goal, domain, files)


async def continue_session_validation(manager: DataAdequacyManager, user_answers: Dict[str, str]) -> Dict[str, Any]:
    """Continue an existing validation session."""
    return await manager.continue_validation(user_answers)
