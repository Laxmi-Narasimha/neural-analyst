"""Question Generation Agent - generates clarifying questions for data adequacy assessment."""

import logging
import json
from typing import Dict, List, Any, Optional
import asyncio

from openai import OpenAI
from ..config import config

logger = logging.getLogger(__name__)


class QuestionGenerationAgent:
    """
    Question Generation Agent (QGn) - Core agent for generating clarifying questions.
    
    Responsibilities:
    - Accept user goal and domain context
    - Generate prioritized clarifying questions
    - Map questions to potential failure modes
    - Provide expected evidence descriptions
    - Support iterative question refinement
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.MODELS["research"]  # Use research model for deep thinking
        self.max_retries = 3
        self.question_templates = self._load_question_templates()
    
    async def generate_questions(self, 
                               user_goal: str, 
                               domain: str = "general", 
                               sample_text: str = None,
                               existing_files: List[str] = None) -> Dict[str, Any]:
        """
        Generate clarifying questions based on user goal and context.
        
        Args:
            user_goal: User's description of their AI assistant goal
            domain: Domain context (automotive, manufacturing, real_estate, general)
            sample_text: Optional sample from provided documents
            existing_files: List of existing file names for context
            
        Returns:
            Dictionary containing generated questions and metadata
        """
        try:
            # Build the prompt
            system_prompt = self._build_system_prompt(domain)
            user_prompt = self._build_user_prompt(user_goal, domain, sample_text, existing_files)
            
            # Call OpenAI with function calling for structured output
            response = await self._call_openai_with_retry(system_prompt, user_prompt)
            
            if not response:
                return self._create_error_result("Failed to generate questions after retries")
            
            # Parse and validate the response
            questions_data = self._parse_and_validate_response(response, user_goal, domain)
            
            # Enhance with domain-specific insights
            enhanced_questions = self._enhance_questions_with_domain_knowledge(
                questions_data, domain, user_goal
            )
            
            return {
                "success": True,
                "questions": enhanced_questions,
                "domain": domain,
                "user_goal": user_goal,
                "total_questions": len(enhanced_questions),
                "priority_breakdown": self._get_priority_breakdown(enhanced_questions),
                "estimated_time_minutes": len(enhanced_questions) * 2,  # 2 minutes per question
                "recommendations": self._generate_question_recommendations(enhanced_questions, domain)
            }
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return self._create_error_result(f"Question generation failed: {str(e)}")
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build the system prompt for question generation."""
        domain_context = config.get_domain_config(domain)
        
        return f"""You are an expert question generation assistant for AI data adequacy assessment. Your role is to generate precise, actionable clarifying questions that will help determine whether provided data is sufficient for a user's AI assistant goal.

CONTEXT:
- Domain: {domain}
- Required fields for this domain: {domain_context.get('required_fields', [])}
- Staleness thresholds: {domain_context.get('staleness_threshold_days', 365)} days general, {domain_context.get('pricing_staleness_days', 90)} days pricing

INSTRUCTIONS:
1. Generate 8-15 clarifying questions that directly impact data adequacy assessment
2. Each question should map to specific failure modes that could block AI assistant deployment
3. Focus on completeness, accuracy, timeliness, and consistency concerns
4. Prioritize questions that reveal critical missing data or quality issues
5. Include questions about expected user interactions and edge cases

FAILURE MODES TO CONSIDER:
- MISSING_COVERAGE: Essential topics/scenarios not covered
- TIMELINESS: Data staleness affecting accuracy
- INCONSISTENT_DATA: Contradictory information
- INCOMPLETE_ENTITIES: Missing required fields or attributes
- QUALITY_ISSUES: Formatting, parsing, or accuracy problems
- SCOPE_MISMATCH: Data doesn't match intended use case
- COMPLIANCE_GAPS: Missing regulatory or business-critical information

RESPONSE FORMAT:
Return valid JSON with this structure:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Clear, specific question text",
      "priority": "critical|high|medium|low",
      "failure_mode": "MISSING_COVERAGE|TIMELINESS|etc",
      "expected_evidence": "Description of what evidence would satisfy this question",
      "rationale": "Why this question is important for data adequacy"
    }}
  ]
}}"""
    
    def _build_user_prompt(self, 
                          user_goal: str, 
                          domain: str, 
                          sample_text: str = None,
                          existing_files: List[str] = None) -> str:
        """Build the user prompt with context."""
        prompt_parts = [
            f"USER GOAL: {user_goal}",
            f"DOMAIN: {domain}"
        ]
        
        if existing_files:
            file_list = ", ".join(existing_files[:5])  # First 5 files
            if len(existing_files) > 5:
                file_list += f" (and {len(existing_files) - 5} more files)"
            prompt_parts.append(f"EXISTING FILES: {file_list}")
        
        if sample_text:
            sample_preview = sample_text[:500] + "..." if len(sample_text) > 500 else sample_text
            prompt_parts.append(f"SAMPLE CONTENT: {sample_preview}")
        
        # Add domain-specific context
        domain_templates = self.question_templates.get(domain, [])
        if domain_templates:
            prompt_parts.append(f"DOMAIN CONSIDERATIONS: {', '.join(domain_templates[:3])}")
        
        return "\n\n".join(prompt_parts)
    
    async def _call_openai_with_retry(self, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """Call OpenAI API with retry logic."""
        function_schema = {
            "name": "generate_questions",
            "description": "Generate clarifying questions for data adequacy assessment",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "failure_mode": {"type": "string"},
                                "expected_evidence": {"type": "string"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["id", "text", "priority", "failure_mode", "expected_evidence", "rationale"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    functions=[function_schema],
                    function_call={"name": "generate_questions"},
                    temperature=0.3,
                    max_tokens=2000
                )
                
                function_call = response.choices[0].message.function_call
                if function_call and function_call.name == "generate_questions":
                    return json.loads(function_call.arguments)
                else:
                    # Fallback to content parsing
                    content = response.choices[0].message.content
                    if content and "{" in content:
                        # Try to extract JSON from content
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            return json.loads(content[start:end])
                
                logger.warning(f"Unexpected response format on attempt {attempt + 1}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _parse_and_validate_response(self, response: Dict, user_goal: str, domain: str) -> List[Dict]:
        """Parse and validate the OpenAI response."""
        questions = response.get("questions", [])
        
        if not questions:
            # Generate fallback questions
            return self._generate_fallback_questions(user_goal, domain)
        
        validated_questions = []
        for i, q in enumerate(questions):
            # Ensure required fields
            question = {
                "id": q.get("id", f"q{i+1}"),
                "text": q.get("text", "").strip(),
                "priority": q.get("priority", "medium").lower(),
                "failure_mode": q.get("failure_mode", "MISSING_COVERAGE"),
                "expected_evidence": q.get("expected_evidence", "").strip(),
                "rationale": q.get("rationale", "").strip()
            }
            
            # Validate priority
            if question["priority"] not in ["critical", "high", "medium", "low"]:
                question["priority"] = "medium"
            
            # Ensure minimum text length
            if len(question["text"]) < 10:
                continue
            
            validated_questions.append(question)
        
        # Ensure we have at least some questions
        if len(validated_questions) < 3:
            fallback_questions = self._generate_fallback_questions(user_goal, domain)
            validated_questions.extend(fallback_questions[len(validated_questions):])
        
        return validated_questions
    
    def _enhance_questions_with_domain_knowledge(self, 
                                               questions: List[Dict], 
                                               domain: str, 
                                               user_goal: str) -> List[Dict]:
        """Enhance questions with domain-specific knowledge."""
        domain_config = config.get_domain_config(domain)
        
        enhanced = []
        for question in questions:
            # Add domain-specific context to rationale
            if domain != "general":
                domain_context = f" (Domain: {domain})"
                question["rationale"] += domain_context
            
            # Add expected timeframes for timeliness questions
            if "TIMELINESS" in question["failure_mode"]:
                staleness_days = domain_config.get("staleness_threshold_days", 365)
                question["expected_evidence"] += f" Data should be newer than {staleness_days} days."
            
            # Add required fields context
            required_fields = domain_config.get("required_fields", [])
            if required_fields and "INCOMPLETE_ENTITIES" in question["failure_mode"]:
                question["expected_evidence"] += f" Should include: {', '.join(required_fields)}."
            
            enhanced.append(question)
        
        return enhanced
    
    def _generate_fallback_questions(self, user_goal: str, domain: str) -> List[Dict]:
        """Generate fallback questions when AI generation fails."""
        fallback = [
            {
                "id": "fallback_1",
                "text": f"Does the provided data comprehensively cover all aspects needed for: {user_goal}?",
                "priority": "critical",
                "failure_mode": "MISSING_COVERAGE",
                "expected_evidence": "Documentation covering all key topics and use cases mentioned in the goal.",
                "rationale": "Ensures data completeness for the specified use case."
            },
            {
                "id": "fallback_2",
                "text": "How recent is the data, and are there any time-sensitive elements?",
                "priority": "high",
                "failure_mode": "TIMELINESS",
                "expected_evidence": "Clear timestamps and indication of data freshness requirements.",
                "rationale": "Outdated data can lead to incorrect AI responses."
            },
            {
                "id": "fallback_3",
                "text": "Are there any contradictory or inconsistent pieces of information?",
                "priority": "high",
                "failure_mode": "INCONSISTENT_DATA",
                "expected_evidence": "Data consistency across all sources with no conflicting facts.",
                "rationale": "Inconsistent data can confuse the AI and produce unreliable outputs."
            },
            {
                "id": "fallback_4",
                "text": "What are the typical edge cases or unusual scenarios users might ask about?",
                "priority": "medium",
                "failure_mode": "MISSING_COVERAGE",
                "expected_evidence": "Documentation of edge cases, exceptions, and unusual scenarios.",
                "rationale": "Edge cases often reveal gaps in data coverage."
            },
            {
                "id": "fallback_5",
                "text": "Are there any regulatory, legal, or compliance requirements that must be addressed?",
                "priority": "medium",
                "failure_mode": "COMPLIANCE_GAPS",
                "expected_evidence": "Relevant compliance documentation and legal requirements.",
                "rationale": "Compliance gaps can create legal and business risks."
            }
        ]
        
        return fallback
    
    def _get_priority_breakdown(self, questions: List[Dict]) -> Dict[str, int]:
        """Get breakdown of questions by priority."""
        breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for q in questions:
            priority = q.get("priority", "medium")
            breakdown[priority] += 1
        return breakdown
    
    def _generate_question_recommendations(self, questions: List[Dict], domain: str) -> List[str]:
        """Generate recommendations for using the questions effectively."""
        recommendations = []
        
        critical_count = sum(1 for q in questions if q["priority"] == "critical")
        high_count = sum(1 for q in questions if q["priority"] == "high")
        
        if critical_count > 0:
            recommendations.append(f"Address all {critical_count} critical questions before proceeding - these represent deployment blockers.")
        
        if high_count > 0:
            recommendations.append(f"Strongly consider addressing {high_count} high-priority questions to ensure quality.")
        
        recommendations.append("Review expected evidence descriptions to understand what documentation gaps need filling.")
        
        if domain != "general":
            recommendations.append(f"Domain-specific ({domain}) considerations have been included - pay attention to industry requirements.")
        
        return recommendations
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load domain-specific question templates."""
        return {
            "automotive": [
                "VIN and vehicle identification completeness",
                "Pricing and inventory accuracy",
                "Safety and recall information",
                "Manufacturer specifications and warranties"
            ],
            "manufacturing": [
                "Product specifications and tolerances",
                "Compliance and safety standards",
                "Supply chain and inventory data",
                "Quality control processes"
            ],
            "real_estate": [
                "Property details and square footage",
                "Pricing and market conditions",
                "Legal and zoning information",
                "Historical transaction data"
            ],
            "general": [
                "Data completeness and coverage",
                "Information accuracy and timeliness",
                "Consistency and conflicting data",
                "Edge cases and exceptions"
            ]
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "success": False,
            "error": error_message,
            "questions": [],
            "recommendations": ["Manual question generation recommended due to system error"]
        }
    
    async def refine_questions(self, 
                               original_questions: List[Dict], 
                               user_feedback: str,
                               additional_context: str = None) -> Dict[str, Any]:
        """Refine questions based on user feedback."""
        try:
            refinement_prompt = f"""
            Original questions generated: {len(original_questions)}
            User feedback: {user_feedback}
            Additional context: {additional_context or 'None'}
            
            Please refine the question set based on the feedback. Focus on:
            1. Addressing user's specific concerns
            2. Removing irrelevant questions
            3. Adding missing aspects they mentioned
            4. Adjusting priority levels based on their input
            """
            
            # This would call OpenAI again with the refinement prompt
            # Implementation similar to generate_questions but with refinement context
            return {
                "success": True,
                "message": "Question refinement would be implemented here",
                "refined_questions": original_questions  # Placeholder
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Question refinement failed: {str(e)}"
            }


# Convenience function
async def generate_clarifying_questions(user_goal: str, 
                                      domain: str = "general",
                                      sample_text: str = None,
                                      existing_files: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to generate questions.
    
    Args:
        user_goal: User's AI assistant goal description
        domain: Domain context
        sample_text: Optional sample content
        existing_files: List of existing file names
        
    Returns:
        Question generation result
    """
    agent = QuestionGenerationAgent()
    return await agent.generate_questions(user_goal, domain, sample_text, existing_files)
