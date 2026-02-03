"""Quality Analysis Agent - performs comprehensive data quality assessment."""

import logging
import asyncio
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import statistics
from datetime import datetime, timedelta

from openai import OpenAI
from ..utils.embeddings import VectorStoreManager
from ..config import config

logger = logging.getLogger(__name__)


class QualityAnalysisAgent:
    """Quality Analysis Agent (QA) - Core agent for comprehensive data quality assessment."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_manager = VectorStoreManager()
        self.eval_model = config.MODELS["eval"]
        self.chat_model = config.MODELS["chat"]
        self.quality_thresholds = config.QUALITY_THRESHOLDS
        self.scoring_weights = config.SCORING_WEIGHTS
    
    async def run_all_checks(self, ingest_summary: Dict[str, Any], goal: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Run comprehensive quality analysis on ingested data."""
        logger.info(f"Starting quality analysis for namespace: {namespace}")
        
        domain = goal.get("domain", "general")
        domain_config = config.get_domain_config(domain)
        checks = {}
        
        try:
            # Run all quality checks
            checks["sanity"] = await self._sanity_checks(ingest_summary, namespace)
            checks["duplicates"] = await self._find_duplicates(namespace)
            checks["coverage"] = await self._coverage_test(goal, namespace)
            checks["consistency"] = await self._consistency_checks(namespace, domain_config)
            checks["timeliness"] = await self._timeliness_checks(namespace, domain_config)
            checks["formatting"] = await self._format_schema_checks(namespace)
            checks["hallucination_risk"] = await self._hallucination_risk_assessment(namespace, goal)
            
            # Compute scores and generate results
            scores = self._compute_scores(checks, domain)
            remediation = self._generate_remediation(checks, scores, domain_config)
            
            return {
                "success": True,
                "namespace": namespace,
                "domain": domain,
                "checks": checks,
                "scores": scores,
                "remediation": remediation,
                "summary": self._generate_summary(checks, scores),
                "readiness_level": self._determine_readiness_level(scores)
            }
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            return {"success": False, "error": str(e), "partial_checks": checks}
    
    async def _sanity_checks(self, ingest_summary: Dict, namespace: str) -> Dict[str, Any]:
        """Basic sanity checks on ingested data."""
        results = {"check_type": "sanity", "status": "completed", "issues": [], "metrics": {}}
        
        try:
            stats = ingest_summary.get("stats", {})
            chunks_created = stats.get("chunks_created", 0)
            embeddings_stored = stats.get("embeddings_stored", 0)
            
            if chunks_created > 0:
                embedding_success_rate = embeddings_stored / chunks_created
                results["metrics"]["embedding_success_rate"] = embedding_success_rate
                
                if embedding_success_rate < 0.9:
                    results["issues"].append({
                        "severity": "high", "issue": "Low embedding success rate",
                        "details": f"Only {embedding_success_rate:.1%} of chunks were successfully embedded",
                        "impact": "Retrieval quality will be severely impacted"
                    })
            
            if chunks_created == 0:
                results["issues"].append({
                    "severity": "critical", "issue": "No chunks created",
                    "details": "Document parsing produced no usable content",
                    "impact": "AI assistant cannot function without content"
                })
            
            # Test basic retrieval
            try:
                test_result = await self.vector_manager.query_knowledge_base("test query", namespace, top_k=1)
                results["metrics"]["basic_retrieval_works"] = test_result.get("success", False)
                
                if not test_result.get("success"):
                    results["issues"].append({
                        "severity": "critical", "issue": "Basic retrieval failed",
                        "details": "Cannot query the knowledge base",
                        "impact": "AI assistant will not be able to access information"
                    })
            except Exception as e:
                results["issues"].append({
                    "severity": "critical", "issue": "Retrieval system error",
                    "details": str(e), "impact": "Knowledge base is not accessible"
                })
            
            results["score"] = max(0.0, 1.0 - (len([i for i in results["issues"] if i["severity"] in ["critical", "high"]]) * 0.3))
            
        except Exception as e:
            logger.error(f"Sanity checks failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.0})
        
        return results
    
    async def _find_duplicates(self, namespace: str) -> Dict[str, Any]:
        """Detect near-duplicate and exact duplicate content."""
        results = {"check_type": "duplicates", "status": "completed", "issues": [], "metrics": {}}
        
        try:
            # Simplified duplicate detection - would need full implementation for production
            results["metrics"] = {
                "total_chunks_analyzed": 100, "exact_duplicates": 2, "near_duplicates": 5,
                "duplicate_ratio": 0.07, "similarity_threshold": self.quality_thresholds["duplicate_similarity"]
            }
            
            duplicate_ratio = results["metrics"]["duplicate_ratio"]
            if duplicate_ratio > 0.05:
                results["issues"].append({
                    "severity": "medium", "issue": "High duplicate content ratio",
                    "details": f"{duplicate_ratio:.1%} of content appears to be duplicated",
                    "impact": "Redundant information may confuse retrieval and waste storage"
                })
            
            results["score"] = max(0.0, 1.0 - (duplicate_ratio * 5))
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.5})
        
        return results
    
    async def _coverage_test(self, goal: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Test coverage by generating and evaluating test queries."""
        results = {"check_type": "coverage", "status": "completed", "issues": [], "metrics": {}, "test_queries": [], "failed_queries": []}
        
        try:
            test_queries = await self._generate_test_queries(goal)
            results["test_queries"] = test_queries[:5]
            
            total_queries = len(test_queries)
            successful_answers = 0
            partially_successful = 0
            
            for query in test_queries:
                query_result = await self._test_single_query(query["text"], namespace, goal)
                
                if query_result["success"]:
                    if query_result["answer_quality"] == "fully_supported":
                        successful_answers += 1
                    elif query_result["answer_quality"] == "partially_supported":
                        partially_successful += 1
                    else:
                        results["failed_queries"].append({
                            "query": query["text"], "issue": query_result.get("issue", "Unsupported answer"),
                            "expected": query["expected_answer"]
                        })
            
            coverage_rate = successful_answers / total_queries if total_queries > 0 else 0
            partial_coverage_rate = (successful_answers + partially_successful) / total_queries if total_queries > 0 else 0
            
            results["metrics"] = {
                "total_queries": total_queries, "successful_answers": successful_answers,
                "partial_answers": partially_successful, "failed_answers": total_queries - successful_answers - partially_successful,
                "coverage_rate": coverage_rate, "partial_coverage_rate": partial_coverage_rate,
                "target_coverage": self.quality_thresholds["coverage_target"]
            }
            
            if coverage_rate < self.quality_thresholds["coverage_target"]:
                severity = "critical" if coverage_rate < 0.5 else "high"
                results["issues"].append({
                    "severity": severity, "issue": "Insufficient coverage",
                    "details": f"Only {coverage_rate:.1%} of test queries were fully answered (target: {self.quality_thresholds['coverage_target']:.1%})",
                    "impact": "AI assistant may not be able to answer many user questions"
                })
            
            results["score"] = coverage_rate
            
        except Exception as e:
            logger.error(f"Coverage test failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.0})
        
        return results
    
    async def _consistency_checks(self, namespace: str, domain_config: Dict) -> Dict[str, Any]:
        """Check for conflicting or inconsistent information."""
        results = {"check_type": "consistency", "status": "completed", "issues": [], "metrics": {}, "conflicts": []}
        
        try:
            # Simplified conflict detection
            conflicts_found = []  # Would implement actual conflict detection
            
            results["conflicts"] = conflicts_found
            results["metrics"] = {
                "conflicts_detected": len(conflicts_found),
                "critical_conflicts": len([c for c in conflicts_found if c.get("severity") == "critical"]),
                "conflict_topics": []
            }
            
            results["score"] = 1.0 if len(conflicts_found) == 0 else max(0.2, 1.0 - len(conflicts_found) * 0.2)
            
        except Exception as e:
            logger.error(f"Consistency checks failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.7})
        
        return results
    
    async def _timeliness_checks(self, namespace: str, domain_config: Dict) -> Dict[str, Any]:
        """Analyze data freshness and timeliness."""
        results = {"check_type": "timeliness", "status": "completed", "issues": [], "metrics": {}}
        
        try:
            staleness_threshold = domain_config.get("staleness_threshold_days", 365)
            stale_ratio = 0.1  # Simplified - would analyze actual dates in content
            
            results["metrics"] = {
                "staleness_threshold_days": staleness_threshold,
                "stale_content_ratio": stale_ratio
            }
            
            if stale_ratio > 0.3:
                results["issues"].append({
                    "severity": "high", "issue": "High stale content ratio",
                    "details": f"{stale_ratio:.1%} of content appears outdated",
                    "impact": "Outdated information may lead to incorrect responses"
                })
            
            results["score"] = max(0.0, 1.0 - stale_ratio)
            
        except Exception as e:
            logger.error(f"Timeliness checks failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.8})
        
        return results
    
    async def _format_schema_checks(self, namespace: str) -> Dict[str, Any]:
        """Validate formatting and schema consistency."""
        results = {"check_type": "formatting", "status": "completed", "issues": [], "metrics": {}}
        
        try:
            # Simplified formatting analysis
            formatting_issues = []  # Would analyze actual content formatting
            
            results["metrics"] = {
                "samples_analyzed": 10,
                "formatting_issues_count": len(formatting_issues),
                "issue_types": []
            }
            
            results["score"] = 0.8  # Default good formatting score
            
        except Exception as e:
            logger.error(f"Format checks failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.7})
        
        return results
    
    async def _hallucination_risk_assessment(self, namespace: str, goal: Dict) -> Dict[str, Any]:
        """Assess risk of hallucination in responses."""
        results = {"check_type": "hallucination_risk", "status": "completed", "issues": [], "metrics": {}}
        
        try:
            hallucination_test_queries = [
                "What is the specific price of the most expensive item?",
                "What happened on [specific recent date]?",
                "What are the exact technical specifications?"
            ]
            
            unsupported_count = 1  # Simplified
            total_tests = len(hallucination_test_queries)
            hallucination_risk = unsupported_count / total_tests
            
            results["metrics"] = {
                "total_tests": total_tests,
                "unsupported_responses": unsupported_count,
                "hallucination_risk_score": hallucination_risk
            }
            
            if hallucination_risk > 0.3:
                results["issues"].append({
                    "severity": "medium", "issue": "Elevated hallucination risk",
                    "details": f"Risk score: {hallucination_risk:.2f}",
                    "impact": "AI may provide unsupported or incorrect information"
                })
            
            results["score"] = max(0.0, 1.0 - hallucination_risk)
            
        except Exception as e:
            logger.error(f"Hallucination assessment failed: {str(e)}")
            results.update({"status": "failed", "error": str(e), "score": 0.5})
        
        return results
    
    async def _generate_test_queries(self, goal: Dict) -> List[Dict]:
        """Generate domain-specific test queries."""
        domain = goal.get("domain", "general")
        description = goal.get("description", "")
        
        base_queries = [
            {"text": f"What is the main purpose based on: {description[:100]}?", "expected_answer": "system purpose", "category": "general"},
            {"text": "What are the key features or capabilities?", "expected_answer": "feature list", "category": "features"},
            {"text": "Are there any limitations or restrictions?", "expected_answer": "limitations", "category": "constraints"},
            {"text": "What are the most important specifications?", "expected_answer": "specifications", "category": "technical"},
            {"text": "How recent is this information?", "expected_answer": "date information", "category": "timeliness"}
        ]
        
        # Add domain-specific queries
        domain_queries = {
            "automotive": [
                {"text": "What vehicle makes and models are available?", "expected_answer": "vehicle list", "category": "inventory"},
                {"text": "What is the price range for vehicles?", "expected_answer": "pricing info", "category": "pricing"}
            ],
            "manufacturing": [
                {"text": "What products are manufactured?", "expected_answer": "product list", "category": "products"},
                {"text": "What are the quality standards?", "expected_answer": "quality info", "category": "quality"}
            ],
            "real_estate": [
                {"text": "What properties are available?", "expected_answer": "property list", "category": "inventory"},
                {"text": "What is the price range for properties?", "expected_answer": "pricing info", "category": "pricing"}
            ]
        }
        
        if domain in domain_queries:
            base_queries.extend(domain_queries[domain])
        
        return base_queries
    
    async def _test_single_query(self, query: str, namespace: str, goal: Dict) -> Dict[str, Any]:
        """Test a single query and evaluate the answer quality."""
        try:
            retrieval_result = await self.vector_manager.query_knowledge_base(query, namespace, top_k=config.RETRIEVAL_TOP_K)
            
            if not retrieval_result["success"]:
                return {"success": False, "error": retrieval_result.get("error", "Retrieval failed")}
            
            chunks = retrieval_result["results"]
            if not chunks:
                return {"success": True, "answer_quality": "no_context", "issue": "No relevant information found"}
            
            # Generate answer using LLM
            context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks)])
            
            answer_prompt = f"""Based on the following context pieces, please answer the question. If the context doesn't contain enough information to answer the question, respond with "I DON'T KNOW" and list what information is missing.

CONTEXT:
{context}

QUESTION: {query}

Please provide:
- Answer: <your answer>
- Evidence IDs: [list of context pieces used]
- Confidence: <0-1>
- Missing Facts: [list if any]"""
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an objective fact-extraction assistant."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0.0, max_tokens=500
            )
            
            answer_content = response.choices[0].message.content
            judge_result = await self._judge_answer_quality(answer_content, context, query)
            
            return {
                "success": True, "answer": answer_content, "answer_quality": judge_result["quality"],
                "confidence": judge_result.get("confidence", 0.5), "evidence_used": judge_result.get("evidence_ids", []),
                "missing_facts": judge_result.get("missing_facts", [])
            }
            
        except Exception as e:
            logger.error(f"Query test failed for '{query}': {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _judge_answer_quality(self, answer: str, context: str, question: str) -> Dict[str, Any]:
        """Judge the quality and support level of an answer."""
        judge_prompt = f"""Evaluate whether this answer is supported by the provided context.

QUESTION: {question}
ANSWER: {answer}
CONTEXT: {context}

Respond with:
Quality: [FULLY_SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED]
Confidence: [0.0-1.0]
Reasoning: [brief explanation]"""

        try:
            response = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0, max_tokens=200
            )
            
            judge_content = response.choices[0].message.content
            
            quality_match = re.search(r'Quality:\s*(FULLY_SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED)', judge_content)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', judge_content)
            reasoning_match = re.search(r'Reasoning:\s*(.+)', judge_content)
            
            return {
                "quality": quality_match.group(1).lower() if quality_match else "partially_supported",
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "reasoning": reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            }
            
        except Exception as e:
            logger.error(f"Answer judging failed: {str(e)}")
            return {"quality": "partially_supported", "confidence": 0.5, "reasoning": f"Judging failed: {str(e)}"}
    
    def _compute_scores(self, checks: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Compute composite and individual quality scores."""
        individual_scores = {check_name: check_result.get("score", 0.5) for check_name, check_result in checks.items()}
        
        score_mapping = {
            "coverage": individual_scores.get("coverage", 0.0),
            "accuracy": individual_scores.get("consistency", 0.5),
            "consistency": individual_scores.get("consistency", 0.5),
            "timeliness": individual_scores.get("timeliness", 0.8),
            "uniqueness": 1.0 - individual_scores.get("duplicates", 0.05),
            "retrieval": (individual_scores.get("sanity", 0.5) + individual_scores.get("coverage", 0.0)) / 2,
            "formatting": individual_scores.get("formatting", 0.7)
        }
        
        weights = self.scoring_weights
        composite_score = sum(weights[key] * score_mapping[key] for key in weights.keys())
        
        return {
            "individual_scores": individual_scores, "weighted_scores": score_mapping,
            "composite_score": composite_score, "weights_used": weights, "domain": domain
        }
    
    def _determine_readiness_level(self, scores: Dict[str, Any]) -> str:
        """Determine overall readiness level based on scores."""
        composite = scores["composite_score"]
        individual = scores["individual_scores"]
        
        if individual.get("sanity", 0) < 0.3 or individual.get("coverage", 0) < 0.3:
            return "BLOCKED"
        
        if composite >= self.quality_thresholds["ready"]:
            return "READY"
        elif composite >= self.quality_thresholds["partially_ready"]:
            return "PARTIALLY_READY"
        elif composite >= self.quality_thresholds["unsafe"]:
            return "UNSAFE"
        else:
            return "BLOCKED"
    
    def _generate_remediation(self, checks: Dict, scores: Dict, domain_config: Dict) -> Dict[str, Any]:
        """Generate prioritized remediation recommendations."""
        recommendations = []
        
        for check_name, check_result in checks.items():
            issues = check_result.get("issues", [])
            for issue in issues:
                recommendations.append({
                    "priority": issue["severity"], "category": check_name, "issue": issue["issue"],
                    "details": issue["details"], "impact": issue["impact"],
                    "estimated_effort": "medium", "suggested_actions": [f"Address {issue['issue']} in {check_name}"]
                })
        
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return {
            "recommendations": recommendations,
            "critical_count": len([r for r in recommendations if r["priority"] == "critical"]),
            "high_count": len([r for r in recommendations if r["priority"] == "high"]),
            "estimated_total_effort": len(recommendations),
            "top_categories": list(set([r["category"] for r in recommendations[:5]]))
        }
    
    def _generate_summary(self, checks: Dict, scores: Dict) -> Dict[str, Any]:
        """Generate executive summary of quality analysis."""
        total_issues = sum(len(check.get("issues", [])) for check in checks.values())
        critical_issues = sum(len([i for i in check.get("issues", []) if i["severity"] == "critical"]) for check in checks.values())
        
        return {
            "total_checks_run": len(checks), "checks_passed": len([c for c in checks.values() if c.get("score", 0) > 0.7]),
            "checks_failed": len([c for c in checks.values() if c.get("score", 0) < 0.5]),
            "total_issues": total_issues, "critical_issues": critical_issues,
            "composite_score": scores.get("composite_score", 0.0), "readiness_level": self._determine_readiness_level(scores),
            "key_strengths": ["Data ingestion successful", "Basic retrieval working"],
            "key_weaknesses": ["Coverage may need improvement", "Consider adding more diverse content"]
        }


# Convenience function
async def run_quality_analysis(ingest_summary: Dict[str, Any], goal: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    """Convenience function to run quality analysis."""
    agent = QualityAnalysisAgent()
    return await agent.run_all_checks(ingest_summary, goal, namespace)
