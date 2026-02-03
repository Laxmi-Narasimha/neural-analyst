"""Validation Results Agent - compiles and generates comprehensive reports."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from ..config import config

logger = logging.getLogger(__name__)


class ValidationResultsAgent:
    """
    Validation Results & Report Agent (VR) - Final report generation agent.
    
    Responsibilities:
    - Aggregate quality analysis outputs into reports
    - Generate human-readable Markdown reports
    - Create machine-readable JSON outputs
    - Provide prioritized remediation recommendations
    - Determine final readiness levels with evidence
    """
    
    def __init__(self):
        self.quality_thresholds = config.QUALITY_THRESHOLDS
        self.domain_configs = config.DOMAIN_CONFIGS
    
    def compile_report(self, quality_report: Dict[str, Any], goal: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Compile comprehensive validation report from quality analysis.
        
        Args:
            quality_report: Output from Quality Analysis Agent
            goal: User goal context
            session_id: Optional session identifier
            
        Returns:
            Complete validation report with Markdown and JSON formats
        """
        logger.info("Compiling validation report...")
        
        try:
            # Extract key information
            domain = quality_report.get("domain", "general")
            namespace = quality_report.get("namespace", "unknown")
            scores = quality_report.get("scores", {})
            checks = quality_report.get("checks", {})
            remediation = quality_report.get("remediation", {})
            summary = quality_report.get("summary", {})
            readiness_level = quality_report.get("readiness_level", "UNKNOWN")
            
            # Generate report sections
            executive_summary = self._generate_executive_summary(summary, scores, readiness_level, goal)
            scoring_dashboard = self._generate_scoring_dashboard(scores, checks)
            detailed_issues = self._generate_detailed_issues(remediation, checks)
            remediation_plan = self._generate_remediation_plan(remediation, domain)
            evidence_section = self._generate_evidence_section(checks, quality_report)
            
            # Create Markdown report
            markdown_report = self._generate_markdown_report(
                executive_summary, scoring_dashboard, detailed_issues, 
                remediation_plan, evidence_section, goal, namespace, readiness_level
            )
            
            # Create JSON report
            json_report = self._generate_json_report(
                quality_report, executive_summary, detailed_issues, 
                remediation_plan, goal, session_id
            )
            
            # Save reports if session_id provided
            report_files = {}
            if session_id:
                report_files = self._save_reports(markdown_report, json_report, session_id)
            
            return {
                "success": True,
                "readiness_level": readiness_level,
                "composite_score": scores.get("composite_score", 0.0),
                "markdown_report": markdown_report,
                "json_report": json_report,
                "report_files": report_files,
                "summary": executive_summary,
                "recommendations": remediation.get("recommendations", [])[:5],  # Top 5
                "next_steps": self._generate_next_steps(readiness_level, remediation)
            }
            
        except Exception as e:
            logger.error(f"Report compilation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "readiness_level": "UNKNOWN",
                "markdown_report": f"# Error\n\nReport generation failed: {str(e)}",
                "json_report": {"error": str(e)}
            }
    
    def _generate_executive_summary(self, summary: Dict, scores: Dict, readiness_level: str, goal: Dict) -> str:
        """Generate executive summary paragraph."""
        composite_score = scores.get("composite_score", 0.0)
        total_issues = summary.get("total_issues", 0)
        critical_issues = summary.get("critical_issues", 0)
        
        goal_desc = goal.get("description", "the specified AI assistant")
        domain = goal.get("domain", "general")
        
        if readiness_level == "READY":
            return f"✅ **READY FOR DEPLOYMENT** - The provided data is sufficient for {goal_desc} with a quality score of {composite_score:.1%}. {total_issues} minor issues identified with recommended optimizations available."
        elif readiness_level == "PARTIALLY_READY":
            return f"⚠️ **PARTIALLY READY** - The data shows promise for {goal_desc} (score: {composite_score:.1%}) but requires attention to {total_issues} issues, including {critical_issues} critical items that should be addressed before deployment."
        elif readiness_level == "UNSAFE":
            return f"❌ **UNSAFE FOR DEPLOYMENT** - Significant data quality issues detected (score: {composite_score:.1%}). {critical_issues} critical problems must be resolved before {goal_desc} can function reliably."
        else:  # BLOCKED
            return f"🚫 **BLOCKED** - Critical infrastructure or data issues prevent deployment (score: {composite_score:.1%}). Fundamental problems with data ingestion or quality require immediate attention."
    
    def _generate_scoring_dashboard(self, scores: Dict, checks: Dict) -> str:
        """Generate scoring dashboard table."""
        dashboard = "| Metric | Score | Threshold | Status |\n|--------|-------|-----------|--------|\n"
        
        individual_scores = scores.get("individual_scores", {})
        
        metrics_info = {
            "coverage": ("Coverage", 0.85, "Query answering capability"),
            "consistency": ("Consistency", 0.70, "Data conflict detection"),
            "timeliness": ("Timeliness", 0.70, "Information freshness"),
            "duplicates": ("Uniqueness", 0.95, "Duplicate content ratio"),
            "formatting": ("Formatting", 0.70, "Structure and parsing quality"),
            "sanity": ("Basic Quality", 0.80, "Fundamental data health")
        }
        
        for check_key, (name, threshold, description) in metrics_info.items():
            score = individual_scores.get(check_key, 0.0)
            if check_key == "duplicates":
                score = 1.0 - score  # Invert for display
            
            status = "✅ Pass" if score >= threshold else "❌ Fail" if score < threshold * 0.7 else "⚠️ Warning"
            dashboard += f"| {name} | {score:.1%} | {threshold:.1%} | {status} |\n"
        
        # Add composite score
        composite = scores.get("composite_score", 0.0)
        composite_threshold = self.quality_thresholds["ready"]
        composite_status = "✅ Pass" if composite >= composite_threshold else "❌ Fail"
        dashboard += f"| **Overall** | **{composite:.1%}** | **{composite_threshold:.1%}** | **{composite_status}** |\n"
        
        return dashboard
    
    def _generate_detailed_issues(self, remediation: Dict, checks: Dict) -> List[Dict]:
        """Generate detailed issues list with evidence."""
        detailed_issues = []
        
        recommendations = remediation.get("recommendations", [])
        
        for i, rec in enumerate(recommendations[:10]):  # Top 10 issues
            issue = {
                "id": f"issue_{i+1}",
                "priority": rec.get("priority", "medium"),
                "title": rec.get("issue", "Unknown issue"),
                "category": rec.get("category", "general"),
                "description": rec.get("details", "No details available"),
                "impact": rec.get("impact", "Unknown impact"),
                "evidence": self._get_evidence_for_issue(rec, checks),
                "remediation_steps": rec.get("suggested_actions", ["Review and address manually"]),
                "estimated_effort": rec.get("estimated_effort", "medium")
            }
            detailed_issues.append(issue)
        
        return detailed_issues
    
    def _generate_remediation_plan(self, remediation: Dict, domain: str) -> Dict[str, Any]:
        """Generate structured remediation plan."""
        recommendations = remediation.get("recommendations", [])
        
        # Group by priority
        critical_items = [r for r in recommendations if r.get("priority") == "critical"]
        high_items = [r for r in recommendations if r.get("priority") == "high"]
        medium_items = [r for r in recommendations if r.get("priority") == "medium"]
        
        # Generate domain-specific suggestions
        domain_suggestions = self._get_domain_specific_suggestions(domain, recommendations)
        
        return {
            "immediate_actions": [{"action": item.get("issue", ""), "reason": item.get("impact", "")} for item in critical_items],
            "high_priority": [{"action": item.get("issue", ""), "reason": item.get("impact", "")} for item in high_items],
            "medium_priority": [{"action": item.get("issue", ""), "reason": item.get("impact", "")} for item in medium_items],
            "domain_specific_suggestions": domain_suggestions,
            "estimated_timeline": self._estimate_remediation_timeline(recommendations),
            "success_criteria": self._define_success_criteria(domain, recommendations)
        }
    
    def _generate_evidence_section(self, checks: Dict, quality_report: Dict) -> Dict[str, Any]:
        """Generate evidence section with supporting data."""
        evidence = {
            "data_statistics": {
                "namespace": quality_report.get("namespace", "unknown"),
                "total_checks_performed": len(checks),
                "checks_passed": len([c for c in checks.values() if c.get("score", 0) > 0.7]),
                "checks_failed": len([c for c in checks.values() if c.get("score", 0) < 0.5])
            },
            "key_findings": [],
            "sample_queries": [],
            "technical_details": {}
        }
        
        # Extract key findings from each check
        for check_name, check_result in checks.items():
            if check_result.get("issues"):
                evidence["key_findings"].extend([
                    f"{check_name.title()}: {issue.get('issue', 'Unknown issue')}" 
                    for issue in check_result["issues"][:2]  # Top 2 per check
                ])
            
            # Add sample queries if available
            if check_name == "coverage" and "test_queries" in check_result:
                evidence["sample_queries"] = [
                    q.get("text", "") for q in check_result["test_queries"][:3]
                ]
            
            # Add technical metrics
            if "metrics" in check_result:
                evidence["technical_details"][check_name] = check_result["metrics"]
        
        return evidence
    
    def _generate_markdown_report(self, executive_summary: str, scoring_dashboard: str, 
                                detailed_issues: List[Dict], remediation_plan: Dict, 
                                evidence_section: Dict, goal: Dict, namespace: str, 
                                readiness_level: str) -> str:
        """Generate comprehensive Markdown report."""
        
        report_sections = [
            f"# AI Data Adequacy Assessment Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Domain:** {goal.get('domain', 'general').title()}  ",
            f"**Namespace:** {namespace}  ",
            f"**Goal:** {goal.get('description', 'No description provided')}  ",
            f"",
            f"## Executive Summary",
            f"",
            executive_summary,
            f"",
            f"## Quality Score Dashboard",
            f"",
            scoring_dashboard,
            f"",
            f"## Critical Issues & Remediation",
            f""
        ]
        
        # Add detailed issues
        critical_issues = [issue for issue in detailed_issues if issue["priority"] == "critical"]
        high_issues = [issue for issue in detailed_issues if issue["priority"] == "high"]
        
        if critical_issues:
            report_sections.extend([
                f"### 🚨 Critical Issues (Must Fix)",
                f""
            ])
            for issue in critical_issues:
                report_sections.extend([
                    f"#### {issue['title']}",
                    f"**Impact:** {issue['impact']}  ",
                    f"**Details:** {issue['description']}  ",
                    f"**Actions:**",
                    *[f"- {action}" for action in issue['remediation_steps']],
                    f""
                ])
        
        if high_issues:
            report_sections.extend([
                f"### ⚠️ High Priority Issues",
                f""
            ])
            for issue in high_issues[:3]:  # Top 3 high priority
                report_sections.extend([
                    f"#### {issue['title']}",
                    f"**Impact:** {issue['impact']}  ",
                    f"**Actions:** {', '.join(issue['remediation_steps'][:2])}  ",
                    f""
                ])
        
        # Add remediation plan
        report_sections.extend([
            f"## Recommended Action Plan",
            f"",
            f"### Immediate Actions Required",
        ])
        
        for action in remediation_plan.get("immediate_actions", [])[:3]:
            report_sections.append(f"1. **{action['action']}** - {action['reason']}")
        
        # Add success criteria
        report_sections.extend([
            f"",
            f"### Success Criteria",
            f""
        ])
        
        for criteria in remediation_plan.get("success_criteria", []):
            report_sections.append(f"- {criteria}")
        
        # Add technical evidence
        report_sections.extend([
            f"",
            f"## Technical Evidence",
            f"",
            f"**Data Statistics:**",
            f"- Namespace: {evidence_section['data_statistics']['namespace']}",
            f"- Total Checks: {evidence_section['data_statistics']['total_checks_performed']}",
            f"- Checks Passed: {evidence_section['data_statistics']['checks_passed']}",
            f"- Checks Failed: {evidence_section['data_statistics']['checks_failed']}",
            f""
        ])
        
        if evidence_section.get("sample_queries"):
            report_sections.extend([
                f"**Sample Test Queries:**",
                *[f"- {query}" for query in evidence_section["sample_queries"]],
                f""
            ])
        
        # Footer
        report_sections.extend([
            f"---",
            f"",
            f"*Report generated by AI Data Adequacy Agent v1.0*  ",
            f"*For questions or support, review the remediation plan above.*"
        ])
        
        return "\n".join(report_sections)
    
    def _generate_json_report(self, quality_report: Dict, executive_summary: str, 
                            detailed_issues: List[Dict], remediation_plan: Dict, 
                            goal: Dict, session_id: str = None) -> Dict[str, Any]:
        """Generate machine-readable JSON report."""
        
        return {
            "report_metadata": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "session_id": session_id,
                "domain": goal.get("domain", "general"),
                "namespace": quality_report.get("namespace", "unknown")
            },
            "assessment_results": {
                "readiness_level": quality_report.get("readiness_level", "UNKNOWN"),
                "composite_score": quality_report.get("scores", {}).get("composite_score", 0.0),
                "individual_scores": quality_report.get("scores", {}).get("individual_scores", {}),
                "total_issues": len(detailed_issues),
                "critical_issues": len([i for i in detailed_issues if i["priority"] == "critical"])
            },
            "executive_summary": executive_summary,
            "detailed_issues": detailed_issues,
            "remediation_plan": remediation_plan,
            "evidence": quality_report.get("checks", {}),
            "recommendations": quality_report.get("remediation", {}).get("recommendations", []),
            "next_steps": self._generate_next_steps(quality_report.get("readiness_level", "UNKNOWN"), quality_report.get("remediation", {}))
        }
    
    def _save_reports(self, markdown_report: str, json_report: Dict, session_id: str) -> Dict[str, str]:
        """Save reports to files."""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_filename = f"report_{session_id}_{timestamp}.md"
            json_filename = f"report_{session_id}_{timestamp}.json"
            
            md_path = os.path.join(reports_dir, md_filename)
            json_path = os.path.join(reports_dir, json_filename)
            
            # Save files
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            return {
                "markdown_file": md_path,
                "json_file": json_path,
                "report_id": f"{session_id}_{timestamp}"
            }
            
        except Exception as e:
            logger.error(f"Failed to save reports: {str(e)}")
            return {"error": str(e)}
    
    def _get_evidence_for_issue(self, recommendation: Dict, checks: Dict) -> List[str]:
        """Get supporting evidence for an issue."""
        category = recommendation.get("category", "")
        evidence = []
        
        if category in checks:
            check_result = checks[category]
            metrics = check_result.get("metrics", {})
            
            for key, value in list(metrics.items())[:2]:  # Top 2 metrics
                if isinstance(value, (int, float)):
                    evidence.append(f"{key}: {value}")
                else:
                    evidence.append(f"{key}: {str(value)[:50]}")
        
        return evidence or ["Manual review recommended"]
    
    def _get_domain_specific_suggestions(self, domain: str, recommendations: List[Dict]) -> List[str]:
        """Generate domain-specific improvement suggestions."""
        suggestions = {
            "automotive": [
                "Ensure VIN numbers and vehicle specifications are complete",
                "Verify pricing information is current and accurate",
                "Include safety ratings and recall information"
            ],
            "manufacturing": [
                "Validate product specifications and technical documentation",
                "Ensure compliance and safety standards are documented",
                "Include quality control processes and standards"
            ],
            "real_estate": [
                "Verify property details and square footage accuracy",
                "Ensure pricing reflects current market conditions",
                "Include legal and zoning information"
            ],
            "general": [
                "Review data completeness across all topic areas",
                "Ensure information is current and relevant",
                "Validate consistency across all data sources"
            ]
        }
        
        return suggestions.get(domain, suggestions["general"])
    
    def _estimate_remediation_timeline(self, recommendations: List[Dict]) -> str:
        """Estimate timeline for remediation."""
        critical_count = len([r for r in recommendations if r.get("priority") == "critical"])
        high_count = len([r for r in recommendations if r.get("priority") == "high"])
        
        if critical_count > 3:
            return "2-4 weeks (multiple critical issues require significant work)"
        elif critical_count > 0:
            return "1-2 weeks (critical issues need immediate attention)"
        elif high_count > 5:
            return "1-2 weeks (many high priority improvements needed)"
        elif high_count > 0:
            return "3-7 days (some important improvements recommended)"
        else:
            return "1-3 days (minor optimizations only)"
    
    def _define_success_criteria(self, domain: str, recommendations: List[Dict]) -> List[str]:
        """Define success criteria for remediation."""
        criteria = [
            "All critical issues resolved",
            "Coverage rate above 85%",
            "No conflicting information detected",
            "Data freshness meets domain requirements"
        ]
        
        domain_criteria = {
            "automotive": ["Vehicle inventory completeness verified", "Pricing accuracy confirmed"],
            "manufacturing": ["Product specifications validated", "Compliance documentation complete"],
            "real_estate": ["Property details accuracy verified", "Market pricing current"],
            "general": ["Content coverage comprehensive", "Information accuracy validated"]
        }
        
        if domain in domain_criteria:
            criteria.extend(domain_criteria[domain])
        
        return criteria
    
    def _generate_next_steps(self, readiness_level: str, remediation: Dict) -> List[str]:
        """Generate specific next steps based on readiness level."""
        critical_count = remediation.get("critical_count", 0)
        high_count = remediation.get("high_count", 0)
        
        if readiness_level == "READY":
            return [
                "✅ System is ready for deployment",
                "Consider implementing the optional improvements listed",
                "Set up monitoring for ongoing data quality",
                "Plan regular data updates and reviews"
            ]
        elif readiness_level == "PARTIALLY_READY":
            return [
                f"Address {critical_count} critical issues immediately" if critical_count > 0 else "Review high-priority recommendations",
                f"Plan remediation for {high_count} high-priority items",
                "Test system functionality after critical fixes",
                "Re-run assessment after remediation"
            ]
        elif readiness_level in ["UNSAFE", "BLOCKED"]:
            return [
                "🚫 Do not deploy until critical issues are resolved",
                f"Focus on {critical_count} critical problems first",
                "Consider adding more comprehensive source material",
                "Review data ingestion and parsing processes",
                "Re-run complete assessment after major fixes"
            ]
        else:
            return [
                "Review assessment results and error logs",
                "Ensure API connections are working properly",
                "Consider manual data review and validation"
            ]


# Convenience function
def generate_validation_report(quality_report: Dict[str, Any], goal: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
    """Generate validation report from quality analysis results."""
    agent = ValidationResultsAgent()
    return agent.compile_report(quality_report, goal, session_id)
