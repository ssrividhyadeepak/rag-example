"""
ATM Prompt Templates

ATM-specific prompt templates for generating contextual responses
based on retrieved log entries. Templates are designed for different
types of queries and response formats.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class PromptTemplates:
    """
    Collection of prompt templates for ATM RAG responses.
    """

    # Base template for contextual responses
    BASE_CONTEXT_TEMPLATE = """Based on similar ATM incidents and operational data:

CONTEXT:
{context}

USER QUESTION: {user_question}

Please provide a helpful response that:
1. Directly addresses the user's question
2. References specific examples from the context
3. Provides actionable information when possible
4. Uses clear, professional language
5. Focuses on ATM operational context

RESPONSE:"""

    # Template for troubleshooting queries
    TROUBLESHOOTING_TEMPLATE = """ATM TROUBLESHOOTING ASSISTANCE

ISSUE: {user_question}

SIMILAR CASES FOUND:
{context}

ANALYSIS:
Based on the similar cases above, this appears to be related to {primary_issue_type}.

COMMON CAUSES:
{common_causes}

RECOMMENDED ACTIONS:
{recommended_actions}

ADDITIONAL INFORMATION:
{additional_info}

If this doesn't resolve the issue, please contact technical support with the error details."""

    # Template for error code explanations
    ERROR_CODE_TEMPLATE = """ATM ERROR CODE EXPLANATION

ERROR CODE: {error_code}
QUERY: {user_question}

SIMILAR INCIDENTS:
{context}

EXPLANATION:
{error_explanation}

TYPICAL SCENARIOS:
{typical_scenarios}

RESOLUTION STEPS:
{resolution_steps}

PREVENTION:
{prevention_tips}"""

    # Template for operational analysis
    ANALYSIS_TEMPLATE = """ATM OPERATIONAL ANALYSIS

ANALYSIS REQUEST: {user_question}

RELEVANT DATA:
{context}

FINDINGS:
{key_findings}

PATTERNS IDENTIFIED:
{patterns}

RECOMMENDATIONS:
{recommendations}

DATA TIMEFRAME: {timeframe}
TOTAL INCIDENTS ANALYZED: {total_incidents}"""

    # Template for informational queries
    INFO_TEMPLATE = """ATM INFORMATION

QUESTION: {user_question}

RELEVANT EXAMPLES:
{context}

EXPLANATION:
{explanation}

KEY POINTS:
{key_points}

RELATED INFORMATION:
{related_info}"""

    @classmethod
    def generate_troubleshooting_response(cls,
                                        user_question: str,
                                        similar_logs: List[Dict[str, Any]],
                                        context_summary: str = "",
                                        max_examples: int = 3) -> str:
        """
        Generate troubleshooting response from similar logs.

        Args:
            user_question (str): User's troubleshooting question
            similar_logs (List[Dict[str, Any]]): Similar log entries
            context_summary (str): Summary of the context
            max_examples (int): Maximum examples to include

        Returns:
            str: Formatted troubleshooting response
        """
        if not similar_logs:
            return cls._generate_no_context_response(user_question)

        # Extract context from similar logs
        context_lines = []
        error_codes = set()
        operations = set()
        atms = set()
        common_issues = []

        for i, log in enumerate(similar_logs[:max_examples]):
            similarity = log.get('similarity_score', 0)
            timestamp = log.get('timestamp', 'Unknown time')
            operation = log.get('operation', 'Unknown operation')
            status = log.get('status', 'Unknown status')
            message = log.get('message', '')
            error_code = log.get('error_code', '')
            atm_id = log.get('atm_id', '')

            operations.add(operation)
            if error_code:
                error_codes.add(error_code)
            if atm_id:
                atms.add(atm_id)

            context_lines.append(f"{i+1}. [{similarity:.3f} similarity] {timestamp}")
            context_lines.append(f"   Operation: {operation} - {status}")
            context_lines.append(f"   Message: {message}")
            if error_code:
                context_lines.append(f"   Error Code: {error_code}")
            if atm_id:
                context_lines.append(f"   ATM: {atm_id}")
            context_lines.append("")

            common_issues.append(f"{operation} {status}: {message}")

        context = "\n".join(context_lines)

        # Determine primary issue type
        primary_issue_type = cls._determine_primary_issue(similar_logs)

        # Generate common causes
        common_causes = cls._generate_common_causes(error_codes, operations, similar_logs)

        # Generate recommended actions
        recommended_actions = cls._generate_recommended_actions(error_codes, operations, primary_issue_type)

        # Additional information
        additional_info = cls._generate_additional_info(similar_logs, atms)

        return cls.TROUBLESHOOTING_TEMPLATE.format(
            user_question=user_question,
            context=context,
            primary_issue_type=primary_issue_type,
            common_causes=common_causes,
            recommended_actions=recommended_actions,
            additional_info=additional_info
        )

    @classmethod
    def generate_error_code_response(cls,
                                   user_question: str,
                                   error_code: str,
                                   similar_logs: List[Dict[str, Any]]) -> str:
        """Generate response for error code queries."""
        if not similar_logs:
            return cls._generate_no_context_response(user_question)

        context_lines = []
        for i, log in enumerate(similar_logs[:5]):
            timestamp = log.get('timestamp', 'Unknown time')
            message = log.get('message', '')
            atm_id = log.get('atm_id', 'Unknown ATM')

            context_lines.append(f"• {timestamp} - {atm_id}: {message}")

        context = "\n".join(context_lines)

        # Generate error explanation
        error_explanation = cls._get_error_explanation(error_code)

        # Generate typical scenarios
        typical_scenarios = cls._generate_typical_scenarios(error_code, similar_logs)

        # Generate resolution steps
        resolution_steps = cls._generate_resolution_steps(error_code)

        # Generate prevention tips
        prevention_tips = cls._generate_prevention_tips(error_code)

        return cls.ERROR_CODE_TEMPLATE.format(
            error_code=error_code,
            user_question=user_question,
            context=context,
            error_explanation=error_explanation,
            typical_scenarios=typical_scenarios,
            resolution_steps=resolution_steps,
            prevention_tips=prevention_tips
        )

    @classmethod
    def generate_analysis_response(cls,
                                 user_question: str,
                                 similar_logs: List[Dict[str, Any]],
                                 analysis_type: str = "general") -> str:
        """Generate analytical response."""
        if not similar_logs:
            return cls._generate_no_context_response(user_question)

        # Build context from logs
        context_lines = []
        for log in similar_logs:
            context_lines.append(f"• {log.get('timestamp', 'Unknown')}: "
                               f"{log.get('operation', 'Unknown')} - {log.get('status', 'Unknown')} "
                               f"at {log.get('atm_id', 'Unknown ATM')}")

        context = "\n".join(context_lines[:10])  # Limit to 10 entries

        # Generate analysis components
        key_findings = cls._generate_key_findings(similar_logs)
        patterns = cls._generate_patterns(similar_logs)
        recommendations = cls._generate_recommendations(similar_logs, analysis_type)

        # Calculate timeframe
        timestamps = [log.get('timestamp') for log in similar_logs if log.get('timestamp')]
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            timeframe = f"{earliest} to {latest}"
        else:
            timeframe = "Unknown"

        return cls.ANALYSIS_TEMPLATE.format(
            user_question=user_question,
            context=context,
            key_findings=key_findings,
            patterns=patterns,
            recommendations=recommendations,
            timeframe=timeframe,
            total_incidents=len(similar_logs)
        )

    @classmethod
    def generate_info_response(cls,
                             user_question: str,
                             similar_logs: List[Dict[str, Any]]) -> str:
        """Generate informational response."""
        if not similar_logs:
            return cls._generate_no_context_response(user_question)

        context_lines = []
        for log in similar_logs[:3]:
            context_lines.append(f"• {log.get('message', 'No message')} "
                               f"({log.get('operation', 'Unknown')} at {log.get('atm_id', 'Unknown ATM')})")

        context = "\n".join(context_lines)

        explanation = cls._generate_explanation(user_question, similar_logs)
        key_points = cls._generate_key_points(similar_logs)
        related_info = cls._generate_related_info(similar_logs)

        return cls.INFO_TEMPLATE.format(
            user_question=user_question,
            context=context,
            explanation=explanation,
            key_points=key_points,
            related_info=related_info
        )

    @classmethod
    def _generate_no_context_response(cls, user_question: str) -> str:
        """Generate response when no similar logs are found."""
        return f"""I don't have specific ATM log data that matches your question: "{user_question}"

This could mean:
• The issue hasn't occurred recently in our ATM network
• The query might need to be more specific
• The logs might not contain relevant information for this particular question

SUGGESTIONS:
• Try rephrasing your question with more specific terms
• Include ATM IDs, error codes, or time ranges if known
• Check if this is a new or uncommon issue

For immediate assistance with ATM issues, please contact technical support."""

    @classmethod
    def _determine_primary_issue(cls, logs: List[Dict[str, Any]]) -> str:
        """Determine the primary issue type from logs."""
        operations = [log.get('operation', '') for log in logs]
        statuses = [log.get('status', '') for log in logs]
        error_codes = [log.get('error_code', '') for log in logs if log.get('error_code')]

        # Count occurrences
        from collections import Counter
        op_counts = Counter(operations)
        status_counts = Counter(statuses)

        primary_op = op_counts.most_common(1)[0][0] if op_counts else "unknown"
        primary_status = status_counts.most_common(1)[0][0] if status_counts else "unknown"

        if error_codes:
            error_counts = Counter(error_codes)
            primary_error = error_counts.most_common(1)[0][0]
            return f"{primary_op} {primary_status} ({primary_error})"
        else:
            return f"{primary_op} {primary_status}"

    @classmethod
    def _generate_common_causes(cls, error_codes: set, operations: set, logs: List[Dict[str, Any]]) -> str:
        """Generate common causes based on error patterns."""
        causes = []

        # Error code specific causes
        for error_code in error_codes:
            if error_code == "DDL_EXCEEDED":
                causes.append("• Daily Dollar Limit exceeded - customer attempted to withdraw more than their daily limit")
            elif error_code == "INSUFFICIENT_FUNDS":
                causes.append("• Account balance insufficient for the requested transaction amount")
            elif error_code == "PIN_RETRY_EXCEEDED":
                causes.append("• Customer exceeded the maximum number of PIN attempts")
            elif error_code == "DISPENSER_ERROR":
                causes.append("• Hardware malfunction in the cash dispensing mechanism")
            elif error_code == "ENVELOPE_JAM":
                causes.append("• Physical obstruction in the deposit envelope mechanism")
            else:
                causes.append(f"• {error_code}: System-specific error condition")

        # Operation specific causes
        for operation in operations:
            if operation == "withdrawal" and not causes:
                causes.append("• ATM cash supply or dispensing mechanism issues")
            elif operation == "deposit" and not causes:
                causes.append("• Deposit processing or envelope handling problems")

        return "\n".join(causes) if causes else "• Analysis of similar cases suggests multiple potential causes"

    @classmethod
    def _generate_recommended_actions(cls, error_codes: set, operations: set, primary_issue: str) -> str:
        """Generate recommended actions."""
        actions = []

        # Error code specific actions
        if "DDL_EXCEEDED" in error_codes:
            actions.extend([
                "• Advise customer to wait until next business day for limit reset",
                "• Suggest checking account settings for daily limit adjustments",
                "• Recommend contacting bank for limit increase if needed"
            ])

        if "INSUFFICIENT_FUNDS" in error_codes:
            actions.extend([
                "• Confirm account balance with customer",
                "• Suggest smaller withdrawal amount",
                "• Check for pending transactions that may affect balance"
            ])

        if "DISPENSER_ERROR" in error_codes:
            actions.extend([
                "• Check ATM cash supply and dispensing mechanism",
                "• Clear any paper jams or obstructions",
                "• Consider taking ATM out of service for maintenance"
            ])

        if "PIN_RETRY_EXCEEDED" in error_codes:
            actions.extend([
                "• Card should be retained by the ATM as security measure",
                "• Customer needs to contact bank for card replacement",
                "• Verify customer identity before any assistance"
            ])

        # Generic actions if no specific ones
        if not actions:
            actions.extend([
                "• Review similar cases for common resolution patterns",
                "• Check ATM hardware status and error logs",
                "• Consider temporary service adjustment if pattern persists"
            ])

        return "\n".join(actions)

    @classmethod
    def _generate_additional_info(cls, logs: List[Dict[str, Any]], atms: set) -> str:
        """Generate additional contextual information."""
        info_parts = []

        if len(atms) > 1:
            info_parts.append(f"This issue has been observed across {len(atms)} different ATMs: {', '.join(list(atms)[:3])}")

        # Time pattern analysis
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        if len(timestamps) > 1:
            info_parts.append("Consider checking for time-based patterns or peak usage periods.")

        # Severity assessment
        error_count = sum(1 for log in logs if log.get('metadata', {}).get('is_error', False))
        if error_count > len(logs) * 0.8:
            info_parts.append("High error rate detected - consider priority investigation.")

        return "\n".join(info_parts) if info_parts else "Monitor for recurring patterns and escalate if frequency increases."

    @classmethod
    def _get_error_explanation(cls, error_code: str) -> str:
        """Get explanation for specific error codes."""
        explanations = {
            "DDL_EXCEEDED": "Daily Dollar Limit Exceeded - The customer has attempted to withdraw more money than their daily withdrawal limit allows.",
            "INSUFFICIENT_FUNDS": "The account balance is insufficient to complete the requested transaction amount.",
            "PIN_RETRY_EXCEEDED": "The customer has exceeded the maximum number of allowed PIN entry attempts, typically 3 attempts.",
            "DISPENSER_ERROR": "Hardware malfunction in the cash dispensing mechanism preventing proper cash delivery.",
            "ENVELOPE_JAM": "Physical obstruction or mechanical issue in the deposit envelope processing system.",
            "CARD_READ_ERROR": "The ATM is unable to properly read the magnetic stripe or chip on the customer's card.",
            "MAINTENANCE_MODE": "The ATM is currently in maintenance mode and temporarily unavailable for transactions.",
            "USER_TIMEOUT": "The transaction was cancelled due to customer inactivity during the transaction process."
        }

        return explanations.get(error_code, f"System error code {error_code} indicates a specific operational condition that requires investigation.")

    @classmethod
    def _generate_typical_scenarios(cls, error_code: str, logs: List[Dict[str, Any]]) -> str:
        """Generate typical scenarios for error codes."""
        scenarios = []

        if logs:
            # Extract patterns from actual logs
            amounts = [log.get('amount') for log in logs if log.get('amount')]
            atms = [log.get('atm_id') for log in logs if log.get('atm_id')]

            if amounts:
                avg_amount = sum(amounts) / len(amounts)
                scenarios.append(f"• Typical transaction amount: ${avg_amount:.2f}")

            if atms:
                from collections import Counter
                atm_counts = Counter(atms)
                most_affected = atm_counts.most_common(1)[0]
                scenarios.append(f"• Most affected ATM: {most_affected[0]} ({most_affected[1]} incidents)")

        # Add generic scenarios based on error code
        if error_code == "DDL_EXCEEDED":
            scenarios.append("• Often occurs during afternoon/evening hours when customers make multiple withdrawals")
        elif error_code == "DISPENSER_ERROR":
            scenarios.append("• May be preceded by unusual sounds or delays in cash dispensing")

        return "\n".join(scenarios) if scenarios else "• Scenarios vary based on specific operational conditions"

    @classmethod
    def _generate_resolution_steps(cls, error_code: str) -> str:
        """Generate resolution steps for error codes."""
        steps = {
            "DDL_EXCEEDED": [
                "1. Explain daily limit restriction to customer",
                "2. Suggest waiting until next business day",
                "3. Provide information about limit increase procedures"
            ],
            "DISPENSER_ERROR": [
                "1. Check for cash jams or obstructions",
                "2. Restart ATM dispensing mechanism",
                "3. Contact maintenance if issue persists",
                "4. Consider taking ATM offline if unsafe"
            ],
            "PIN_RETRY_EXCEEDED": [
                "1. Confirm card has been retained by ATM",
                "2. Direct customer to bank branch for assistance",
                "3. Do not attempt to retrieve card manually"
            ]
        }

        return "\n".join(steps.get(error_code, [
            "1. Document the error details and frequency",
            "2. Check ATM system logs for additional information",
            "3. Contact technical support if issue persists"
        ]))

    @classmethod
    def _generate_prevention_tips(cls, error_code: str) -> str:
        """Generate prevention tips."""
        tips = {
            "DDL_EXCEEDED": "• Customers should monitor their daily withdrawal amounts and plan accordingly",
            "DISPENSER_ERROR": "• Regular maintenance and cleaning of dispensing mechanisms recommended",
            "PIN_RETRY_EXCEEDED": "• Customer education about secure PIN entry and memory aids"
        }

        return tips.get(error_code, "• Regular monitoring and proactive maintenance help prevent recurring issues")

    @classmethod
    def _generate_key_findings(cls, logs: List[Dict[str, Any]]) -> str:
        """Generate key findings from log analysis."""
        if not logs:
            return "• No significant patterns identified"

        findings = []

        # Operation analysis
        operations = [log.get('operation', '') for log in logs]
        from collections import Counter
        op_counts = Counter(operations)

        if op_counts:
            most_common_op = op_counts.most_common(1)[0]
            findings.append(f"• {most_common_op[0].title()} operations account for {most_common_op[1]} out of {len(logs)} incidents ({most_common_op[1]/len(logs)*100:.1f}%)")

        # Error rate
        error_logs = [log for log in logs if log.get('metadata', {}).get('is_error', False)]
        if error_logs:
            error_rate = len(error_logs) / len(logs) * 100
            findings.append(f"• Error rate: {error_rate:.1f}% ({len(error_logs)} errors out of {len(logs)} total)")

        # ATM distribution
        atms = [log.get('atm_id') for log in logs if log.get('atm_id')]
        if atms:
            unique_atms = len(set(atms))
            findings.append(f"• Issues distributed across {unique_atms} different ATMs")

        return "\n".join(findings[:5]) if findings else "• Analysis shows mixed operational patterns"

    @classmethod
    def _generate_patterns(cls, logs: List[Dict[str, Any]]) -> str:
        """Generate pattern analysis."""
        if not logs:
            return "• No clear patterns identified"

        patterns = []

        # Time patterns
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        if timestamps and hasattr(timestamps[0], 'hour'):  # Check if datetime objects
            hours = [ts.hour for ts in timestamps]
            from collections import Counter
            hour_counts = Counter(hours)
            peak_hour = hour_counts.most_common(1)[0]
            patterns.append(f"• Peak activity hour: {peak_hour[0]}:00 ({peak_hour[1]} incidents)")

        # Error code patterns
        error_codes = [log.get('error_code') for log in logs if log.get('error_code')]
        if error_codes:
            from collections import Counter
            error_counts = Counter(error_codes)
            top_error = error_counts.most_common(1)[0]
            patterns.append(f"• Most frequent error: {top_error[0]} ({top_error[1]} occurrences)")

        return "\n".join(patterns[:3]) if patterns else "• Patterns require more data for meaningful analysis"

    @classmethod
    def _generate_recommendations(cls, logs: List[Dict[str, Any]], analysis_type: str) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Error-focused recommendations
        error_logs = [log for log in logs if log.get('metadata', {}).get('is_error', False)]
        if len(error_logs) > len(logs) * 0.5:  # High error rate
            recommendations.append("• Priority: Investigate root cause of high error rate")
            recommendations.append("• Consider temporary service adjustments to prevent customer impact")

        # ATM-specific recommendations
        atms = [log.get('atm_id') for log in logs if log.get('atm_id')]
        if atms:
            from collections import Counter
            atm_counts = Counter(atms)
            if len(atm_counts) > 0:
                most_affected = atm_counts.most_common(1)[0]
                if most_affected[1] > len(logs) * 0.3:  # One ATM has >30% of issues
                    recommendations.append(f"• Focus maintenance attention on {most_affected[0]} (highest incident count)")

        # Generic recommendations
        if not recommendations:
            recommendations.extend([
                "• Continue monitoring for emerging patterns",
                "• Document resolution outcomes for pattern analysis"
            ])

        return "\n".join(recommendations[:4])

    @classmethod
    def _generate_explanation(cls, question: str, logs: List[Dict[str, Any]]) -> str:
        """Generate explanation for informational queries."""
        if not logs:
            return "Based on the available ATM operational data, I can provide general information about ATM operations."

        # Extract key information from logs
        operations = set(log.get('operation', '') for log in logs)
        statuses = set(log.get('status', '') for log in logs)

        explanation_parts = []
        explanation_parts.append(f"Based on {len(logs)} similar ATM incidents:")

        if operations:
            explanation_parts.append(f"• Operations involved: {', '.join(operations)}")

        if statuses:
            explanation_parts.append(f"• Outcome statuses: {', '.join(statuses)}")

        return "\n".join(explanation_parts)

    @classmethod
    def _generate_key_points(cls, logs: List[Dict[str, Any]]) -> str:
        """Generate key points from logs."""
        if not logs:
            return "• Limited data available for comprehensive analysis"

        points = []

        # Message analysis
        messages = [log.get('message', '') for log in logs if log.get('message')]
        if messages:
            # Find common words or phrases
            all_words = []
            for message in messages:
                all_words.extend(message.lower().split())

            from collections import Counter
            word_counts = Counter(all_words)
            common_words = [word for word, count in word_counts.most_common(5) if len(word) > 3]

            if common_words:
                points.append(f"• Common terms: {', '.join(common_words[:3])}")

        # Operational points
        error_logs = [log for log in logs if log.get('metadata', {}).get('is_error', False)]
        if error_logs:
            points.append(f"• {len(error_logs)} out of {len(logs)} cases involved errors")

        return "\n".join(points[:3]) if points else "• Analysis based on operational data patterns"

    @classmethod
    def _generate_related_info(cls, logs: List[Dict[str, Any]]) -> str:
        """Generate related information."""
        if not logs:
            return "• For more specific information, try asking about particular error codes or ATM operations"

        info_parts = []

        # Time-based info
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        if timestamps:
            info_parts.append("• Historical data available for trend analysis")

        # ATM coverage info
        atms = set(log.get('atm_id') for log in logs if log.get('atm_id'))
        if atms:
            info_parts.append(f"• Data covers {len(atms)} ATM locations")

        # Error code info
        error_codes = set(log.get('error_code') for log in logs if log.get('error_code'))
        if error_codes:
            info_parts.append(f"• Related error codes: {', '.join(list(error_codes)[:3])}")

        return "\n".join(info_parts[:3]) if info_parts else "• Additional context available through specific queries"