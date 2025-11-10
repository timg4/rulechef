"""LLM-based rule learning"""

import json
import re
from typing import Dict, List, Optional
from anthropic import Anthropic

from rulechef.core import Rule, RuleFormat, Span, Dataset, Correction


class RuleLearner:
    """Learns extraction rules from examples using LLM"""

    def __init__(
        self, llm: Anthropic, allowed_formats: Optional[List[RuleFormat]] = None
    ):
        self.llm = llm
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]

    # ========================================
    # Rule Synthesis
    # ========================================

    def synthesize_ruleset(self, dataset: Dataset, max_rules: int = 10) -> List[Rule]:
        """Generate initial ruleset from dataset"""
        import time

        prompt = self._build_synthesis_prompt(dataset, max_rules)

        print("📚 Synthesizing rules from dataset...")

        start = time.time()
        try:
            response = self.llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.content[0].text)

            # Extract rules from response
            rules = []
            for i, rule_data in enumerate(result.get("rules", [])[:max_rules]):
                rule_format = RuleFormat(rule_data.get("format", "regex"))

                # Skip rules not in allowed formats
                if rule_format not in self.allowed_formats:
                    print(
                        f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', f'Rule {i + 1}')}"
                    )
                    continue

                rule = Rule(
                    id=self._generate_id(),
                    name=rule_data.get("name", f"Rule {i + 1}"),
                    description=rule_data.get("description", ""),
                    format=rule_format,
                    content=rule_data.get("content", ""),
                    priority=rule_data.get("priority", 5),
                )
                # Validate rule before adding
                if self._validate_rule(rule):
                    rules.append(rule)
                else:
                    print(f"   ⚠ Skipped invalid rule: {rule.name}")

            elapsed = time.time() - start
            print(f"✓ Synthesized {len(rules)} rules ({elapsed:.1f}s)")
            return rules

        except Exception as e:
            print(f"Error synthesizing rules: {e}")
            return []

    def _build_synthesis_prompt(self, dataset: Dataset, max_rules: int) -> str:
        """Build prompt for rule synthesis"""

        prompt = f"""Task: {dataset.task.name}
Description: {dataset.task.description}

Input schema: {dataset.task.input_schema}
Output schema: {dataset.task.output_schema}

"""

        # Add corrections (highest priority)
        if dataset.corrections:
            prompt += (
                "CORRECTIONS (Learn from failures - these show what went wrong):\n"
            )
            for corr in dataset.corrections[:5]:
                prompt += f"\nInput: {json.dumps(corr.input)}\n"
                prompt += f"Got (WRONG): {json.dumps(corr.model_output)}\n"
                prompt += f"Expected (CORRECT): {json.dumps(corr.expected_output)}\n"
                if corr.feedback:
                    prompt += f"Feedback: {corr.feedback}\n"

        # Add examples
        if dataset.examples:
            prompt += "\nTRAINING EXAMPLES:\n"
            for ex in dataset.examples[:5]:
                prompt += f"\nInput: {json.dumps(ex.input)}\n"
                prompt += f"Output: {json.dumps(ex.expected_output)}\n"

        # Add feedback
        if dataset.feedback:
            prompt += "\n\nUSER FEEDBACK:\n"
            for fb in dataset.feedback:
                prompt += f"- {fb}\n"

        prompt += f"""

YOUR TASK:
Synthesize a complete ruleset (max {max_rules} rules) that:
1. Handles all corrections correctly (CRITICAL - these show failure modes)
2. Works on all examples
3. Respects user feedback
4. Is general and minimal (avoid redundant rules)

RULES CAN BE:"""

        # Show allowed formats
        if RuleFormat.REGEX in self.allowed_formats:
            prompt += "\n- Regex patterns (for structured extraction)"
        if RuleFormat.CODE in self.allowed_formats:
            prompt += "\n- Python code (for complex logic)"

        prompt += f"""

Return JSON:
{{
  "analysis": "What patterns did you find? What went wrong in corrections?",
  "strategy": "Overall approach",
  "rules": [
    {{
      "name": "Short rule name",
      "description": "What this rule does",
      "format": "regex"{' or "code"' if RuleFormat.CODE in self.allowed_formats else ""},
      "content": "regex pattern{" OR python code" if RuleFormat.CODE in self.allowed_formats else ""}",
      "priority": 1-10 (higher = more important),
      "reasoning": "Why this rule is needed"
    }}
  ]
}}
"""

        if RuleFormat.CODE in self.allowed_formats:
            prompt += """
For CODE format, provide a function that takes input dict and returns list of dicts:
```python
def extract(input_data):
    # input_data is dict with keys from input_schema
    # return list of dicts: [{"text": "...", "start": 0, "end": 10}, ...]
    import re
    spans = []
    # your logic here
    return spans
```
"""

        prompt += """
Focus on learning from CORRECTIONS - they show exactly what went wrong!

IMPORTANT: Return ONLY valid JSON. Ensure:
- All strings use double quotes and are properly escaped
- All braces and brackets are balanced
- No trailing commas
- Response is complete (not truncated)
"""

        return prompt

    # ========================================
    # Rule Evaluation & Refinement
    # ========================================

    def evaluate_and_refine(
        self, rules: List[Rule], dataset: Dataset, max_iterations: int = 3
    ) -> tuple:
        """Evaluate rules and refine through agentic loop"""
        import time

        print(f"\n🔄 Refinement loop (max {max_iterations} iterations)")

        best_rules = rules
        best_accuracy = 0.0
        results = None

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"[{iter_num}/{max_iterations}] Evaluating rules...")

            # Evaluate
            results = self._evaluate_rules(rules, dataset)
            accuracy = results["accuracy"]

            print(
                f"[{iter_num}/{max_iterations}] Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})"
            )

            if accuracy > best_accuracy:
                best_rules = rules
                best_accuracy = accuracy

            # Stop if good enough
            if accuracy >= 0.90:
                print("✓ Achieved 90%+ accuracy!")
                break

            # Refine based on failures
            if results["failures"]:
                print(
                    f"[{iter_num}/{max_iterations}] Refining based on {len(results['failures'])} failures..."
                )
                start = time.time()
                rules = self._refine_rules(rules, results["failures"], dataset)
                elapsed = time.time() - start
                if not rules:
                    print("⚠ Refinement failed, keeping best rules")
                    rules = best_rules
                else:
                    print(
                        f"[{iter_num}/{max_iterations}] Refined {len(rules)} rules ({elapsed:.1f}s)"
                    )
            else:
                print("✓ No failures to fix!")
                break

        return best_rules, {
            "accuracy": best_accuracy,
            "total": results["total"],
            "correct": int(best_accuracy * results["total"]),
        }

    def _evaluate_rules(self, rules: List[Rule], dataset: Dataset) -> Dict:
        """Test rules on all training data"""

        all_data = dataset.get_all_training_data()
        total = len(all_data)
        correct = 0
        failures = []

        for item in all_data:
            # Apply rules
            extracted = self._apply_rules(rules, item.input)
            expected = item.expected_output

            # Check correctness
            if self._outputs_match(extracted, expected):
                correct += 1
                if hasattr(item, "update_stats"):
                    # Update rule stats
                    for rule in rules:
                        rule.update_stats(True)
            else:
                failures.append(
                    {
                        "input": item.input,
                        "expected": expected,
                        "got": extracted,
                        "is_correction": isinstance(item, Correction),
                    }
                )
                if hasattr(item, "update_stats"):
                    for rule in rules:
                        rule.update_stats(False)

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "failures": failures,
        }

    def _refine_rules(
        self, current_rules: List[Rule], failures: List[Dict], dataset: Dataset
    ) -> Optional[List[Rule]]:
        """Refine rules based on failures"""

        # Prioritize correction failures
        priority_failures = sorted(
            failures, key=lambda f: f.get("is_correction", False), reverse=True
        )

        prompt = f"""You previously generated these rules:

{self._format_rules(current_rules)}

But they failed on these cases:
{json.dumps(priority_failures[:10], indent=2)}

Refine the ruleset to fix these failures while maintaining performance on other examples.

CRITICAL: Pay special attention to correction failures (is_correction: true) - these are user-verified mistakes.

Allowed rule formats: {", ".join(fmt.value for fmt in self.allowed_formats)}
"""

        if RuleFormat.CODE in self.allowed_formats:
            prompt += """
For CODE rules, return list of dicts: [{"text": "...", "start": 0, "end": 10}, ...]
"""

        prompt += """
Return refined ruleset in same JSON format:
{
  "reasoning": "Why these changes fix the failures",
  "rules": [...]
}
"""

        try:
            response = self.llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.content[0].text)

            # Extract refined rules
            rules = []
            for rule_data in result.get("rules", []):
                rule_format = RuleFormat(rule_data.get("format", "regex"))

                # Skip rules not in allowed formats
                if rule_format not in self.allowed_formats:
                    print(
                        f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', 'Refined rule')}"
                    )
                    continue

                rule = Rule(
                    id=self._generate_id(),
                    name=rule_data.get("name", "Refined rule"),
                    description=rule_data.get("description", ""),
                    format=rule_format,
                    content=rule_data.get("content", ""),
                    priority=rule_data.get("priority", 5),
                )
                # Validate rule before adding
                if self._validate_rule(rule):
                    rules.append(rule)
                else:
                    print(f"   ⚠ Skipped invalid refined rule: {rule.name}")

            return rules if rules else None

        except Exception as e:
            print(f"Error refining rules: {e}")
            return None

    def _apply_rules(self, rules: List[Rule], input_data: Dict) -> Dict:
        """Apply rules to input and return output"""

        all_spans = []

        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            try:
                spans = self._execute_rule(rule, input_data)
                all_spans.extend(spans)
            except Exception:
                # Rules should have been validated during learning, but warn if they fail
                # This might indicate invalid saved rules or edge cases
                pass

        # Deduplicate
        unique_spans = self._deduplicate_spans(all_spans)

        return {"spans": [s.to_dict() for s in unique_spans]}

    def _execute_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute a single rule"""

        if rule.format == RuleFormat.REGEX:
            return self._execute_regex_rule(rule, input_data)
        elif rule.format == RuleFormat.CODE:
            return self._execute_code_rule(rule, input_data)

        return []

    def _execute_regex_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute regex rule"""

        pattern = re.compile(rule.content)
        context = input_data.get("context", "")
        spans = []

        for match in pattern.finditer(context):
            spans.append(
                Span(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=rule.confidence,
                )
            )

        return spans

    def _execute_code_rule(self, rule: Rule, input_data: Dict) -> List[Span]:
        """Execute code rule"""

        try:
            namespace = {"Span": Span, "re": re}
            exec(rule.content, namespace)
            extract_func = namespace.get("extract")

            if extract_func:
                results = extract_func(input_data)
                # Handle case where results is not a list (e.g., returns a string)
                if not isinstance(results, list):
                    return []

                # Convert dicts to Span objects and ensure all have scores
                spans = []
                for result in results:
                    if isinstance(result, dict):
                        # Convert dict to Span object
                        span = Span(
                            text=result.get("text", ""),
                            start=result.get("start", 0),
                            end=result.get("end", 0),
                            score=result.get("score", rule.confidence),
                        )
                        spans.append(span)
                    else:
                        # Already a Span object
                        if not hasattr(result, "score"):
                            result.score = rule.confidence
                        spans.append(result)
                return spans
        except Exception:
            # Rules are validated during learning, so execution errors are rare
            # but silently continue to next rule
            pass

        return []

    def _deduplicate_spans(self, spans: List[Span]) -> List[Span]:
        """Remove overlapping spans"""

        if not spans:
            return []

        # Convert dicts to Span objects if needed
        span_objects = []
        for s in spans:
            if isinstance(s, dict):
                span_objects.append(
                    Span(
                        text=s["text"],
                        start=s["start"],
                        end=s["end"],
                        score=s.get("score", 0.5),
                    )
                )
            else:
                span_objects.append(s)

        sorted_spans = sorted(span_objects, key=lambda s: s.score, reverse=True)
        unique = []

        for span in sorted_spans:
            is_dup = False
            for existing in unique:
                if span.overlap_ratio(existing) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(span)

        return unique[:5]

    def _outputs_match(self, output1: Dict, output2: Dict) -> bool:
        """Check if two outputs match"""

        spans1 = output1.get("spans", [])
        spans2 = output2.get("spans", [])

        if len(spans1) != len(spans2):
            return False

        # Check text match (order independent)
        texts1 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans1])
        texts2 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans2])

        return texts1 == texts2

    def _format_output(self, output: Dict) -> str:
        """Format output for prompt"""
        spans = output.get("spans", [])
        return json.dumps(
            [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in spans]
        )

    def _format_rules(self, rules: List[Rule]) -> str:
        """Format rules for display"""
        formatted = []
        for i, rule in enumerate(rules, 1):
            formatted.append(f"{i}. {rule.name}")
            formatted.append(f"   Format: {rule.format.value}")
            formatted.append(f"   Priority: {rule.priority}")
            formatted.append(f"   Content: {rule.content[:100]}...")
        return "\n".join(formatted)

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response with error handling"""

        # Extract JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Log error with context
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n⚠️ JSON parsing error: {e}")
            print(f"Failed to parse: {preview}")
            raise

    def _generate_input(self, task, dataset, seed: int = 0) -> Dict:
        """Generate a synthetic input example"""

        prompt = f"""Generate a realistic training example for this task:

Task: {task.name}
Description: {task.description}
Input schema: {task.input_schema}
Output schema: {task.output_schema}

Return JSON:
{{
  "question": "...",
  "context": "..."
}}

Example #{seed + 1}:"""

        response = self.llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except:
            return {"question": "When?", "context": "In 1995"}

    def _validate_rule(self, rule: Rule) -> bool:
        """Validate that a rule's syntax is correct before saving"""
        try:
            if rule.format == RuleFormat.REGEX:
                # Test compile regex pattern
                re.compile(rule.content)
            elif rule.format == RuleFormat.CODE:
                # Test syntax of Python code
                compile(rule.content, "<string>", "exec")
                # Also check that extract function is defined
                if "def extract(" not in rule.content:
                    print("      Code rule must define extract() function")
                    return False
            return True
        except re.error as e:
            print(f"      Regex error: {e}")
            return False
        except SyntaxError as e:
            print(f"      Python syntax error: {e}")
            return False
        except Exception as e:
            print(f"      Validation error: {e}")
            return False

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid

        return str(uuid.uuid4())[:8]
