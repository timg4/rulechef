"""Main RuleChef orchestrator"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from anthropic import Anthropic

from rulechef.core import Task, Dataset, Example, Correction, Rule, RuleFormat, Span
from rulechef.learner import RuleLearner


class RuleChef:
    """Main orchestrator for learning and applying rules"""

    def __init__(
        self,
        task: Task,
        client: Optional[Anthropic] = None,
        dataset_name: str = "default",
        storage_path: str = "./rulechef_data",
        allowed_formats: Optional[List[RuleFormat]] = None
    ):
        self.task = task
        self.llm = client or Anthropic()
        self.dataset = Dataset(name=dataset_name, task=task)
        self.storage_path = Path(storage_path)
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]
        self.learner = RuleLearner(self.llm, allowed_formats=self.allowed_formats)

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing dataset if available
        self._load_dataset()

    # ========================================
    # Data Collection
    # ========================================

    def add_example(
        self,
        input_data: Dict,
        output_data: Dict,
        source: str = "human_labeled"
    ):
        """Add a labeled training example"""
        example = Example(
            id=self._generate_id(),
            input=input_data,
            expected_output=output_data,
            source=source
        )
        self.dataset.examples.append(example)
        self._save_dataset()
        print(f"✓ Added example (total: {len(self.dataset.examples)})")

    def add_correction(
        self,
        input_data: Dict,
        model_output: Dict,
        expected_output: Dict,
        feedback: Optional[str] = None
    ):
        """Add a user correction (high value signal)"""
        correction = Correction(
            id=self._generate_id(),
            input=input_data,
            model_output=model_output,
            expected_output=expected_output,
            feedback=feedback
        )
        self.dataset.corrections.append(correction)
        self._save_dataset()
        print(f"✓ Added correction (total: {len(self.dataset.corrections)})")

    def add_feedback(self, feedback: str):
        """Add general user feedback"""
        self.dataset.feedback.append(feedback)
        self._save_dataset()

    def generate_llm_examples(self, num_examples: int = 5, seed: int = 42):
        """Generate synthetic training examples using LLM"""
        print(f"\n🤖 Generating {num_examples} examples with LLM...")
        for i in range(num_examples):
            input_data = self.learner._generate_input(self.task, self.dataset, seed + i)
            self.add_example(input_data, {"spans": []}, source="llm_generated")

    # ========================================
    # Learning
    # ========================================

    def learn_rules(self, run_evaluation: bool = True, min_examples: int = 1):
        """
        Learn rules from all collected data

        This is the core batch learning process
        """

        total_data = len(self.dataset.get_all_training_data())

        if total_data < min_examples:
            print(f"Need at least {min_examples} examples/corrections")
            print(f"Currently have: {len(self.dataset.corrections)} corrections, "
                  f"{len(self.dataset.examples)} examples")
            return

        print(f"\n{'='*60}")
        print(f"Learning rules from {total_data} training items")
        print(f"  Corrections: {len(self.dataset.corrections)} (high value)")
        print(f"  Examples: {len(self.dataset.examples)}")
        print(f"{'='*60}\n")

        # Synthesize initial ruleset
        rules = self.learner.synthesize_ruleset(self.dataset)

        if not rules:
            print("Failed to synthesize rules")
            return None

        print(f"✓ Generated {len(rules)} rules")

        # Evaluate and refine
        if run_evaluation:
            rules, metrics = self.learner.evaluate_and_refine(rules, self.dataset)
        else:
            metrics = {"accuracy": 0.0, "total": total_data, "correct": 0}

        # Save learned rules
        self.dataset.rules = rules
        self._save_dataset()

        print(f"\n{'='*60}")
        print(f"Learning complete!")
        print(f"  Rules: {len(rules)}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"{'='*60}\n")

        return rules, metrics

    # ========================================
    # Execution
    # ========================================

    def extract(self, input_data: Dict) -> Dict:
        """
        Extract spans from input
        Uses learned rules first, falls back to LLM if low confidence
        """

        if not self.dataset.rules:
            # No rules learned, use LLM fallback
            return self._execute_with_llm(input_data)

        # Apply rules
        output = self.learner._apply_rules(self.dataset.rules, input_data)

        # Store current extraction for potential correction
        self.current_extraction = output

        # Check confidence
        spans = output.get("spans", [])
        if spans:
            avg_confidence = sum(s.get("score", 0.5) for s in spans) / len(spans)
            if avg_confidence < 0.3:
                print(f"Low confidence ({avg_confidence:.1%}), using LLM fallback...")
                return self._execute_with_llm(input_data)

        return output

    def _execute_with_llm(self, input_data: Dict) -> Dict:
        """Execute extraction using LLM directly"""

        # Format input for prompt
        prompt = f"""Extract answer spans from the following:

Question: {input_data.get('question', '')}
Context: {input_data.get('context', '')}

Return JSON:
{{
  "spans": [
    {{"text": "...", "start": 0, "end": 10}}
  ]
}}
"""

        response = self.llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            return {"spans": []}

    # ========================================
    # Utils
    # ========================================

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            "task": self.dataset.task.name,
            "dataset": self.dataset.name,
            "corrections": len(self.dataset.corrections),
            "examples": len(self.dataset.examples),
            "feedback": len(self.dataset.feedback),
            "rules": len(self.dataset.rules),
            "description": self.dataset.description
        }

    def get_rules_summary(self) -> List[Dict]:
        """Get formatted summary of learned rules"""
        summaries = []
        for rule in sorted(self.dataset.rules, key=lambda r: r.priority, reverse=True):
            success_rate = (rule.successes / rule.times_applied * 100
                           if rule.times_applied > 0 else 0)
            summaries.append({
                "name": rule.name,
                "description": rule.description,
                "format": rule.format.value,
                "priority": rule.priority,
                "confidence": f"{rule.confidence:.2f}",
                "times_applied": rule.times_applied,
                "success_rate": f"{success_rate:.1f}%" if rule.times_applied > 0 else "N/A"
            })
        return summaries

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    # ========================================
    # Persistence
    # ========================================

    def _save_dataset(self):
        """Save dataset to disk"""
        filepath = self.storage_path / f"{self.dataset.name}.json"
        with open(filepath, 'w') as f:
            json.dump(self.dataset.to_dict(), f, indent=2, default=str)

    def _load_dataset(self):
        """Load dataset from disk if it exists"""
        filepath = self.storage_path / f"{self.dataset.name}.json"

        if not filepath.exists():
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore examples
            for ex in data.get("examples", []):
                example = Example(
                    id=ex["id"],
                    input=ex["input"],
                    expected_output=ex["expected_output"],
                    source=ex["source"],
                    confidence=ex.get("confidence", 0.8)
                )
                self.dataset.examples.append(example)

            # Restore corrections
            for corr in data.get("corrections", []):
                correction = Correction(
                    id=corr["id"],
                    input=corr["input"],
                    model_output=corr["model_output"],
                    expected_output=corr["expected_output"],
                    feedback=corr.get("feedback")
                )
                self.dataset.corrections.append(correction)

            # Restore feedback
            self.dataset.feedback = data.get("feedback", [])

            # Restore rules
            for rule_data in data.get("rules", []):
                rule = Rule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    format=RuleFormat(rule_data["format"]),
                    content=rule_data["content"],
                    priority=rule_data.get("priority", 5),
                    confidence=rule_data.get("confidence", 0.5),
                    times_applied=rule_data.get("times_applied", 0),
                    successes=rule_data.get("successes", 0),
                    failures=rule_data.get("failures", 0)
                )
                self.dataset.rules.append(rule)

            print(f"✓ Loaded dataset: {len(self.dataset.corrections)} corrections, {len(self.dataset.examples)} examples")

        except Exception as e:
            print(f"Error loading dataset: {e}")
