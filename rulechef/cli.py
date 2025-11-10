"""Interactive CLI for RuleChef"""

import os
from anthropic import Anthropic
from rulechef.engine import RuleChef
from rulechef.core import Task


def main():
    """Main CLI loop"""

    # Initialize
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter ANTHROPIC_API_KEY: ")

    client = Anthropic(api_key=api_key)

    # Create task
    task = Task(
        name="Q&A",
        description="Extract answer spans from text",
        input_schema={"question": "str", "context": "str"},
        output_schema={"spans": "List[Span]"}
    )

    chef = RuleChef(task, client)

    # Main menu
    while True:
        print("\n" + "="*60)
        print("RuleChef - Learn extraction rules")
        print("="*60)
        print("[a]dd example     [e]xtract     [g]enerate examples")
        print("[l]earn rules     [r]ules       [s]tats     [q]uit")
        print("="*60)

        choice = input("\nChoice: ").strip().lower()

        if choice in ['a', 'add']:
            _add_example(chef)
        elif choice in ['e', 'extract']:
            _extract(chef)
        elif choice in ['g', 'generate']:
            _generate_examples(chef)
        elif choice in ['l', 'learn']:
            _learn_rules(chef)
        elif choice in ['r', 'rules']:
            _view_rules(chef)
        elif choice in ['s', 'stats']:
            _view_stats(chef)
        elif choice in ['q', 'quit']:
            print("Goodbye!")
            break
        else:
            print("Unknown command")


def _add_example(chef: RuleChef):
    """Add a training example"""
    print("\n--- Add Example ---")
    question = input("Question: ").strip()
    context = input("Context (multi-line, end with blank line):\n")
    lines = [context]
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    context = "\n".join(lines)

    answer_text = input("Answer text: ").strip()
    answer_start = int(input("Answer start position: "))
    answer_end = int(input("Answer end position: "))

    chef.add_example(
        {"question": question, "context": context},
        {
            "spans": [
                {"text": answer_text, "start": answer_start, "end": answer_end}
            ]
        }
    )


def _extract(chef: RuleChef):
    """Extract from input"""
    print("\n--- Extract ---")
    question = input("Question: ").strip()
    context = input("Context (multi-line, end with blank line):\n")
    lines = [context]
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    context = "\n".join(lines)

    result = chef.extract({"question": question, "context": context})
    spans = result.get("spans", [])

    print(f"\nExtracted {len(spans)} span(s):")
    for span in spans:
        print(f"  - {span}")

    # Ask for correction if needed
    if spans:
        correct = input("\nCorrect? (y/n): ").strip().lower()
        if correct != 'y':
            expected_text = input("Correct answer: ").strip()
            expected_start = int(input("Start: "))
            expected_end = int(input("End: "))
            chef.add_correction(
                {"question": question, "context": context},
                result,
                {"spans": [{"text": expected_text, "start": expected_start, "end": expected_end}]},
                input("Feedback (optional): ").strip() or None
            )


def _generate_examples(chef: RuleChef):
    """Generate examples with LLM"""
    num = int(input("How many examples? (default 5): ") or "5")
    chef.generate_llm_examples(num)


def _learn_rules(chef: RuleChef):
    """Learn rules"""
    stats = chef.get_stats()
    total = stats['corrections'] + stats['examples']

    if total < 1:
        print(f"Need at least 1 training item (have {total})")
        return

    confirm = input(f"\nLearn from {total} items? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    chef.learn_rules(run_evaluation=True)


def _view_rules(chef: RuleChef):
    """View learned rules"""
    if not chef.dataset.rules:
        print("\nNo rules learned yet")
        return

    print("\n--- Learned Rules ---\n")
    for i, rule in enumerate(chef.get_rules_summary(), 1):
        print(f"{i}. {rule['name']}")
        print(f"   {rule['description']}")
        print(f"   Format: {rule['format']}, Priority: {rule['priority']}")
        print(f"   Confidence: {rule['confidence']}, Success: {rule['success_rate']}\n")


def _view_stats(chef: RuleChef):
    """View dataset statistics"""
    stats = chef.get_stats()
    print("\n--- Dataset Stats ---")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
