"""Basic example usage of RuleChef"""

from anthropic import Anthropic
from rulechef import RuleChef, Task

# Setup
client = Anthropic()
task = Task(
    name="Q&A",
    description="Extract answer spans from historical text",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
)

chef = RuleChef(task, client, "qa_example")

# Add training examples
print("Adding training examples...")
chef.add_example(
    {
        "question": "When was it built?",
        "context": "The building was constructed in 1995.",
    },
    {"spans": [{"text": "1995", "start": 39, "end": 43}]},
)

chef.add_example(
    {
        "question": "What year?",
        "context": "Established in 1987, it became famous worldwide.",
    },
    {"spans": [{"text": "1987", "start": 13, "end": 17}]},
)

chef.add_example(
    {
        "question": "When did it happen?",
        "context": "The event occurred on January 15, 2001.",
    },
    {"spans": [{"text": "January 15, 2001", "start": 22, "end": 38}]},
)

# Learn rules
print("\nLearning rules...")
rules, metrics = chef.learn_rules()

# Test extraction
print("\n--- Testing Extraction ---")
test_input = {"question": "When was it built?", "context": "Built in 1990"}
result = chef.extract(test_input)
print(f"Result: {result}")

# View learned rules
print("\nLearned rules:")
for rule in chef.get_rules_summary():
    print(f"- {rule['name']}: {rule['description']}")
