import os

from anthropic import Anthropic
from rulechef import RuleChef, Task
from rulechef.core import RuleFormat

# Setup
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
task = Task(
    name="Q&A",
    description="Extract answer spans from text",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
)

chef = RuleChef(task, client, allowed_formats=[RuleFormat.REGEX])

# Add examples
chef.add_example(
    {"question": "When?", "context": "Built in 1991"},
    {"spans": [{"text": "1991", "start": 9, "end": 13}]},
)

# Learn rules
chef.learn_rules()

print(chef.dataset.rules)
# Use
result = chef.extract(
    {"question": "When?", "context": "The building was constructed in 2025."}
)
spans = result.get("spans", [])
print([s["text"] if isinstance(s, dict) else s.text for s in spans])  # ["1995"]
