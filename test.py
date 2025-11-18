import os

from openai import OpenAI
from rulechef import RuleChef, Task
from rulechef.core import RuleFormat

# Setup
api_key = os.environ.get("GROQ_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1/")

task = Task(
    name="Q&A",
    description="Extract answer spans from text",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
)

chef = RuleChef(
    task,
    client,
    allowed_formats=[RuleFormat.REGEX],
    model="moonshotai/kimi-k2-instruct-0905",
)

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
