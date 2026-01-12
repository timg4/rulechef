"""OpenAI protocol wrapper for observing LLM interactions"""

import json
import re
from typing import Dict, Any, Callable, Optional
from rulechef.buffer import ExampleBuffer
from rulechef.core import Task


class OpenAIObserver:
    """
    Wraps OpenAI-compatible clients to observe calls and extract training examples.

    Works with any OpenAI protocol client (OpenAI, Azure, local models, etc.)
    """

    def __init__(
        self,
        buffer: ExampleBuffer,
        task: Task,
        extract_input: Optional[Callable] = None,
        extract_output: Optional[Callable] = None,
    ):
        """
        Args:
            buffer: ExampleBuffer to store observed examples
            task: Task definition (for parsing)
            extract_input: Custom function to parse API kwargs into task input
            extract_output: Custom function to parse API response into task output
        """
        self.buffer = buffer
        self.task = task
        self._original_create = None
        self._client = None

        # Custom extractors (user can override for their task)
        self._extract_input_fn = extract_input or self._default_extract_input
        self._extract_output_fn = extract_output or self._default_extract_output

    def attach(self, client):
        """
        Attach to OpenAI client, start observing.

        Args:
            client: OpenAI client (or compatible)

        Returns:
            Wrapped client (use this for API calls)
        """
        self._client = client

        # Save original method
        self._original_create = client.chat.completions.create

        # Wrap with observation
        def observed_create(*args, **kwargs):
            msgs = kwargs.get("messages")
            if isinstance(msgs, dict):
                kwargs["messages"] = [msgs]
            elif isinstance(msgs, tuple):
                kwargs["messages"] = list(msgs)
            # Call original API
            response = self._original_create(*args, **kwargs)

            # Extract and record
            try:
                input_data = self._extract_input_fn(kwargs)
                output_data = self._extract_output_fn(response)

                if input_data and output_data:
                    self.buffer.add_llm_observation(
                        input_data,
                        output_data,
                        metadata={
                            "model": kwargs.get("model"),
                            "temperature": kwargs.get("temperature"),
                        },
                    )
            except Exception as e:
                # Don't break user's code if extraction fails
                print(f"Warning: Failed to extract example from LLM call: {e}")

            return response

        # Replace method
        client.chat.completions.create = observed_create

        return client

    def detach(self):
        """Stop observing, restore original client"""
        if self._client and self._original_create:
            self._client.chat.completions.create = self._original_create
            self._client = None
            self._original_create = None

    def _default_extract_input(self, api_kwargs: Dict) -> Optional[Dict[str, Any]]:
        """
        Default input extractor for Q&A tasks.

        Override this for custom tasks by passing extract_input to __init__.
        """
        messages = api_kwargs.get("messages", [])
        if not messages:
            return None

        # Find last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return None

        # Try to parse Q&A format from message
        # Expected format: "Question: ...\nContext: ..."
        question_match = re.search(
            r"Question:\s*(.+?)(?:\n|$)", user_message, re.IGNORECASE
        )
        context_match = re.search(
            r"Context:\s*(.+)", user_message, re.IGNORECASE | re.DOTALL
        )

        if question_match and context_match:
            return {
                "question": question_match.group(1).strip(),
                "context": context_match.group(1).strip(),
            }

        # Fallback: treat entire message as context with empty question
        return {"question": "", "context": user_message}

    def _default_extract_output(self, response) -> Optional[Dict[str, Any]]:
        """
        Default output extractor for Q&A tasks.

        Expects JSON response with spans.
        Override this for custom tasks by passing extract_output to __init__.
        """
        try:
            # Get response content
            content = response.choices[0].message.content

            # Try to parse as JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Validate it has spans
            if "spans" in data:
                return data
            else:
                # Try to extract spans from any format
                return {"spans": data.get("spans", [])}

        except Exception:
            # If not JSON, return as plain text (could be answer)
            content = response.choices[0].message.content
            return {"spans": [{"text": content, "start": 0, "end": len(content)}]}


def create_qa_extractor():
    """Helper: Create extractor for Q&A task"""

    def extract_input(api_kwargs):
        messages = api_kwargs.get("messages", [])
        if not messages:
            return None

        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return None

        question_match = re.search(
            r"Question:\s*(.+?)(?:\n|$)", user_message, re.IGNORECASE
        )
        context_match = re.search(
            r"Context:\s*(.+)", user_message, re.IGNORECASE | re.DOTALL
        )

        if question_match and context_match:
            return {
                "question": question_match.group(1).strip(),
                "context": context_match.group(1).strip(),
            }

        return {"question": "", "context": user_message}

    def extract_output(response):
        try:
            content = response.choices[0].message.content

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            return data if "spans" in data else {"spans": []}

        except Exception:
            content = response.choices[0].message.content
            return {"spans": [{"text": content, "start": 0, "end": len(content)}]}

    return extract_input, extract_output
