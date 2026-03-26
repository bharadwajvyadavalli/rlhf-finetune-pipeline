"""
Chat template formatting module for various model architectures.

This module provides utilities for formatting conversations into the
appropriate chat templates required by different model families:
- ChatML (OpenAI, Mistral)
- Alpaca (Stanford)
- Vicuna (LMSYS)
- Llama-2 (Meta)
- Zephyr (HuggingFace)

Proper formatting is crucial for models to understand conversation structure
and generate appropriate responses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from datasets import Dataset


class TemplateType(Enum):
    """Supported chat template types."""

    CHATML = "chatml"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    LLAMA2 = "llama2"
    ZEPHYR = "zephyr"
    MISTRAL = "mistral"
    PHI = "phi"
    CUSTOM = "custom"


class Role(Enum):
    """Conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The message content
        name: Optional name for function messages
    """

    role: Role
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class Conversation:
    """A conversation consisting of multiple messages.

    Attributes:
        messages: List of messages in the conversation
        system_message: Optional system message to prepend
    """

    messages: list[Message] = field(default_factory=list)
    system_message: str | None = None

    def add_message(self, role: Role, content: str) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message.

        Args:
            content: Message content
        """
        self.add_message(Role.USER, content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message.

        Args:
            content: Message content
        """
        self.add_message(Role.ASSISTANT, content)

    def to_dict_list(self) -> list[dict[str, str]]:
        """Convert conversation to list of message dictionaries.

        Returns:
            List of message dictionaries
        """
        result = []
        if self.system_message:
            result.append({"role": "system", "content": self.system_message})
        result.extend([msg.to_dict() for msg in self.messages])
        return result


@dataclass
class TemplateConfig:
    """Configuration for a chat template.

    Attributes:
        template_type: Type of template
        system_prefix: Prefix for system messages
        user_prefix: Prefix for user messages
        assistant_prefix: Prefix for assistant messages
        system_suffix: Suffix for system messages
        user_suffix: Suffix for user messages
        assistant_suffix: Suffix for assistant messages
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        sep_token: Separator between messages
    """

    template_type: TemplateType = TemplateType.CHATML
    system_prefix: str = ""
    user_prefix: str = ""
    assistant_prefix: str = ""
    system_suffix: str = ""
    user_suffix: str = ""
    assistant_suffix: str = ""
    bos_token: str = ""
    eos_token: str = ""
    sep_token: str = ""


class BaseTemplateFormatter(ABC):
    """Abstract base class for template formatters."""

    def __init__(self, config: TemplateConfig | None = None) -> None:
        self.config = config or TemplateConfig()

    @abstractmethod
    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a conversation into template string.

        Args:
            conversation: Conversation to format
            add_generation_prompt: Whether to add prompt for generation

        Returns:
            Formatted string
        """
        pass

    @abstractmethod
    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format an instruction-input-response triple.

        Args:
            instruction: The instruction
            input_text: Optional input context
            response: Optional response

        Returns:
            Formatted string
        """
        pass

    def get_response_template(self) -> str:
        """Get the response template marker for loss masking.

        Returns:
            The string that marks the start of the assistant response
        """
        return self.config.assistant_prefix


class ChatMLFormatter(BaseTemplateFormatter):
    """Formatter for ChatML template (OpenAI, Mistral).

    Format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the ChatML formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.CHATML,
            system_prefix="<|im_start|>system\n",
            user_prefix="<|im_start|>user\n",
            assistant_prefix="<|im_start|>assistant\n",
            system_suffix="<|im_end|>\n",
            user_suffix="<|im_end|>\n",
            assistant_suffix="<|im_end|>\n",
            eos_token="<|im_end|>",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in ChatML format."""
        parts = []

        # Add system message
        if conversation.system_message:
            parts.append(
                f"{self.config.system_prefix}{conversation.system_message}{self.config.system_suffix}"
            )

        # Add messages
        for msg in conversation.messages:
            if msg.role == Role.USER:
                parts.append(f"{self.config.user_prefix}{msg.content}{self.config.user_suffix}")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"{self.config.assistant_prefix}{msg.content}{self.config.assistant_suffix}")
            elif msg.role == Role.SYSTEM:
                parts.append(f"{self.config.system_prefix}{msg.content}{self.config.system_suffix}")

        # Add generation prompt
        if add_generation_prompt:
            parts.append(self.config.assistant_prefix)

        return "".join(parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in ChatML format."""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        parts = [
            f"{self.config.user_prefix}{user_content}{self.config.user_suffix}",
        ]

        if response:
            parts.append(f"{self.config.assistant_prefix}{response}{self.config.assistant_suffix}")
        else:
            parts.append(self.config.assistant_prefix)

        return "".join(parts)


class AlpacaFormatter(BaseTemplateFormatter):
    """Formatter for Alpaca template.

    Format:
    Below is an instruction that describes a task...

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {response}
    """

    SYSTEM_PROMPT = (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the request."
    )

    SYSTEM_PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the Alpaca formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.ALPACA,
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            user_suffix="\n\n",
            assistant_suffix="\n\n",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in Alpaca format.

        Note: Alpaca format is primarily for single-turn instruction following.
        For multi-turn, we concatenate all user messages as instruction.
        """
        # Extract user and assistant messages
        user_messages = [m.content for m in conversation.messages if m.role == Role.USER]
        assistant_messages = [m.content for m in conversation.messages if m.role == Role.ASSISTANT]

        instruction = "\n".join(user_messages) if user_messages else ""
        response = "\n".join(assistant_messages) if assistant_messages else None

        return self.format_instruction(instruction, "", response if not add_generation_prompt else None)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in Alpaca format."""
        if input_text:
            parts = [
                f"{self.SYSTEM_PROMPT}\n\n",
                f"### Instruction:\n{instruction}\n\n",
                f"### Input:\n{input_text}\n\n",
                "### Response:\n",
            ]
        else:
            parts = [
                f"{self.SYSTEM_PROMPT_NO_INPUT}\n\n",
                f"### Instruction:\n{instruction}\n\n",
                "### Response:\n",
            ]

        if response:
            parts.append(response)

        return "".join(parts)

    def get_response_template(self) -> str:
        """Get the response template marker."""
        return "### Response:\n"


class Llama2Formatter(BaseTemplateFormatter):
    """Formatter for Llama-2 chat template.

    Format:
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {user_message} [/INST] {assistant_message} </s>
    """

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the Llama-2 formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.LLAMA2,
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="[INST] ",
            user_suffix=" [/INST] ",
            system_prefix="<<SYS>>\n",
            system_suffix="\n<</SYS>>\n\n",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in Llama-2 format."""
        parts = [self.config.bos_token]

        # Handle system message in first turn
        system_handled = False

        for i, msg in enumerate(conversation.messages):
            if msg.role == Role.USER:
                if i == 0 and conversation.system_message and not system_handled:
                    # Include system in first user message
                    parts.append(
                        f"[INST] <<SYS>>\n{conversation.system_message}\n<</SYS>>\n\n{msg.content} [/INST]"
                    )
                    system_handled = True
                else:
                    if i > 0:
                        parts.append(f"<s>[INST] {msg.content} [/INST]")
                    else:
                        parts.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == Role.ASSISTANT:
                parts.append(f" {msg.content} </s>")

        if add_generation_prompt:
            # Ensure we're ready for assistant response
            if not parts[-1].endswith("[/INST]"):
                parts.append(" ")

        return "".join(parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in Llama-2 format."""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        result = f"<s>[INST] {user_content} [/INST]"

        if response:
            result += f" {response} </s>"

        return result

    def get_response_template(self) -> str:
        """Get the response template marker."""
        return "[/INST]"


class VicunaFormatter(BaseTemplateFormatter):
    """Formatter for Vicuna chat template.

    Format:
    USER: {user_message}
    ASSISTANT: {assistant_message}</s>
    """

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the Vicuna formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.VICUNA,
            user_prefix="USER: ",
            user_suffix="\n",
            assistant_prefix="ASSISTANT: ",
            assistant_suffix="</s>\n",
            eos_token="</s>",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in Vicuna format."""
        parts = []

        # Add system message
        if conversation.system_message:
            parts.append(f"{conversation.system_message}\n\n")

        # Add messages
        for msg in conversation.messages:
            if msg.role == Role.USER:
                parts.append(f"USER: {msg.content}\n")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"ASSISTANT: {msg.content}</s>\n")

        if add_generation_prompt:
            parts.append("ASSISTANT:")

        return "".join(parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in Vicuna format."""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        result = f"USER: {user_content}\nASSISTANT:"

        if response:
            result += f" {response}</s>"

        return result

    def get_response_template(self) -> str:
        """Get the response template marker."""
        return "ASSISTANT:"


class ZephyrFormatter(BaseTemplateFormatter):
    """Formatter for Zephyr chat template.

    Format:
    <|system|>
    {system_message}</s>
    <|user|>
    {user_message}</s>
    <|assistant|>
    {assistant_message}</s>
    """

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the Zephyr formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.ZEPHYR,
            system_prefix="<|system|>\n",
            user_prefix="<|user|>\n",
            assistant_prefix="<|assistant|>\n",
            system_suffix="</s>\n",
            user_suffix="</s>\n",
            assistant_suffix="</s>\n",
            eos_token="</s>",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in Zephyr format."""
        parts = []

        # Add system message
        if conversation.system_message:
            parts.append(f"<|system|>\n{conversation.system_message}</s>\n")

        # Add messages
        for msg in conversation.messages:
            if msg.role == Role.USER:
                parts.append(f"<|user|>\n{msg.content}</s>\n")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"<|assistant|>\n{msg.content}</s>\n")

        if add_generation_prompt:
            parts.append("<|assistant|>\n")

        return "".join(parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in Zephyr format."""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        result = f"<|user|>\n{user_content}</s>\n<|assistant|>\n"

        if response:
            result += f"{response}</s>"

        return result

    def get_response_template(self) -> str:
        """Get the response template marker."""
        return "<|assistant|>\n"


class MistralFormatter(BaseTemplateFormatter):
    """Formatter for Mistral chat template.

    Format:
    <s>[INST] {instruction} [/INST] {response}</s>
    """

    def __init__(self, config: TemplateConfig | None = None) -> None:
        """Initialize the Mistral formatter."""
        default_config = TemplateConfig(
            template_type=TemplateType.MISTRAL,
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="[INST] ",
            user_suffix=" [/INST]",
        )
        super().__init__(config or default_config)

    def format_conversation(
        self,
        conversation: Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format conversation in Mistral format."""
        parts = ["<s>"]

        for i, msg in enumerate(conversation.messages):
            if msg.role == Role.USER:
                parts.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"{msg.content}</s>")

        return "".join(parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format instruction in Mistral format."""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        result = f"<s>[INST] {user_content} [/INST]"

        if response:
            result += f" {response}</s>"

        return result

    def get_response_template(self) -> str:
        """Get the response template marker."""
        return "[/INST]"


# Template registry
FORMATTERS: dict[TemplateType, type[BaseTemplateFormatter]] = {
    TemplateType.CHATML: ChatMLFormatter,
    TemplateType.ALPACA: AlpacaFormatter,
    TemplateType.LLAMA2: Llama2Formatter,
    TemplateType.VICUNA: VicunaFormatter,
    TemplateType.ZEPHYR: ZephyrFormatter,
    TemplateType.MISTRAL: MistralFormatter,
}

# Model name to template type mapping
MODEL_TEMPLATES: dict[str, TemplateType] = {
    "llama-2": TemplateType.LLAMA2,
    "llama2": TemplateType.LLAMA2,
    "mistral": TemplateType.MISTRAL,
    "mixtral": TemplateType.MISTRAL,
    "zephyr": TemplateType.ZEPHYR,
    "vicuna": TemplateType.VICUNA,
    "alpaca": TemplateType.ALPACA,
    "chatml": TemplateType.CHATML,
    "openai": TemplateType.CHATML,
    "gpt": TemplateType.CHATML,
}


class ChatTemplateFormatter:
    """Unified interface for chat template formatting.

    Factory class that creates the appropriate formatter based on
    the specified template type.
    """

    def __init__(
        self,
        template_type: TemplateType | str = TemplateType.CHATML,
        custom_config: TemplateConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the chat template formatter.

        Args:
            template_type: Template type
            custom_config: Custom configuration
            tokenizer: Optional tokenizer
        """
        if isinstance(template_type, str):
            template_type = TemplateType(template_type.lower())

        self.template_type = template_type
        self.tokenizer = tokenizer

        # Get the appropriate formatter
        formatter_cls = FORMATTERS.get(template_type, ChatMLFormatter)
        self._formatter = formatter_cls(custom_config)

    def format(
        self,
        messages: list[dict[str, str]] | Conversation,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages into template string.

        Args:
            messages: Messages or Conversation object
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatted string
        """
        if isinstance(messages, list):
            # Convert dict list to Conversation
            conversation = Conversation()
            for msg in messages:
                role = Role(msg.get("role", "user"))
                content = msg.get("content", "")
                if role == Role.SYSTEM:
                    conversation.system_message = content
                else:
                    conversation.add_message(role, content)
        else:
            conversation = messages

        # Try tokenizer's chat_template first if available
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    conversation.to_dict_list(),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        return self._formatter.format_conversation(conversation, add_generation_prompt)

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        response: str | None = None,
    ) -> str:
        """Format an instruction.

        Args:
            instruction: The instruction
            input_text: Optional input
            response: Optional response

        Returns:
            Formatted string
        """
        return self._formatter.format_instruction(instruction, input_text, response)

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        output_text_column: str = "text",
    ) -> Dataset:
        """Format an entire dataset.

        Args:
            dataset: Dataset to format
            instruction_column: Column with instructions
            input_column: Column with inputs
            output_column: Column with outputs
            output_text_column: Name for output text column

        Returns:
            Formatted dataset with new text column
        """
        def format_example(example: dict) -> dict:
            instruction = example.get(instruction_column, "")
            input_text = example.get(input_column, "")
            output = example.get(output_column, "")

            formatted = self.format_instruction(instruction, input_text, output)
            return {**example, output_text_column: formatted}

        return dataset.map(format_example)

    def get_special_tokens(self) -> dict[str, str]:
        """Get special tokens used by the template.

        Returns:
            Dictionary of special tokens
        """
        return {
            "bos_token": self._formatter.config.bos_token,
            "eos_token": self._formatter.config.eos_token,
            "sep_token": self._formatter.config.sep_token,
        }

    def get_response_template(self) -> str:
        """Get the response template for loss masking.

        Returns:
            The string marking start of assistant response
        """
        return self._formatter.get_response_template()

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "ChatTemplateFormatter":
        """Create formatter from a tokenizer's chat template.

        Args:
            tokenizer: HuggingFace tokenizer with chat_template

        Returns:
            Configured ChatTemplateFormatter
        """
        # Try to detect template type from tokenizer
        template_type = TemplateType.CHATML

        if hasattr(tokenizer, 'name_or_path'):
            model_name = tokenizer.name_or_path.lower()
            for key, ttype in MODEL_TEMPLATES.items():
                if key in model_name:
                    template_type = ttype
                    break

        return cls(template_type=template_type, tokenizer=tokenizer)

    @classmethod
    def get_template_for_model(cls, model_name: str) -> "ChatTemplateFormatter":
        """Get the appropriate formatter for a model.

        Args:
            model_name: Name/path of the model

        Returns:
            Appropriate ChatTemplateFormatter
        """
        model_lower = model_name.lower()

        for key, template_type in MODEL_TEMPLATES.items():
            if key in model_lower:
                return cls(template_type=template_type)

        # Default to ChatML
        return cls(template_type=TemplateType.CHATML)


def format_for_model(
    model_name: str,
    instruction: str,
    input_text: str = "",
    response: str | None = None,
) -> str:
    """Convenience function to format for a specific model.

    Args:
        model_name: Name of the model
        instruction: The instruction
        input_text: Optional input
        response: Optional response

    Returns:
        Formatted string
    """
    formatter = ChatTemplateFormatter.get_template_for_model(model_name)
    return formatter.format_instruction(instruction, input_text, response)


def convert_dataset(
    dataset: Dataset,
    model_name: str,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
) -> Dataset:
    """Convenience function to format a dataset for a model.

    Args:
        dataset: Dataset to format
        model_name: Model name for template selection
        instruction_column: Column with instructions
        input_column: Column with inputs
        output_column: Column with outputs

    Returns:
        Dataset with formatted_text column
    """
    formatter = ChatTemplateFormatter.get_template_for_model(model_name)
    return formatter.format_dataset(
        dataset,
        instruction_column=instruction_column,
        input_column=input_column,
        output_column=output_column,
        output_text_column="formatted_text",
    )
