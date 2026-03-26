"""
Custom data collators for causal language modeling and chat.

This module provides data collators that handle the specific requirements
of fine-tuning language models:

- Proper padding and attention masking
- Label masking for prompt tokens (completion-only training)
- Chat template handling with turn boundaries
- Efficient batching for variable-length sequences

Data collators are critical for correct training behavior, especially
when training on instruction/response pairs where we only want to compute
loss on the response tokens.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizer


# Response templates for different model formats
RESPONSE_TEMPLATES = {
    "alpaca": "### Response:",
    "chatml": "<|im_start|>assistant",
    "llama2": "[/INST]",
    "vicuna": "ASSISTANT:",
    "zephyr": "<|assistant|>",
    "mistral": "[/INST]",
}


@dataclass
class DataCollatorForCausalLM:
    """Data collator for standard causal language modeling.

    Handles padding, attention masking, and label preparation for
    next-token prediction training.

    Args:
        tokenizer: Tokenizer for padding
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value (for efficiency)
        return_tensors: Return type ("pt" for PyTorch)
        mlm: Whether this is masked language modeling (False for causal)
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    return_tensors: str = "pt"
    mlm: bool = False

    def __post_init__(self) -> None:
        """Set up pad token if not set."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of features.

        Args:
            features: List of tokenized examples

        Returns:
            Batched tensors with input_ids, attention_mask, and labels
        """
        # Extract input_ids from features
        if "input_ids" in features[0]:
            input_ids = [
                torch.tensor(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor)
                else f["input_ids"]
                for f in features
            ]
        else:
            raise ValueError("Features must contain 'input_ids'")

        # Pad sequences
        padded_input_ids = self.pad_sequence(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Create attention mask
        attention_mask = self.create_attention_mask(
            padded_input_ids,
            self.tokenizer.pad_token_id,
        )

        # Create labels
        labels = self.create_labels(
            padded_input_ids,
            self.tokenizer.pad_token_id,
        )

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch

    def pad_sequence(
        self,
        sequences: list[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        """Pad a list of sequences to the same length.

        Args:
            sequences: List of 1D tensors
            padding_value: Value to use for padding

        Returns:
            Padded 2D tensor
        """
        # Find max length
        max_len = max(len(seq) for seq in sequences)

        # Apply max_length limit
        max_len = min(max_len, self.max_length)

        # Round up to pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                       self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad each sequence
        padded = []
        for seq in sequences:
            # Truncate if needed
            if len(seq) > max_len:
                seq = seq[:max_len]

            # Pad to max_len
            if len(seq) < max_len:
                padding = torch.full((max_len - len(seq),), padding_value, dtype=seq.dtype)
                seq = torch.cat([seq, padding])

            padded.append(seq)

        return torch.stack(padded)

    def create_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Create attention mask from input IDs.

        Args:
            input_ids: Input token IDs
            pad_token_id: Padding token ID

        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        return (input_ids != pad_token_id).long()

    def create_labels(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Create labels for causal LM training.

        Labels are shifted input_ids with padding tokens set to -100
        to ignore them in the loss computation.

        Args:
            input_ids: Input token IDs
            pad_token_id: Padding token ID

        Returns:
            Labels tensor
        """
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        return labels


@dataclass
class DataCollatorForCompletionOnly:
    """Data collator that only computes loss on completion tokens.

    This collator masks out the prompt tokens in the labels, so the
    model only learns to generate the response/completion portion.
    Essential for instruction fine-tuning.

    Args:
        tokenizer: Tokenizer for padding
        response_template: String or token IDs marking response start
        instruction_template: Optional string marking instruction start
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        ignore_index: Label value to ignore in loss (-100 for PyTorch)
    """

    tokenizer: PreTrainedTokenizer
    response_template: str | list[int] = "### Response:"
    instruction_template: str | list[int] | None = None
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    ignore_index: int = -100

    _response_template_ids: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize response template IDs."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._response_template_ids = self.get_response_template_ids()

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch with completion-only labels.

        Args:
            features: List of tokenized examples

        Returns:
            Batched tensors with prompt tokens masked in labels
        """
        # Use base collator for padding
        base_collator = DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = base_collator(features)

        # Mask prompt tokens in labels
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]

            # Find where response starts
            response_start = self.find_response_start(input_ids)

            if response_start > 0:
                # Mask all tokens before response
                labels = self.mask_prompt_tokens(labels, response_start)
                batch["labels"][i] = labels

        return batch

    def find_response_start(
        self,
        input_ids: torch.Tensor,
    ) -> int:
        """Find the index where the response starts.

        Args:
            input_ids: Token IDs for the sequence

        Returns:
            Index of first response token
        """
        input_ids_list = input_ids.tolist()
        template_ids = self._response_template_ids

        # Search for template in input_ids
        for i in range(len(input_ids_list) - len(template_ids) + 1):
            if input_ids_list[i:i + len(template_ids)] == template_ids:
                # Return index after the template
                return i + len(template_ids)

        # If not found, return 0 (train on everything)
        return 0

    def mask_prompt_tokens(
        self,
        labels: torch.Tensor,
        response_start_idx: int,
    ) -> torch.Tensor:
        """Mask prompt tokens in labels.

        Args:
            labels: Full labels tensor
            response_start_idx: Index where response starts

        Returns:
            Labels with prompt tokens set to ignore_index
        """
        labels[:response_start_idx] = self.ignore_index
        return labels

    def get_response_template_ids(self) -> list[int]:
        """Get token IDs for the response template.

        Returns:
            List of token IDs
        """
        if isinstance(self.response_template, list):
            return self.response_template
        else:
            # Tokenize without special tokens
            ids = self.tokenizer.encode(
                self.response_template,
                add_special_tokens=False,
            )
            return ids


@dataclass
class DataCollatorForChat:
    """Data collator for multi-turn chat conversations.

    Handles chat-specific formatting including:
    - Multiple turns with user/assistant alternation
    - System message handling
    - Per-turn label masking (only train on assistant turns)

    Args:
        tokenizer: Tokenizer with chat template
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        train_on_input: Whether to also train on user inputs
        ignore_index: Label value to ignore in loss
        mask_system_message: Whether to mask system message in labels
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    train_on_input: bool = False
    ignore_index: int = -100
    mask_system_message: bool = True

    # Role markers for detecting turns
    user_markers: list[str] = field(default_factory=lambda: [
        "<|im_start|>user", "[INST]", "USER:", "<|user|>", "Human:"
    ])
    assistant_markers: list[str] = field(default_factory=lambda: [
        "<|im_start|>assistant", "[/INST]", "ASSISTANT:", "<|assistant|>", "Assistant:"
    ])

    def __post_init__(self) -> None:
        """Initialize tokenizer pad token."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of chat conversations.

        Args:
            features: List of tokenized conversations

        Returns:
            Batched tensors with appropriate label masking
        """
        # Check if features contain 'messages' or 'input_ids'
        if "messages" in features[0]:
            # Process conversations
            processed = [self.process_conversation(f["messages"]) for f in features]
            features = processed

        # Use base collator for padding
        base_collator = DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = base_collator(features)

        if not self.train_on_input:
            # Mask non-assistant turns
            for i in range(len(batch["input_ids"])):
                input_ids = batch["input_ids"][i]
                labels = batch["labels"][i]

                turn_boundaries = self.find_turn_boundaries(input_ids)
                labels = self.mask_non_assistant_turns(labels, turn_boundaries)
                batch["labels"][i] = labels

        return batch

    def process_conversation(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        """Process a single conversation.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Tokenized conversation with labels
        """
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback: simple concatenation
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def find_turn_boundaries(
        self,
        input_ids: torch.Tensor,
    ) -> list[tuple[int, int, str]]:
        """Find the boundaries of each turn in the conversation.

        Args:
            input_ids: Token IDs for the conversation

        Returns:
            List of (start_idx, end_idx, role) tuples
        """
        boundaries = []
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find all user and assistant markers
        current_pos = 0
        current_role = "system"

        while current_pos < len(text):
            # Find next user marker
            user_pos = len(text)
            for marker in self.user_markers:
                pos = text.find(marker, current_pos)
                if pos != -1 and pos < user_pos:
                    user_pos = pos

            # Find next assistant marker
            assistant_pos = len(text)
            for marker in self.assistant_markers:
                pos = text.find(marker, current_pos)
                if pos != -1 and pos < assistant_pos:
                    assistant_pos = pos

            # Determine next turn
            if user_pos < assistant_pos:
                # User turn next
                if current_pos < user_pos:
                    boundaries.append((current_pos, user_pos, current_role))
                current_role = "user"
                current_pos = user_pos
            elif assistant_pos < user_pos:
                # Assistant turn next
                if current_pos < assistant_pos:
                    boundaries.append((current_pos, assistant_pos, current_role))
                current_role = "assistant"
                current_pos = assistant_pos
            else:
                # No more markers
                if current_pos < len(text):
                    boundaries.append((current_pos, len(text), current_role))
                break

            # Move past marker
            current_pos += 1

        # Convert character positions to token positions
        token_boundaries = []
        for start_char, end_char, role in boundaries:
            start_text = text[:start_char]
            end_text = text[:end_char]

            start_tokens = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_text, add_special_tokens=False))

            token_boundaries.append((start_tokens, end_tokens, role))

        return token_boundaries

    def mask_non_assistant_turns(
        self,
        labels: torch.Tensor,
        turn_boundaries: list[tuple[int, int, str]],
    ) -> torch.Tensor:
        """Mask labels for non-assistant turns.

        Args:
            labels: Full labels tensor
            turn_boundaries: Turn boundary information

        Returns:
            Labels with non-assistant turns masked
        """
        for start, end, role in turn_boundaries:
            if role != "assistant":
                # Mask this turn
                end = min(end, len(labels))
                start = min(start, len(labels))
                labels[start:end] = self.ignore_index

        return labels


def get_response_template_for_model(
    model_name: str,
) -> str:
    """Get the appropriate response template for a model.

    Args:
        model_name: Name or path of the model

    Returns:
        Response template string
    """
    model_lower = model_name.lower()

    if "llama-2" in model_lower or "llama2" in model_lower:
        return RESPONSE_TEMPLATES["llama2"]
    elif "mistral" in model_lower:
        return RESPONSE_TEMPLATES["mistral"]
    elif "zephyr" in model_lower:
        return RESPONSE_TEMPLATES["zephyr"]
    elif "vicuna" in model_lower:
        return RESPONSE_TEMPLATES["vicuna"]
    elif "chatml" in model_lower or "openai" in model_lower:
        return RESPONSE_TEMPLATES["chatml"]
    else:
        return RESPONSE_TEMPLATES["alpaca"]


def create_completion_only_collator(
    tokenizer: PreTrainedTokenizer,
    template_type: str = "alpaca",
    max_length: int = 2048,
) -> DataCollatorForCompletionOnly:
    """Create a completion-only collator for a specific template.

    Args:
        tokenizer: The tokenizer
        template_type: Chat template type
        max_length: Maximum sequence length

    Returns:
        Configured DataCollatorForCompletionOnly
    """
    response_template = RESPONSE_TEMPLATES.get(template_type, RESPONSE_TEMPLATES["alpaca"])

    return DataCollatorForCompletionOnly(
        tokenizer=tokenizer,
        response_template=response_template,
        max_length=max_length,
    )


@dataclass
class PackedDataCollator:
    """Data collator for sequence packing.

    Packs multiple examples into a single sequence for efficient training.
    Uses document boundaries to prevent cross-contamination.

    Args:
        tokenizer: Tokenizer for special tokens
        max_length: Maximum packed sequence length
        pad_to_multiple_of: Pad to multiple of this value
        separator_token: Token to separate packed examples
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    separator_token: str | int | None = None

    def __post_init__(self) -> None:
        """Initialize separator token."""
        if self.separator_token is None:
            self.separator_token = self.tokenizer.eos_token_id
        elif isinstance(self.separator_token, str):
            self.separator_token = self.tokenizer.encode(
                self.separator_token, add_special_tokens=False
            )[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Pack and collate features.

        Args:
            features: List of tokenized examples

        Returns:
            Packed batch tensors
        """
        # Extract input_ids
        sequences = [
            torch.tensor(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor)
            else f["input_ids"]
            for f in features
        ]

        # Pack sequences
        packed = self.pack_sequences(sequences)

        # Create batched tensors
        padded_input_ids = self._pad_packed(packed)
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()
        labels = padded_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def pack_sequences(
        self,
        sequences: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Pack multiple sequences into longer sequences.

        Args:
            sequences: List of token ID tensors

        Returns:
            List of packed sequences
        """
        packed = []
        current_pack = []
        current_length = 0

        for seq in sequences:
            seq_len = len(seq)

            # Check if adding this sequence would exceed max_length
            if current_length + seq_len + 1 > self.max_length:  # +1 for separator
                if current_pack:
                    # Finalize current pack
                    packed.append(torch.cat(current_pack))
                    current_pack = []
                    current_length = 0

            # Add sequence with separator
            if current_pack:
                current_pack.append(torch.tensor([self.separator_token]))
                current_length += 1

            current_pack.append(seq[:self.max_length - current_length])
            current_length += len(seq)

        # Add remaining
        if current_pack:
            packed.append(torch.cat(current_pack))

        return packed

    def _pad_packed(self, packed: list[torch.Tensor]) -> torch.Tensor:
        """Pad packed sequences."""
        max_len = max(len(p) for p in packed)

        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                       self.pad_to_multiple_of * self.pad_to_multiple_of)

        padded = []
        for seq in packed:
            if len(seq) < max_len:
                padding = torch.full(
                    (max_len - len(seq),),
                    self.tokenizer.pad_token_id,
                    dtype=seq.dtype
                )
                seq = torch.cat([seq, padding])
            padded.append(seq)

        return torch.stack(padded)

    def create_position_ids(
        self,
        input_ids: torch.Tensor,
        document_boundaries: list[int],
    ) -> torch.Tensor:
        """Create position IDs respecting document boundaries.

        Args:
            input_ids: Packed input token IDs
            document_boundaries: Indices of document separators

        Returns:
            Position IDs tensor
        """
        position_ids = torch.zeros_like(input_ids)
        current_pos = 0

        boundaries = [0] + document_boundaries + [len(input_ids)]

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            doc_length = end - start
            position_ids[start:end] = torch.arange(doc_length)

        return position_ids
