"""
Synthetic data generation module using teacher models.

This module implements methods for generating synthetic training data:
- Self-Instruct: Generate instructions from seed tasks using LLMs
- Evol-Instruct: Iteratively evolve instructions for complexity

Synthetic data generation enables bootstrapping diverse instruction-following
datasets without extensive human annotation.
"""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from datasets import Dataset
from tqdm import tqdm

# Lazy imports for optional dependencies
_TRANSFORMERS_AVAILABLE = False
_OPENAI_AVAILABLE = False
_ROUGE_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    pass


class EvolutionType(Enum):
    """Types of instruction evolution for Evol-Instruct."""

    ADD_CONSTRAINTS = "add_constraints"
    DEEPEN = "deepen"
    CONCRETIZE = "concretize"
    INCREASE_REASONING = "increase_reasoning"
    COMPLICATE_INPUT = "complicate_input"


# Evolution prompts for each evolution type
EVOLUTION_PROMPTS = {
    EvolutionType.ADD_CONSTRAINTS: """I want you to add constraints or requirements to the given instruction, making it more complex. The new instruction should be reasonable and can be understood and responded to by humans.

Original Instruction: {instruction}

Please add constraints or requirements to the instruction to make it more complex. Only output the new instruction, nothing else.""",

    EvolutionType.DEEPEN: """I want you to deepen the given instruction by asking for more depth or details. The new instruction should be reasonable and can be understood and responded to by humans.

Original Instruction: {instruction}

Please deepen the instruction by requiring more in-depth analysis or explanation. Only output the new instruction, nothing else.""",

    EvolutionType.CONCRETIZE: """I want you to make the given instruction more concrete and specific. Replace general concepts with specific examples or scenarios.

Original Instruction: {instruction}

Please make the instruction more concrete and specific. Only output the new instruction, nothing else.""",

    EvolutionType.INCREASE_REASONING: """I want you to rewrite the instruction to require more step-by-step reasoning or logical thinking.

Original Instruction: {instruction}

Please rewrite the instruction to require more reasoning and logical steps. Only output the new instruction, nothing else.""",

    EvolutionType.COMPLICATE_INPUT: """I want you to complicate the input of the given instruction by adding more context, data, or conditions that need to be considered.

Original Instruction: {instruction}

Please complicate the input while keeping the core task similar. Only output the new instruction, nothing else.""",
}


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation.

    Attributes:
        model_name: Name/path of the teacher model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum tokens to generate
        num_return_sequences: Number of sequences per prompt
        do_sample: Whether to use sampling
        device: Device for model inference
        batch_size: Batch size for generation
        use_openai: Whether to use OpenAI API instead of local model
        openai_model: OpenAI model to use
    """

    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    num_return_sequences: int = 1
    do_sample: bool = True
    device: str | None = None  # Auto-detect
    batch_size: int = 8
    use_openai: bool = False
    openai_model: str = "gpt-3.5-turbo"


@dataclass
class SelfInstructConfig:
    """Configuration for Self-Instruct generation.

    Attributes:
        num_instructions: Number of instructions to generate
        num_seed_tasks: Number of seed tasks to use as examples
        num_examples_per_prompt: Number of examples in each generation prompt
        include_response: Whether to also generate responses
        diversity_threshold: Minimum ROUGE-L distance from existing instructions
        max_retries: Maximum retries for failed generations
        filter_similar: Whether to filter similar instructions
    """

    num_instructions: int = 1000
    num_seed_tasks: int = 175
    num_examples_per_prompt: int = 3
    include_response: bool = True
    diversity_threshold: float = 0.7
    max_retries: int = 3
    filter_similar: bool = True


@dataclass
class EvolInstructConfig:
    """Configuration for Evol-Instruct generation.

    Attributes:
        num_evolutions: Number of evolution iterations
        evolution_types: Types of evolution to apply
        max_depth: Maximum evolution depth per instruction
        filter_failed: Whether to filter failed evolutions
        response_model: Model for generating responses (can differ from evolution model)
        include_original: Whether to keep original instructions in output
    """

    num_evolutions: int = 3
    evolution_types: list[EvolutionType] = field(
        default_factory=lambda: list(EvolutionType)
    )
    max_depth: int = 3
    filter_failed: bool = True
    response_model: str | None = None
    include_original: bool = True


# Default seed tasks for Self-Instruct
DEFAULT_SEED_TASKS = [
    {
        "instruction": "Write a poem about the changing seasons.",
        "input": "",
        "output": "Spring awakens the sleeping earth,\nFlowers bloom with vibrant mirth.\nSummer brings the golden sun,\nEndless days of outdoor fun.\n\nAutumn paints the leaves in gold,\nStories of harvest to be told.\nWinter wraps the world in white,\nSilent snow on peaceful night."
    },
    {
        "instruction": "Explain the concept of machine learning to a beginner.",
        "input": "",
        "output": "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. Instead of writing rules for every possible scenario, we give the computer examples and let it figure out the patterns. For example, to teach a computer to recognize cats in photos, we show it thousands of cat pictures until it learns what makes a cat a cat."
    },
    {
        "instruction": "Convert the following temperature from Celsius to Fahrenheit.",
        "input": "25 degrees Celsius",
        "output": "25 degrees Celsius is equal to 77 degrees Fahrenheit. The formula is: F = (C × 9/5) + 32 = (25 × 9/5) + 32 = 45 + 32 = 77°F"
    },
]


class BaseSyntheticGenerator(ABC):
    """Abstract base class for synthetic data generators."""

    def __init__(self, generation_config: GenerationConfig | None = None) -> None:
        self.config = generation_config or GenerationConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._openai_client = None

        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate(self, num_samples: int) -> Dataset:
        """Generate synthetic data samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            HuggingFace Dataset with generated samples
        """
        pass

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of generated responses
        """
        if self.config.use_openai:
            return self._generate_openai_batch(prompts)
        else:
            return self._generate_local_batch(prompts)

    def _generate_local_batch(self, prompts: list[str]) -> list[str]:
        """Generate using local model."""
        if self._pipeline is None:
            self.load_model()

        results = []
        for prompt in prompts:
            output = self._pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            # Extract generated text
            generated = output[0]["generated_text"]
            # Remove the prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            results.append(generated)

        return results

    def _generate_openai_batch(self, prompts: list[str]) -> list[str]:
        """Generate using OpenAI API."""
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI generation")

        if self._openai_client is None:
            self._openai_client = openai.OpenAI()

        results = []
        for prompt in prompts:
            response = self._openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
            )
            results.append(response.choices[0].message.content)

        return results

    def load_model(self) -> None:
        """Load the teacher model for generation."""
        if self.config.use_openai:
            if not _OPENAI_AVAILABLE:
                raise ImportError("openai package is required for OpenAI generation")
            self._openai_client = openai.OpenAI()
        else:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required for local model generation")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )


class SelfInstructGenerator(BaseSyntheticGenerator):
    """Generate instructions using the Self-Instruct methodology.

    Self-Instruct generates new instructions by prompting an LLM with
    seed task examples and asking it to create similar but novel tasks.

    Reference: Wang et al., "Self-Instruct: Aligning Language Models with
    Self-Generated Instructions"

    Args:
        generation_config: Configuration for generation
        self_instruct_config: Configuration for Self-Instruct
        seed_tasks: List of seed task dictionaries
    """

    INSTRUCTION_PROMPT = """You are an AI assistant that generates diverse task instructions. Given the examples below, generate a new, unique instruction that is different from the examples.

Examples:
{examples}

Generate a new instruction that is:
1. Clear and unambiguous
2. Solvable by a language model
3. Different from the examples above

Output format (use exactly this format):
### Instruction:
[Your new instruction here]

### Input:
[Optional input data, or leave empty if none needed]

### Output:
[Expected output or response]

Now generate a new task:"""

    def __init__(
        self,
        generation_config: GenerationConfig | None = None,
        self_instruct_config: SelfInstructConfig | None = None,
        seed_tasks: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialize the Self-Instruct generator.

        Args:
            generation_config: Generation parameters
            self_instruct_config: Self-Instruct parameters
            seed_tasks: Seed task examples
        """
        super().__init__(generation_config)
        self.si_config = self_instruct_config or SelfInstructConfig()
        self.seed_tasks = seed_tasks or DEFAULT_SEED_TASKS
        self._generated_instructions: list[str] = []

        # ROUGE scorer for diversity checking
        self._rouge_scorer = None
        if _ROUGE_AVAILABLE:
            self._rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def generate(self, num_samples: int) -> Dataset:
        """Generate synthetic instructions using Self-Instruct.

        Args:
            num_samples: Number of instructions to generate

        Returns:
            Dataset with generated instructions and responses
        """
        generated_data = []
        self._generated_instructions = []

        pbar = tqdm(total=num_samples, desc="Generating instructions")

        while len(generated_data) < num_samples:
            task = self.generate_instruction()

            if task is not None:
                instruction = task.get("instruction", "")

                # Check diversity
                if self.si_config.filter_similar:
                    if not self.is_diverse(instruction, self._generated_instructions):
                        continue

                self._generated_instructions.append(instruction)

                # Generate response if needed
                if self.si_config.include_response and not task.get("output"):
                    input_text = task.get("input", "")
                    task["output"] = self.generate_response(instruction, input_text)

                generated_data.append(task)
                pbar.update(1)

        pbar.close()

        # Convert to dataset
        return Dataset.from_list(generated_data)

    def generate_instruction(self) -> dict[str, str] | None:
        """Generate a single instruction.

        Returns:
            Dictionary with instruction and optional input/output,
            or None if generation failed
        """
        # Sample examples for the prompt
        examples = random.sample(
            self.seed_tasks + [{"instruction": inst, "input": "", "output": ""}
                               for inst in self._generated_instructions[-20:]],
            min(self.si_config.num_examples_per_prompt, len(self.seed_tasks))
        )

        prompt = self.build_generation_prompt(examples)

        for _ in range(self.si_config.max_retries):
            try:
                responses = self.generate_batch([prompt])
                if responses:
                    parsed = self.parse_generated_instruction(responses[0])
                    if parsed:
                        return parsed
            except Exception:
                continue

        return None

    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """Generate a response for an instruction.

        Args:
            instruction: The instruction
            input_text: Optional input context

        Returns:
            Generated response
        """
        if input_text:
            prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\n\nResponse:"

        responses = self.generate_batch([prompt])
        return responses[0] if responses else ""

    def build_generation_prompt(self, examples: list[dict[str, str]]) -> str:
        """Build the prompt for instruction generation.

        Args:
            examples: List of example tasks

        Returns:
            Formatted prompt string
        """
        example_strs = []
        for i, ex in enumerate(examples, 1):
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            output = ex.get("output", "")

            example_str = f"Example {i}:\n### Instruction:\n{instruction}"
            if input_text:
                example_str += f"\n\n### Input:\n{input_text}"
            if output:
                example_str += f"\n\n### Output:\n{output}"
            example_strs.append(example_str)

        examples_text = "\n\n".join(example_strs)
        return self.INSTRUCTION_PROMPT.format(examples=examples_text)

    def parse_generated_instruction(
        self, generated_text: str
    ) -> dict[str, str] | None:
        """Parse a generated instruction from model output.

        Args:
            generated_text: Raw model output

        Returns:
            Parsed instruction dictionary or None if parsing failed
        """
        try:
            result = {"instruction": "", "input": "", "output": ""}

            # Try to find instruction section
            instruction_match = re.search(
                r'###\s*Instruction:?\s*\n(.*?)(?=###|\Z)',
                generated_text,
                re.DOTALL | re.IGNORECASE
            )
            if instruction_match:
                result["instruction"] = instruction_match.group(1).strip()

            # Try to find input section
            input_match = re.search(
                r'###\s*Input:?\s*\n(.*?)(?=###|\Z)',
                generated_text,
                re.DOTALL | re.IGNORECASE
            )
            if input_match:
                result["input"] = input_match.group(1).strip()

            # Try to find output section
            output_match = re.search(
                r'###\s*Output:?\s*\n(.*?)(?=###|\Z)',
                generated_text,
                re.DOTALL | re.IGNORECASE
            )
            if output_match:
                result["output"] = output_match.group(1).strip()

            # Validate - must have instruction
            if not result["instruction"]:
                return None

            # Clean up empty strings
            if result["input"] in ["", "N/A", "None", "none"]:
                result["input"] = ""

            return result

        except Exception:
            return None

    def is_diverse(
        self, new_instruction: str, existing_instructions: list[str]
    ) -> bool:
        """Check if instruction is sufficiently diverse from existing ones.

        Args:
            new_instruction: New instruction to check
            existing_instructions: List of existing instructions

        Returns:
            True if instruction is diverse enough
        """
        if not existing_instructions:
            return True

        if self._rouge_scorer is None:
            # Fallback to simple substring check
            new_lower = new_instruction.lower()
            for existing in existing_instructions[-100:]:  # Check recent ones
                if new_lower in existing.lower() or existing.lower() in new_lower:
                    return False
            return True

        # Check ROUGE-L similarity against recent instructions
        for existing in existing_instructions[-100:]:
            scores = self._rouge_scorer.score(new_instruction, existing)
            if scores['rougeL'].fmeasure > self.si_config.diversity_threshold:
                return False

        return True


class EvolInstructGenerator(BaseSyntheticGenerator):
    """Generate complex instructions using Evol-Instruct methodology.

    Evol-Instruct takes simple instructions and iteratively evolves them
    to create more complex, challenging versions through various evolution
    operations.

    Reference: Xu et al., "WizardLM: Empowering Large Language Models to
    Follow Complex Instructions"

    Args:
        generation_config: Configuration for generation
        evol_config: Configuration for Evol-Instruct
        base_instructions: Initial instructions to evolve
    """

    def __init__(
        self,
        generation_config: GenerationConfig | None = None,
        evol_config: EvolInstructConfig | None = None,
        base_instructions: list[str] | None = None,
    ) -> None:
        """Initialize the Evol-Instruct generator.

        Args:
            generation_config: Generation parameters
            evol_config: Evol-Instruct parameters
            base_instructions: Base instructions to evolve
        """
        super().__init__(generation_config)
        self.evol_config = evol_config or EvolInstructConfig()
        self.base_instructions = base_instructions or []

    def generate(self, num_samples: int) -> Dataset:
        """Generate evolved instructions.

        Args:
            num_samples: Number of evolved instructions to generate

        Returns:
            Dataset with evolved instructions and responses
        """
        if not self.base_instructions:
            raise ValueError("base_instructions must be provided for Evol-Instruct")

        generated_data = []

        # Include original instructions if configured
        if self.evol_config.include_original:
            for instruction in self.base_instructions:
                generated_data.append({
                    "instruction": instruction,
                    "evolution_depth": 0,
                    "evolution_type": "original",
                })

        # Generate evolutions
        for instruction in tqdm(self.base_instructions, desc="Evolving instructions"):
            evolution_types = random.sample(
                self.evol_config.evolution_types,
                min(self.evol_config.num_evolutions, len(self.evol_config.evolution_types))
            )

            evolved_chain = self.evolve_chain(instruction, evolution_types)

            for depth, evolved in enumerate(evolved_chain, 1):
                generated_data.append({
                    "instruction": evolved,
                    "evolution_depth": depth,
                    "evolution_type": evolution_types[depth - 1].value if depth <= len(evolution_types) else "chain",
                    "original_instruction": instruction,
                })

            # Stop if we have enough
            if len(generated_data) >= num_samples:
                break

        # Trim to exact number
        generated_data = generated_data[:num_samples]

        # Generate responses for all
        for item in tqdm(generated_data, desc="Generating responses"):
            if "output" not in item:
                responses = self.generate_batch([f"Complete this task:\n\n{item['instruction']}"])
                item["output"] = responses[0] if responses else ""

        return Dataset.from_list(generated_data)

    def evolve_instruction(
        self,
        instruction: str,
        evolution_type: EvolutionType,
    ) -> str | None:
        """Apply a single evolution to an instruction.

        Args:
            instruction: Original instruction
            evolution_type: Type of evolution to apply

        Returns:
            Evolved instruction or None if evolution failed
        """
        prompt = self.get_evolution_prompt(instruction, evolution_type)

        try:
            responses = self.generate_batch([prompt])
            if responses:
                evolved = responses[0].strip()

                # Validate evolution
                if self.validate_evolution(instruction, evolved):
                    return evolved

        except Exception:
            pass

        return None

    def evolve_chain(
        self,
        instruction: str,
        evolution_types: list[EvolutionType],
    ) -> list[str]:
        """Apply a chain of evolutions to an instruction.

        Args:
            instruction: Original instruction
            evolution_types: List of evolution types to apply

        Returns:
            List of evolved instructions at each step
        """
        evolved_instructions = []
        current = instruction

        for evo_type in evolution_types[:self.evol_config.max_depth]:
            evolved = self.evolve_instruction(current, evo_type)

            if evolved is None:
                if self.evol_config.filter_failed:
                    break
                else:
                    evolved_instructions.append(current)
                    continue

            evolved_instructions.append(evolved)
            current = evolved

        return evolved_instructions

    def get_evolution_prompt(
        self,
        instruction: str,
        evolution_type: EvolutionType,
    ) -> str:
        """Get the prompt for a specific evolution type.

        Args:
            instruction: Instruction to evolve
            evolution_type: Type of evolution

        Returns:
            Evolution prompt
        """
        template = EVOLUTION_PROMPTS.get(evolution_type)
        if template is None:
            template = EVOLUTION_PROMPTS[EvolutionType.ADD_CONSTRAINTS]

        return template.format(instruction=instruction)

    def validate_evolution(
        self,
        original: str,
        evolved: str,
    ) -> bool:
        """Validate that an evolution was successful.

        Args:
            original: Original instruction
            evolved: Evolved instruction

        Returns:
            True if evolution is valid
        """
        # Basic validations
        if not evolved or len(evolved) < 10:
            return False

        # Evolved should be different
        if evolved.lower().strip() == original.lower().strip():
            return False

        # Evolved should be longer (more complex)
        if len(evolved) < len(original) * 0.5:
            return False

        # Should not contain obvious failure patterns
        failure_patterns = [
            "i cannot",
            "i'm sorry",
            "as an ai",
            "i don't understand",
        ]
        evolved_lower = evolved.lower()
        for pattern in failure_patterns:
            if pattern in evolved_lower:
                return False

        return True


class SyntheticDataGenerator:
    """Unified interface for synthetic data generation.

    Combines Self-Instruct and Evol-Instruct methods to generate
    diverse, high-quality synthetic training data.

    Args:
        generation_config: Configuration for generation
        self_instruct_config: Optional Self-Instruct configuration
        evol_config: Optional Evol-Instruct configuration
    """

    def __init__(
        self,
        generation_config: GenerationConfig | None = None,
        self_instruct_config: SelfInstructConfig | None = None,
        evol_config: EvolInstructConfig | None = None,
    ) -> None:
        """Initialize the synthetic data generator.

        Args:
            generation_config: Generation parameters
            self_instruct_config: Self-Instruct parameters
            evol_config: Evol-Instruct parameters
        """
        self.generation_config = generation_config or GenerationConfig()
        self.si_config = self_instruct_config or SelfInstructConfig()
        self.evol_config = evol_config or EvolInstructConfig()

        self._self_instruct_generator: SelfInstructGenerator | None = None
        self._evol_generator: EvolInstructGenerator | None = None

    def generate_self_instruct(
        self,
        num_samples: int,
        seed_tasks: list[dict[str, str]] | None = None,
    ) -> Dataset:
        """Generate data using Self-Instruct.

        Args:
            num_samples: Number of samples to generate
            seed_tasks: Optional seed tasks

        Returns:
            Dataset with generated samples
        """
        if self._self_instruct_generator is None:
            self._self_instruct_generator = SelfInstructGenerator(
                generation_config=self.generation_config,
                self_instruct_config=self.si_config,
                seed_tasks=seed_tasks,
            )

        return self._self_instruct_generator.generate(num_samples)

    def generate_evol_instruct(
        self,
        base_instructions: list[str],
        num_evolutions: int = 3,
    ) -> Dataset:
        """Generate data using Evol-Instruct.

        Args:
            base_instructions: Base instructions to evolve
            num_evolutions: Number of evolution iterations

        Returns:
            Dataset with evolved samples
        """
        evol_config = EvolInstructConfig(
            num_evolutions=num_evolutions,
            evolution_types=self.evol_config.evolution_types,
            max_depth=self.evol_config.max_depth,
            filter_failed=self.evol_config.filter_failed,
            include_original=self.evol_config.include_original,
        )

        self._evol_generator = EvolInstructGenerator(
            generation_config=self.generation_config,
            evol_config=evol_config,
            base_instructions=base_instructions,
        )

        # Estimate total samples
        total = len(base_instructions) * (num_evolutions + (1 if evol_config.include_original else 0))
        return self._evol_generator.generate(total)

    def generate_combined(
        self,
        num_self_instruct: int,
        num_evolutions: int,
        seed_tasks: list[dict[str, str]] | None = None,
    ) -> Dataset:
        """Generate data using both methods combined.

        First generates base instructions with Self-Instruct,
        then evolves them with Evol-Instruct.

        Args:
            num_self_instruct: Number of Self-Instruct samples
            num_evolutions: Evolution iterations per instruction
            seed_tasks: Optional seed tasks

        Returns:
            Combined dataset
        """
        # Generate base instructions with Self-Instruct
        si_dataset = self.generate_self_instruct(num_self_instruct, seed_tasks)

        # Extract instructions for evolution
        base_instructions = [item["instruction"] for item in si_dataset]

        # Evolve instructions
        evol_dataset = self.generate_evol_instruct(base_instructions, num_evolutions)

        # Combine datasets
        combined_data = []

        # Add Self-Instruct data
        for item in si_dataset:
            combined_data.append({
                **item,
                "source": "self_instruct",
            })

        # Add Evol-Instruct data (excluding originals since they're duplicates)
        for item in evol_dataset:
            if item.get("evolution_depth", 0) > 0:
                combined_data.append({
                    **item,
                    "source": "evol_instruct",
                })

        return Dataset.from_list(combined_data)

    def filter_quality(self, dataset: Dataset) -> Dataset:
        """Filter generated data for quality.

        Args:
            dataset: Dataset to filter

        Returns:
            Filtered dataset
        """
        def is_quality(example: dict) -> bool:
            instruction = example.get("instruction", "")
            output = example.get("output", "")

            # Basic length checks
            if len(instruction) < 10:
                return False
            if output and len(output) < 10:
                return False

            # Check for obvious failures
            failure_patterns = [
                "i cannot",
                "i'm sorry",
                "as an ai",
                "i don't have",
            ]
            text = (instruction + " " + output).lower()
            for pattern in failure_patterns:
                if pattern in text:
                    return False

            return True

        return dataset.filter(is_quality)

    def save_dataset(
        self,
        dataset: Dataset,
        output_path: str,
        format: str = "json",
    ) -> None:
        """Save generated dataset to disk.

        Args:
            dataset: Dataset to save
            output_path: Output file path
            format: Output format (json, parquet, arrow)
        """
        if format == "json":
            dataset.to_json(output_path)
        elif format == "parquet":
            dataset.to_parquet(output_path)
        elif format == "arrow":
            dataset.save_to_disk(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


def generate_from_seeds(
    seeds: list[dict[str, str]],
    num_samples: int,
    model_name: str = "gpt-3.5-turbo",
    use_openai: bool = True,
) -> Dataset:
    """Convenience function to generate data from seed tasks.

    Args:
        seeds: List of seed task dictionaries
        num_samples: Number of samples to generate
        model_name: Model to use for generation
        use_openai: Whether to use OpenAI API

    Returns:
        Generated dataset
    """
    config = GenerationConfig(
        model_name=model_name,
        use_openai=use_openai,
        openai_model=model_name if use_openai else "gpt-3.5-turbo",
    )

    generator = SelfInstructGenerator(
        generation_config=config,
        seed_tasks=seeds,
    )

    return generator.generate(num_samples)


def evolve_complexity(
    instructions: list[str],
    depth: int = 3,
    model_name: str = "gpt-3.5-turbo",
    use_openai: bool = True,
) -> Dataset:
    """Convenience function to evolve instruction complexity.

    Args:
        instructions: Base instructions to evolve
        depth: Maximum evolution depth
        model_name: Model to use for evolution
        use_openai: Whether to use OpenAI API

    Returns:
        Dataset with evolved instructions
    """
    config = GenerationConfig(
        model_name=model_name,
        use_openai=use_openai,
        openai_model=model_name if use_openai else "gpt-3.5-turbo",
    )

    evol_config = EvolInstructConfig(
        num_evolutions=depth,
        max_depth=depth,
    )

    generator = EvolInstructGenerator(
        generation_config=config,
        evol_config=evol_config,
        base_instructions=instructions,
    )

    total = len(instructions) * (depth + 1)
    return generator.generate(total)
