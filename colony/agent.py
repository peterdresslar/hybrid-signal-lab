"""Agent interface and implementations.

An agent wraps an LLM and exposes:
  - generate(): run inference, return a response + signal vector
  - signal: the agent's current signal state (persisted in the buffer)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Signal:
    """A signal emitted by an agent after inference.

    Attributes:
        agent_id: identifier for the agent that produced this signal.
        vector: low-dimensional signal vector (the "pheromone").
        entropy: Shannon entropy of the output token distribution.
        timestamp: when the signal was produced (epoch seconds).
    """

    agent_id: str
    vector: np.ndarray
    entropy: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """The result of a single agent inference step.

    Attributes:
        text: the generated text output.
        signal: the signal extracted from model internals.
    """

    text: str
    signal: Signal


class Agent(ABC):
    """Abstract base for all agent backends."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @abstractmethod
    def generate(self, prompt: str, context: dict | None = None) -> AgentResponse:
        """Run inference and return a response with extracted signals.

        Args:
            prompt: the task prompt or input.
            context: optional dict containing signal buffer state,
                     task metadata, or other orchestration context.

        Returns:
            AgentResponse with text output and signal vector.
        """
        ...


class OllamaAgent(Agent):
    """Agent backed by a local Ollama model instance.

    Requires ollama to be running locally with the specified model pulled.
    Extracts entropy from token logprobs when available.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "qwen3.5:2b",
        temperature: float = 0.7,
        signal_dim: int = 16,
    ):
        super().__init__(agent_id)
        self.model = model
        self.temperature = temperature
        self.signal_dim = signal_dim

        # lazy import so the module loads even without ollama installed
        try:
            import ollama as _ollama

            self._client = _ollama.Client()
        except ImportError:
            raise ImportError(
                "ollama package required for OllamaAgent. "
                "Install with: uv add ollama"
            )

    def generate(self, prompt: str, context: dict | None = None) -> AgentResponse:
        """Run inference via Ollama and extract signals."""
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature},
        )

        text = response.get("response", "")

        # extract what signal information we can from the response
        signal_vector = self._extract_signal(response)
        entropy = self._estimate_entropy(response)

        signal = Signal(
            agent_id=self.agent_id,
            vector=signal_vector,
            entropy=entropy,
        )

        return AgentResponse(text=text, signal=signal)

    def _extract_signal(self, response: dict) -> np.ndarray:
        """Extract a signal vector from the Ollama response.

        For now, we hash the response context to produce a deterministic
        low-dimensional vector. This is a placeholder—when we move to
        transformers/vllm with hidden state access, this becomes real
        activation extraction.
        """
        # use the eval_count and eval_duration as crude proxy features
        eval_count = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 1)
        prompt_eval_count = response.get("prompt_eval_count", 0)

        # generate a pseudo-signal from available metadata
        rng = np.random.default_rng(
            seed=hash(response.get("response", "")) % (2**32)
        )
        base = rng.standard_normal(self.signal_dim)

        # modulate by available metrics
        tokens_per_sec = eval_count / max(eval_duration / 1e9, 1e-6)
        base[0] = np.tanh(tokens_per_sec / 100.0)  # speed signal
        base[1] = np.tanh(eval_count / 100.0)  # output length signal
        base[2] = np.tanh(prompt_eval_count / 100.0)  # input complexity signal

        return base

    def _estimate_entropy(self, response: dict) -> float:
        """Estimate output entropy from available response metadata.

        Ollama doesn't expose per-token logprobs in all configurations.
        This is a rough proxy based on generation speed and output length.
        A proper implementation with transformers/vllm would compute
        actual Shannon entropy from the token distribution.
        """
        eval_count = response.get("eval_count", 1)
        eval_duration = response.get("eval_duration", 1)

        # faster generation often correlates with lower entropy
        # (more peaked distributions = faster sampling)
        tokens_per_sec = eval_count / max(eval_duration / 1e9, 1e-6)

        # rough heuristic: normalize to [0, 1] range
        # high speed -> low entropy, low speed -> high entropy
        entropy_proxy = 1.0 / (1.0 + tokens_per_sec / 50.0)

        return float(entropy_proxy)
