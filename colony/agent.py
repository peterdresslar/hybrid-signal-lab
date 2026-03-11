"""Agent interface and implementations.

An agent wraps an LLM and exposes:
  - generate(): run inference, return a response + signal vector
  - signal: the agent's current signal state (persisted in the buffer)
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A signal emitted by an agent after inference.

    Attributes:
        agent_id: identifier for the agent that produced this signal.
        vector: low-dimensional signal vector (the "pheromone").
        entropy: Shannon entropy of the output token distribution.
        thinking_length: number of characters in the thinking trace (0 if disabled).
        timestamp: when the signal was produced (epoch seconds).
    """

    agent_id: str
    vector: np.ndarray
    entropy: float
    thinking_length: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """The result of a single agent inference step.

    Attributes:
        text: the generated text output.
        thinking: the model's reasoning trace (empty string if disabled).
        signal: the signal extracted from model internals.
    """

    text: str
    thinking: str
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
    """Agent backed by a local Ollama model instance. Implements the abstract base class Agent.

    Requires ollama to be running locally with the specified model pulled.
    Extracts real entropy from token logprobs and thinking traces when available.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "qwen3.5:2b",
        temperature: float = 0.7,
        signal_dim: int = 16,
        think: bool = True,
    ):
        super().__init__(agent_id)
        self.model = model
        self.temperature = temperature
        self.signal_dim = signal_dim
        self.think = think

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
        """Run inference via Ollama and extract signals.

        Requests thinking trace and token logprobs for real signal extraction.
        """
        # build generation kwargs
        gen_kwargs: dict = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
        }

        # enable thinking if supported
        if self.think:
            gen_kwargs["think"] = True

        response = self._client.generate(**gen_kwargs)

        text = response.get("response", "")
        thinking = response.get("thinking", "")

        if thinking:
            logger.debug(
                "Agent %s thinking: %d chars", self.agent_id, len(thinking)
            )

        # extract signal from logprobs, thinking, and metadata
        signal_vector = self._extract_signal(response)
        entropy = self._compute_entropy(response)

        signal = Signal(
            agent_id=self.agent_id,
            vector=signal_vector,
            entropy=entropy,
            thinking_length=len(thinking),
        )

        return AgentResponse(text=text, thinking=thinking, signal=signal)

    def _extract_signal(self, response: dict) -> np.ndarray:
        """Extract a signal vector from the Ollama response.

        Uses a mix of real measurements and metadata to build the signal:
          [0] entropy (from logprobs if available, else speed proxy)
          [1] output length (token count)
          [2] input complexity (prompt token count)
          [3] generation speed (tokens/sec)
          [4] thinking effort (thinking trace length, normalized)
          [5:] hash-derived dimensions for content fingerprinting
        """
        eval_count = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 1)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        thinking = response.get("thinking", "")

        tokens_per_sec = eval_count / max(eval_duration / 1e9, 1e-6)

        # content-derived dimensions for the remainder of the vector
        rng = np.random.default_rng(
            seed=hash(response.get("response", "")) % (2**32)
        )
        vec = rng.standard_normal(self.signal_dim)

        # overwrite first dimensions with real measurements
        vec[0] = self._compute_entropy(response)
        vec[1] = np.tanh(eval_count / 100.0)           # output length
        vec[2] = np.tanh(prompt_eval_count / 100.0)     # input complexity
        vec[3] = np.tanh(tokens_per_sec / 100.0)        # speed
        vec[4] = np.tanh(len(thinking) / 500.0)         # thinking effort

        return vec

    def _compute_entropy(self, response: dict) -> float:
        """Compute entropy from token logprobs if available.

        Falls back to a speed-based proxy if logprobs aren't present.
        """
        logprobs = response.get("logprobs")

        if logprobs:
            # logprobs is a list of per-token logprob values
            # Shannon entropy: H = -sum(p * log(p))
            # Since we have log(p), we compute: H = -mean(logprob)
            # (this is cross-entropy with the model's own distribution)
            try:
                lps = [lp for lp in logprobs if isinstance(lp, (int, float))]
                if lps:
                    # negative mean logprob = cross-entropy in nats
                    mean_neglogp = -sum(lps) / len(lps)
                    # convert to bits and clamp
                    entropy_bits = mean_neglogp / math.log(2)
                    logger.debug(
                        "Agent %s real entropy: %.3f bits (%d tokens)",
                        response.get("model", "?"),
                        entropy_bits,
                        len(lps),
                    )
                    return float(np.clip(entropy_bits, 0.0, 20.0))
            except (TypeError, ValueError) as e:
                logger.debug("Logprob parsing failed: %s", e)

        # fallback: speed-based proxy
        eval_count = response.get("eval_count", 1)
        eval_duration = response.get("eval_duration", 1)
        tokens_per_sec = eval_count / max(eval_duration / 1e9, 1e-6)
        entropy_proxy = 1.0 / (1.0 + tokens_per_sec / 50.0)
        return float(entropy_proxy)
