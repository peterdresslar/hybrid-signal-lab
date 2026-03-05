"""Persistent signal buffer with time-decay.

The buffer stores signals from all agents in the colony. Signals decay
exponentially over time, so recently-active agents have stronger presence
than dormant ones. This is the core data structure that enables
asynchronous turn-taking: agents not currently loaded still influence
routing through their persisted (but decaying) signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from colony.agent import Signal


@dataclass
class BufferEntry:
    """A single entry in the signal buffer."""

    signal: Signal
    round_number: int  # which round this signal was written


class SignalBuffer:
    """Persistent signal buffer with exponential time-decay.

    The buffer maintains the most recent signal from each agent.
    When read, signals are attenuated by an exponential decay factor
    based on the time elapsed since they were written.

    Attributes:
        decay_rate: decay factor per second. A value of 0.95 means
            the signal retains 95% of its strength each second.
            Lower values = faster decay.
        signals: dict mapping agent_id -> BufferEntry.
    """

    def __init__(self, decay_rate: float = 0.95):
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        self.decay_rate = decay_rate
        self._entries: dict[str, BufferEntry] = {}
        self._round: int = 0

    @property
    def current_round(self) -> int:
        return self._round

    def advance_round(self) -> int:
        """Advance the round counter. Returns the new round number."""
        self._round += 1
        return self._round

    def write(self, signal: Signal) -> None:
        """Write a signal to the buffer, replacing any previous signal
        from the same agent."""
        self._entries[signal.agent_id] = BufferEntry(
            signal=signal, round_number=self._round
        )

    def read(self, exclude: str | None = None) -> dict[str, np.ndarray]:
        """Read all signals from the buffer, applying time-decay.

        Args:
            exclude: optionally exclude a specific agent_id (useful when
                an agent reads the buffer to avoid seeing its own signal).

        Returns:
            Dict mapping agent_id -> decayed signal vector.
        """
        now = time.time()
        result = {}

        for agent_id, entry in self._entries.items():
            if agent_id == exclude:
                continue

            elapsed = now - entry.signal.timestamp
            decay = self.decay_rate**elapsed
            result[agent_id] = entry.signal.vector * decay

        return result

    def read_aggregate(self, exclude: str | None = None) -> np.ndarray | None:
        """Read the mean of all decayed signals in the buffer.

        This is the "collective signal" that an agent or orchestrator
        can use for routing decisions.

        Args:
            exclude: optionally exclude a specific agent_id.

        Returns:
            Mean decayed signal vector, or None if the buffer is empty.
        """
        signals = self.read(exclude=exclude)
        if not signals:
            return None
        return np.mean(list(signals.values()), axis=0)

    def read_with_metadata(
        self, exclude: str | None = None
    ) -> list[dict]:
        """Read all signals with full metadata for measurement/analysis.

        Returns:
            List of dicts with agent_id, vector, decayed_vector, entropy,
            timestamp, age, decay_factor, and round_number.
        """
        now = time.time()
        result = []

        for agent_id, entry in self._entries.items():
            if agent_id == exclude:
                continue

            elapsed = now - entry.signal.timestamp
            decay = self.decay_rate**elapsed

            result.append(
                {
                    "agent_id": agent_id,
                    "vector": entry.signal.vector,
                    "decayed_vector": entry.signal.vector * decay,
                    "entropy": entry.signal.entropy,
                    "timestamp": entry.signal.timestamp,
                    "age": elapsed,
                    "decay_factor": decay,
                    "round_number": entry.round_number,
                }
            )

        return result

    @property
    def agent_ids(self) -> list[str]:
        """List of all agent IDs with signals in the buffer."""
        return list(self._entries.keys())

    @property
    def size(self) -> int:
        """Number of agents with signals in the buffer."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all signals from the buffer."""
        self._entries.clear()
        self._round = 0
