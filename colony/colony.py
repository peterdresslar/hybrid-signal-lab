"""The Colony: asynchronous turn-taking loop for multi-agent orchestration.

The Colony manages N agents through batched turn-taking. At each round,
a batch of K agents is selected, each reads the signal buffer, performs
inference, and writes its updated signal back. Agents not in the current
batch persist in the buffer through their decaying signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from colony.agent import Agent, AgentResponse
from colony.buffer import SignalBuffer

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single turn-taking round.

    Attributes:
        round_number: which round this was.
        batch: list of agent_ids that participated.
        responses: dict mapping agent_id -> AgentResponse.
    """

    round_number: int
    batch: list[str]
    responses: dict[str, AgentResponse] = field(default_factory=dict)


class Colony:
    """Asynchronous turn-taking colony of LLM agents.

    The colony maintains a pool of agents and a shared signal buffer.
    On each round, a batch of agents is activated: they read the
    collective signal, perform inference, and update the buffer.

    Args:
        agents: list of Agent instances forming the colony.
        buffer: the shared SignalBuffer (created with desired decay_rate).
        batch_size: number of agents to activate per round (K).
    """

    def __init__(
        self,
        agents: list[Agent],
        buffer: SignalBuffer | None = None,
        batch_size: int = 5,
    ):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.buffer = buffer or SignalBuffer()
        self.batch_size = min(batch_size, len(agents))
        self._history: list[RoundResult] = []
        self._batch_cursor: int = 0

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def history(self) -> list[RoundResult]:
        return self._history

    def _select_batch(self) -> list[str]:
        """Select the next batch of agents for turn-taking.

        Uses round-robin ordering so all agents get equal turns.
        More sophisticated selection (e.g., signal-weighted) can be
        added as an orchestration mode.
        """
        agent_ids = list(self.agents.keys())
        n = len(agent_ids)

        start = self._batch_cursor % n
        batch_ids = []
        for i in range(self.batch_size):
            idx = (start + i) % n
            batch_ids.append(agent_ids[idx])

        self._batch_cursor = (start + self.batch_size) % n
        return batch_ids

    def run_round(self, prompt: str) -> RoundResult:
        """Execute one round of turn-taking.

        1. Select a batch of K agents.
        2. Each agent reads the signal buffer and generates a response.
        3. Each agent's signal is written to the buffer.

        Args:
            prompt: the task prompt for this round.

        Returns:
            RoundResult with responses from all agents in the batch.
        """
        round_number = self.buffer.advance_round()
        batch_ids = self._select_batch()

        logger.info(
            f"Round {round_number}: activating batch {batch_ids}"
        )

        result = RoundResult(round_number=round_number, batch=batch_ids)

        for agent_id in batch_ids:
            agent = self.agents[agent_id]

            # read the collective signal (excluding self to avoid feedback)
            collective_signal = self.buffer.read_aggregate(exclude=agent_id)

            # build context for the agent
            context = {
                "round": round_number,
                "collective_signal": collective_signal,
                "buffer_size": self.buffer.size,
            }

            logger.debug(f"  Agent {agent_id} generating...")
            response = agent.generate(prompt=prompt, context=context)

            # write the agent's signal to the buffer
            self.buffer.write(response.signal)
            result.responses[agent_id] = response

            logger.debug(
                f"  Agent {agent_id} done. "
                f"Entropy: {response.signal.entropy:.3f}"
            )

        self._history.append(result)
        return result

    def run(self, prompt: str, rounds: int = 1) -> list[RoundResult]:
        """Run multiple rounds of turn-taking.

        Args:
            prompt: the task prompt.
            rounds: number of rounds to execute. With N agents and
                batch_size K, ceil(N/K) rounds gives every agent
                one turn.

        Returns:
            List of RoundResults.
        """
        results = []
        for _ in range(rounds):
            result = self.run_round(prompt)
            results.append(result)
        return results

    def full_cycle(self, prompt: str) -> list[RoundResult]:
        """Run enough rounds for every agent to participate once.

        This is ceil(N / K) rounds.
        """
        import math

        rounds_needed = math.ceil(self.n_agents / self.batch_size)
        return self.run(prompt, rounds=rounds_needed)
