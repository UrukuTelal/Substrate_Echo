"""Agent View — text-based visualization of cognitive agent activity."""

from __future__ import annotations
from typing import Optional
import numpy as np


class AgentView:
    """Text-based renderer for cognitive agent ecology."""
    
    def render_agent_states(self, ecology, title: str = "Agent Ecology") -> str:
        """Render all agent states."""
        lines = [f"=== {title} ==="]
        
        for role, agent in ecology.agents.items():
            status = "ACTIVE" if agent.is_active else "idle"
            count = agent.activation_count
            threshold = agent.activation_threshold
            
            lines.append(
                f"  {role.name:>12s} | {status:>6s} "
                f"| activations={count:>3d} threshold={threshold:.3f}"
            )
        
        # Energy
        energy = ecology.energy_pool
        max_e = ecology.max_energy
        bar_w = 20
        filled = int((energy / max_e) * bar_w) if max_e > 0 else 0
        bar = "█" * filled + "░" * (bar_w - filled)
        lines.append(f"  {'Energy':>12s} | [{bar}] {energy:.3f}/{max_e:.3f}")
        
        return "\n".join(lines)
    
    def render_responses(self, responses: list, title: str = "Agent Responses") -> str:
        """Render agent responses."""
        lines = [f"=== {title} ({len(responses)} responses) ==="]
        
        for resp in responses:
            lines.append(
                f"  {resp.agent_role.name:>12s} | conf={resp.confidence:.3f} "
                f"| {resp.reasoning[:60]}"
            )
        
        if not responses:
            lines.append("  (no responses)")
        
        return "\n".join(lines)
    
    def render_consensus(self, consensus, title: str = "Consensus") -> str:
        """Render the consensus decision."""
        if consensus is None:
            return f"=== {title}: no consensus ==="
        
        lines = [
            f"=== {title} ===",
            f"  Agent: {consensus.agent_role.name}",
            f"  Confidence: {consensus.confidence:.3f}",
            f"  Reasoning: {consensus.reasoning}",
        ]
        
        if consensus.proposed_action:
            lines.append(f"  Action: {consensus.proposed_action}")
        
        return "\n".join(lines)
    
    def render_message_log(self, messages: list, n: int = 10) -> str:
        """Render recent inter-agent messages."""
        lines = [f"=== Recent Messages (last {min(n, len(messages))}) ==="]
        
        for msg in messages[-n:]:
            sender = msg.sender.name if hasattr(msg.sender, 'name') else str(msg.sender)
            receiver = msg.receiver.name if msg.receiver and hasattr(msg.receiver, 'name') else "broadcast"
            lines.append(
                f"  {sender} -> {receiver} | {msg.message_type.name} "
                f"| conf={msg.confidence:.2f}"
            )
        
        if not messages:
            lines.append("  (no messages)")
        
        return "\n".join(lines)
