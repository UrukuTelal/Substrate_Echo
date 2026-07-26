"""Councils package - Specialized decision-making bodies."""

from substrate_echo.councils.base import BaseCouncil, CouncilRegistry, CouncilState, CouncilMetrics
from substrate_echo.councils.capability import CapabilityCouncil
from substrate_echo.councils.economy import EconomyCouncil
from substrate_echo.councils.reconnaissance import ReconnaissanceCouncil
from substrate_echo.councils.counter_intelligence import CounterIntelligenceCouncil
from substrate_echo.councils.military_industrial import MilitaryIndustrialCouncil
from substrate_echo.councils.technology import TechnologyCouncil
from substrate_echo.councils.strategy import StrategyCouncil
from substrate_echo.councils.logistics import LogisticsCouncil
from substrate_echo.councils.allocator import ResourceAllocator, AllocationStrategy, AllocationResult

__all__ = [
    "BaseCouncil",
    "CouncilRegistry",
    "CouncilState",
    "CouncilMetrics",
    "CapabilityCouncil",
    "EconomyCouncil",
    "ReconnaissanceCouncil",
    "CounterIntelligenceCouncil",
    "MilitaryIndustrialCouncil",
    "TechnologyCouncil",
    "StrategyCouncil",
    "LogisticsCouncil",
    "ResourceAllocator",
    "AllocationStrategy",
    "AllocationResult",
]
