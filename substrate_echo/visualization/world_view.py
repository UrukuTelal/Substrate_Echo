"""World View — text-based visualization of spatial world model."""

from __future__ import annotations
from typing import Optional
import numpy as np


class WorldView:
    """Text-based renderer for spatial world model."""
    
    def render_objects(self, objects: dict, title: str = "World Objects") -> str:
        """Render all objects in the world model."""
        lines = [f"=== {title} ({len(objects)} objects) ==="]
        
        for obj_id, obj in objects.items():
            pos = getattr(obj, 'physical', None)
            pos_str = f"({pos.position[0]:.1f}, {pos.position[1]:.1f}, {pos.position[2]:.1f})" if pos else "(?)"
            
            rel = getattr(obj, 'relational', None)
            fam = rel.familiarity if rel else 0.0
            imp = rel.importance if rel else 0.0
            
            lines.append(
                f"  [{obj_id}] {obj.name} @ {pos_str} "
                f"| fam={fam:.2f} imp={imp:.2f}"
            )
        
        if not objects:
            lines.append("  (empty)")
        
        return "\n".join(lines)
    
    def render_region(self, objects: list, center: tuple, radius: float) -> str:
        """Render objects in a spatial region."""
        lines = [f"=== Region @ ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) r={radius:.1f} ==="]
        
        for obj in objects:
            pos = getattr(obj, 'physical', None)
            pos_str = f"({pos.position[0]:.1f}, {pos.position[1]:.1f}, {pos.position[2]:.1f})" if pos else "(?)"
            lines.append(f"  {obj.name} @ {pos_str}")
        
        if not objects:
            lines.append("  (empty)")
        
        return "\n".join(lines)
    
    def render_graph_stats(self, graph) -> str:
        """Render spatial graph statistics."""
        n_objects = len(graph.objects) if hasattr(graph, 'objects') else 0
        n_edges = len(graph.edges) if hasattr(graph, 'edges') else 0
        
        return (
            f"=== Spatial Graph ===\n"
            f"  Objects: {n_objects}\n"
            f"  Edges: {n_edges}"
        )
