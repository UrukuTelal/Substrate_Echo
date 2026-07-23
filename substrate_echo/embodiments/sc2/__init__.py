"""SC2 Embodiment — StarCraft II as a cognitive substrate.

The SC2 adapter translates between:
- Abstract kernel intent → game actions
- Game state observations → kernel observations

Architecture:
    Substrate Kernel (cognitive)
           ↓
      Abstract Intent
      "Secure expansion"
           ↓
      SC2 Adapter (translation)
           ↓
      Game Actions
      Build Command Center
      Assign workers
      Defend position
           ↓
      SC2 Game Engine
"""
