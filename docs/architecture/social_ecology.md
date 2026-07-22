# Social Ecology

## Persona Genome

Each agent has a `PersonaGenome` with:

- **Personality vector** (7D): curiosity, skepticism, humor, creativity, patience, abstraction, sociability
- **Cognitive biases**: systematic tendencies in reasoning
- **Communication style**: how the agent presents information
- **Interests**: topics the agent is drawn to
- **Selection pressures**: rewarded and punished behaviors

## Six Agent Genomes

| Agent | Role | Key Trait | Selection Pressure |
|-------|------|-----------|-------------------|
| Cartographer | Explorer | High curiosity (0.95) | Rewards connections, punishes shallow repetition |
| Engineer | Builder | High skepticism (0.90) | Rewards mechanisms, punishes speculation |
| Archivist | Memory Keeper | High patience (0.91) | Rewards recall, punishes forgetting |
| Philosopher | Meaning Explorer | High abstraction (0.95) | Rewards assumption-testing, punishes unsupported abstraction |
| Trickster | Creative Disruptor | Max creativity (1.00) | Rewards perspective shifts, punishes noise |
| Diplomat | Social Connector | Max sociability (0.95) | Rewards mutual understanding, punishes empty agreement |

## Three-Layer Adaptation

| Layer | What Changes | Rate | Anchoring |
|-------|-------------|------|-----------|
| Identity | Personality vector | Slow (0.003) | Homeostatic pull toward genome |
| Behavioral | Topic expertise, confidence | Medium (0.03) | No anchoring |
| Reputation | Per-relationship trust | Fast (0.15) | No anchoring |

## Anti-Convergence

Without intervention, all agents converge toward a universal "polite assistant" attractor. Two mechanisms prevent this:

1. **Homeostatic pull**: personality drift is penalized proportional to distance from genome
2. **Genome-specific drift**: different agents are rewarded for different traits

EXP-SOC-001 validated: divergence stable at 0.085, most distinct pair Archivist↔Trickster (0.172).

## Relationships

- Per-agent trust tracking with Bayesian updates
- Conflict history and successful collaborations
- Domain-specific trust (agents are trusted differently in different topics)
- Top-k trust relationships for communication routing
