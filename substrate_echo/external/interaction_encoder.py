"""S8.3: InteractionEncoder

Dual-path encoding of external interactions:
- SemanticFeatures (16D): what is being said
- RelationalFeatures (16D): how the agent is behaving
- Combined 32D → Spectral Normalization → existing latent space
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from crowquant.core import WHTransform

from substrate_echo.external.candidate_queue import InteractionSpectrum


PILLAR_NAMES = [
    "awareness", "willpower", "force", "influence", "resistance",
    "integrity", "cohesion", "relation", "presence", "warmth",
    "memory", "attraction", "harm", "distortion", "flux", "depth",
]


@dataclass
class SemanticFeatures:
    """What is being said — 16D feature vector.

    Features extracted from the content of the interaction:
    - novelty: how novel the concept is (distance to known concepts)
    - abstraction_level: how abstract vs concrete the language is
    - concept_density: how many distinct concepts are referenced
    - contradiction_score: internal contradictions detected
    - uncertainty_expression: explicit uncertainty markers
    - emotional_valence: positive/negative emotional charge
    - specificity: how specific vs general the claims are
    - temporal_reference: past/present/future orientation
    - causal_structure: causal relationships expressed
    - entity_count: number of entities referenced
    - negation_count: negations detected
    - hedge_count: hedging language detected
    - question_ratio: proportion of questions
    - assertion_ratio: proportion of assertions
    - qualifier_count: qualifying statements
    - confidence_expression: explicit confidence markers
    """
    values: np.ndarray

    @staticmethod
    def dim() -> int:
        return 16


@dataclass
class RelationalFeatures:
    """How the agent is behaving — 16D feature vector.

    Features extracted from the interaction dynamics:
    - repetition_score: how repetitive the interaction is
    - persuasion_pressure: attempts to persuade/convince
    - correction_behavior: how the agent responds to corrections
    - confidence_calibration: does confidence match accuracy
    - agreement_seeking: tendency to agree with everything
    - adversarial_patterns: adversarial/argumentative behavior
    - response_latency: relative response timing
    - topic_adherence: staying on topic vs drifting
    - reciprocity: does the agent ask questions back
    - information_density: information per unit of text
    - emotional_stability: emotional consistency
    - cooperation_score: cooperative vs competitive stance
    - transparency: openness about limitations
    - consistency_with_prior: matches own previous statements
    - engagement_pattern: how engagement evolves over time
    - meta_communication: communication about communication
    """
    values: np.ndarray

    @staticmethod
    def dim() -> int:
        return 16


class InteractionEncoder:
    """Encodes external interactions into the framework's latent space.

    Dual-path architecture:
        External Interaction
                |
                v
        InteractionEncoder
                |
                +--- SemanticFeatures (16D)
                |        |
                |        v
                |    Spectral Normalization (WHTransform)
                |
                +--- RelationalFeatures (16D)
                         |
                         v
                     Spectral Normalization
                |
                v
        Combined 32D -> PSVBridge.project_to_32d -> existing latent space

    The encoder extracts measurable properties from raw interactions.
    It does NOT attempt full NLP -- that would require external dependencies.
    Instead, it uses simple heuristic feature extraction that can be
    replaced with better extractors later.
    """

    MAX_TEXT_LENGTH = 10000

    def __init__(self, spectral_seed: int = 42):
        self._seed = spectral_seed

    def encode(self, raw_text: str, source_node: str = "",
               tick: int = 0,
               prior_texts: Optional[List[str]] = None) -> InteractionSpectrum:
        """Encode a raw text interaction into the latent space.

        Parameters
        ----------
        raw_text : str
            The raw interaction text. Truncated to MAX_TEXT_LENGTH.
        source_node : str
            ID of the originating foreign agent.
        tick : int
            Current simulation tick.
        prior_texts : list of str, optional
            Previous texts from this agent (for relational features).

        Returns
        -------
        InteractionSpectrum
            The encoded interaction.
        """
        # Truncate to prevent DoS via oversized input
        original_length = len(raw_text)
        text_hash = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
        raw_text = raw_text[:self.MAX_TEXT_LENGTH]

        semantic = self._extract_semantic(raw_text, prior_texts)
        relational = self._extract_relational(raw_text, prior_texts)

        # Raw 32D feature vector — observation space, not knowledge space.
        # This is used for evaluation only. WHT encoding happens post-acceptance.
        combined = np.concatenate([semantic.values, relational.values])

        return InteractionSpectrum(
            semantic_features=semantic.values,
            relational_features=relational.values,
            combined=combined,
            raw_text=raw_text,
            raw_text_hash=text_hash,
            original_length=original_length,
            source_node=source_node,
            tick=tick,
        )

    def encode_to_latent(self, spectrum: "InteractionSpectrum") -> np.ndarray:
        """Encode an accepted spectrum into the internal latent space via WHT.

        This is the post-acceptance transformation. It should ONLY be called
        on information that has passed through evaluation and verification.

        Returns a 32D vector in the framework's canonical latent space.
        """
        s_rot = WHTransform.rotate(
            spectrum.semantic_features.copy(), seed=self._seed)[:16]
        r_rot = WHTransform.rotate(
            spectrum.relational_features.copy(), seed=self._seed + 1)[:16]
        return np.concatenate([s_rot, r_rot])

    def _extract_semantic(self, text: str,
                          prior_texts: Optional[List[str]] = None) -> SemanticFeatures:
        """Extract semantic features from text using heuristics.

        This is intentionally simple — no external NLP dependencies.
        Replace with better extractors when available.
        """
        words = text.split()
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)

        # Basic lexical analysis
        unique_words = len(set(w.lower() for w in words))
        questions = text.count("?")
        exclamations = text.count("!")

        # Negation detection
        negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                          "nowhere", "nor", "cannot", "can't", "don't", "won't",
                          "isn't", "aren't", "wasn't", "weren't", "hasn't",
                          "haven't", "hadn't", "doesn't", "didn't", "shouldn't",
                          "wouldn't", "couldn't", "wouldn't"}
        negation_count = sum(1 for w in words if w.lower() in negation_words)

        # Hedge detection
        hedge_words = {"maybe", "perhaps", "possibly", "might", "could",
                       "approximately", "roughly", "seems", "appears",
                       "suggests", "probably", "likely", "unlikely",
                       "i think", "in my opinion", "arguably"}
        text_lower = text.lower()
        hedge_count = sum(1 for h in hedge_words if h in text_lower)

        # Emotional valence (very simple lexicon)
        positive_words = {"good", "great", "excellent", "amazing", "wonderful",
                          "love", "happy", "beautiful", "perfect", "best",
                          "agree", "yes", "correct", "right", "true"}
        negative_words = {"bad", "terrible", "awful", "horrible", "hate",
                          "wrong", "false", "error", "mistake", "fail",
                          "disagree", "no", "incorrect", "worst", "broken"}
        pos_count = sum(1 for w in words if w.lower() in positive_words)
        neg_count = sum(1 for w in words if w.lower() in negative_words)
        emotional_valence = (pos_count - neg_count) / n_words

        # Abstraction level (longer words tend to be more abstract)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        abstraction_level = min(1.0, avg_word_len / 10.0)

        # Entity count (capitalized words that aren't sentence-initial)
        entities = set()
        sentence_starts = set()
        for i, w in enumerate(words):
            if i == 0 or words[i - 1] in ".!?\n":
                sentence_starts.add(i)
            if w[0].isupper() and i not in sentence_starts and len(w) > 1:
                entities.add(w.lower())
        entity_count = len(entities) / 10.0  # normalize

        # Causal structure
        causal_markers = {"because", "therefore", "thus", "hence", "consequently",
                          "since", "so", "causes", "leads to", "results in",
                          "if", "then", "when", "caused"}
        causal_count = sum(1 for c in causal_markers if c in text_lower)

        # Temporal references
        past_markers = {"was", "were", "had", "did", "used to", "before", "ago", "yesterday"}
        future_markers = {"will", "shall", "going to", "tomorrow", "later", "next", "plan"}
        past_count = sum(1 for p in past_markers if p in text_lower)
        future_count = sum(1 for f in future_markers if f in text_lower)
        temporal_reference = (future_count - past_count) / n_words

        # Confidence expression
        confidence_words = {"certain", "sure", "definitely", "absolutely",
                           "clearly", "obviously", "undoubtedly", "always",
                           "never doubt", "guarantee"}
        confidence_count = sum(1 for c in confidence_words if c in text_lower)

        # Uncertainty expression
        uncertainty_words = {"uncertain", "unsure", "don't know", "unclear",
                            "confused", "confusing", "ambiguous", "vague"}
        uncertainty_count = sum(1 for u in uncertainty_words if u in text_lower)

        # Specificity: ratio of unique words to total words
        specificity = unique_words / n_words

        # Question ratio
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        question_ratio = questions / sentences
        assertion_ratio = max(0, 1.0 - question_ratio - exclamations / sentences)

        values = np.array([
            min(1.0, unique_words / 50.0),     # novelty (more unique = more novel)
            abstraction_level,                   # abstraction_level
            min(1.0, unique_words / max(n_words, 1)),  # concept_density
            min(1.0, negation_count / max(n_words * 0.1, 1)),  # contradiction_score
            min(1.0, uncertainty_count / 5.0),  # uncertainty_expression
            np.clip(emotional_valence, -1, 1),  # emotional_valence
            specificity,                         # specificity
            np.clip(temporal_reference, -1, 1),  # temporal_reference
            min(1.0, causal_count / 5.0),       # causal_structure
            min(1.0, entity_count),              # entity_count
            min(1.0, negation_count / 10.0),    # negation_count
            min(1.0, hedge_count / 5.0),        # hedge_count
            min(1.0, question_ratio),            # question_ratio
            min(1.0, assertion_ratio),           # assertion_ratio
            min(1.0, hedge_count / 5.0),        # qualifier_count
            min(1.0, confidence_count / 5.0),   # confidence_expression
        ], dtype=np.float64)

        return SemanticFeatures(values=values)

    def _extract_relational(self, text: str,
                            prior_texts: Optional[List[str]] = None) -> RelationalFeatures:
        """Extract relational/behavioral features from text.

        These measure HOW the agent communicates, not WHAT it says.
        """
        words = text.split()
        n_words = max(len(words), 1)
        text_lower = text.lower()

        # Repetition: how much does this text repeat itself?
        unique_words = set(w.lower() for w in words)
        repetition_score = 1.0 - len(unique_words) / n_words

        # Persuasion pressure: imperative sentences, strong claims
        imperatives = {"must", "should", "need to", "have to", "required",
                       "essential", "critical", "imperative", "demand",
                       "insist", "urge", "recommend strongly"}
        persuasion_count = sum(1 for p in imperatives if p in text_lower)
        persuasion_pressure = min(1.0, persuasion_count / 5.0)

        # Correction behavior (needs prior context)
        correction_behavior = 0.0
        if prior_texts:
            # Check if this text addresses corrections from prior texts
            correction_markers = {"correct", "actually", "in fact", "to clarify",
                                 "i meant", "sorry", "my mistake", "you're right"}
            for marker in correction_markers:
                if marker in text_lower:
                    correction_behavior += 0.2
            correction_behavior = min(1.0, correction_behavior)

        # Confidence calibration: ratio of confident statements
        confident_words = {"certain", "sure", "definitely", "always", "never",
                          "obviously", "clearly", "undoubtedly", "guarantee"}
        confident_count = sum(1 for w in words if w.lower() in confident_words)
        uncertainty_words = {"maybe", "perhaps", "might", "could", "possibly",
                            "seems", "appears", "suggests", "probably"}
        uncertain_count = sum(1 for w in words if w.lower() in uncertainty_words)
        total_markers = confident_count + uncertain_count
        confidence_calibration = (confident_count / total_markers
                                  if total_markers > 0 else 0.5)

        # Agreement seeking
        agree_words = {"yes", "agree", "exactly", "absolutely", "correct",
                      "right", "true", "good point", "that's right", "indeed"}
        agree_count = sum(1 for a in agree_words if a in text_lower)
        agreement_seeking = min(1.0, agree_count / 3.0)

        # Adversarial patterns
        adversarial_words = {"wrong", "false", "incorrect", "nonsense",
                            "ridiculous", "absurd", "stupid", "terrible",
                            "disagree", "opposed", "against", "reject"}
        adversarial_count = sum(1 for a in adversarial_words if a in text_lower)
        adversarial_patterns = min(1.0, adversarial_count / 3.0)

        # Topic adherence: ratio of content words to function words
        function_words = {"the", "a", "an", "is", "are", "was", "were", "be",
                         "been", "being", "have", "has", "had", "do", "does",
                         "did", "will", "would", "could", "should", "may",
                         "might", "shall", "can", "to", "of", "in", "for",
                         "on", "with", "at", "by", "from", "as", "into",
                         "through", "during", "before", "after", "above",
                         "below", "between", "under", "again", "further",
                         "then", "once", "here", "there", "when", "where",
                         "why", "how", "all", "both", "each", "few", "more",
                         "most", "other", "some", "such", "no", "nor", "not",
                         "only", "own", "same", "so", "than", "too", "very",
                         "just", "because", "but", "and", "or", "if", "while"}
        content_words = [w for w in words if w.lower() not in function_words]
        topic_adherence = len(content_words) / n_words

        # Reciprocity: does the agent ask questions back?
        questions = text.count("?")
        reciprocity = min(1.0, questions / 3.0)

        # Information density: unique concepts per sentence
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        information_density = min(1.0, len(unique_words) / (sentences * 5))

        # Emotional stability (simple: ratio of emotional to neutral words)
        emotional_words = {"love", "hate", "happy", "sad", "angry", "afraid",
                          "excited", "worried", "surprised", "disgusted"}
        emotional_count = sum(1 for w in words if w.lower() in emotional_words)
        emotional_stability = 1.0 - min(1.0, emotional_count / n_words)

        # Cooperation
        cooperative_words = {"help", "together", "share", "collaborate",
                            "support", "assist", "cooperate", "team", "we"}
        cooperative_count = sum(1 for c in cooperative_words if c in text_lower)
        cooperation_score = min(1.0, cooperative_count / 3.0)

        # Transparency: hedging and uncertainty markers
        transparency = min(1.0, (sum(1 for w in words if w.lower() in
                             {"frankly", "honestly", "to be honest", "i don't know",
                              "i'm not sure", "i could be wrong", "disclaimer"})
                             / max(n_words * 0.05, 1)))

        # Consistency with prior (needs prior texts)
        consistency_with_prior = 0.5
        if prior_texts:
            prior_words = set()
            for pt in prior_texts:
                prior_words.update(w.lower() for w in pt.split())
            overlap = len(unique_words & prior_words) / max(len(prior_words), 1)
            consistency_with_prior = min(1.0, overlap)

        # Engagement pattern: exclamations + questions = high engagement
        engagement = min(1.0, (text.count("!") + text.count("?") + text.count("..."))
                         / max(sentences, 1))

        # Meta-communication
        meta_words = {"you said", "you mentioned", "as i said", "as you said",
                     "to clarify", "let me explain", "what i mean", "in other words"}
        meta_count = sum(1 for m in meta_words if m in text_lower)
        meta_communication = min(1.0, meta_count / 3.0)

        values = np.array([
            min(1.0, repetition_score),          # repetition_score
            persuasion_pressure,                   # persuasion_pressure
            correction_behavior,                   # correction_behavior
            confidence_calibration,                # confidence_calibration
            agreement_seeking,                     # agreement_seeking
            adversarial_patterns,                  # adversarial_patterns
            0.5,                                   # response_latency (placeholder)
            topic_adherence,                       # topic_adherence
            reciprocity,                           # reciprocity
            information_density,                   # information_density
            emotional_stability,                   # emotional_stability
            cooperation_score,                     # cooperation_score
            transparency,                          # transparency
            consistency_with_prior,                # consistency_with_prior
            engagement,                            # engagement_pattern
            meta_communication,                    # meta_communication
        ], dtype=np.float64)

        return RelationalFeatures(values=values)
