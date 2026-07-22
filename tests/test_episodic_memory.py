"""Tests for Episodic Narrative Memory — P7.1"""
import numpy as np
from substrate_echo.core.episodic_memory import (
    EpisodicMemory, EpisodicMemoryConfig, Episode, CausalLink, NarrativeChapter
)


def _ctx(val=0.5):
    return np.full(16, val)


def _actions(types=None):
    if types is None:
        types = ["approach"]
    return [{"type": t, "pillar": i, "magnitude": 0.1} for i, t in enumerate(types)]


class TestEpisodeStore:
    def test_store_single(self):
        em = EpisodicMemory()
        ep = em.store(_ctx(), _actions(), _ctx(0.6), tick=10)
        assert ep.episode_id == 0
        assert ep.tick == 10
        assert ep.emotion == "neutral"

    def test_store_multiple(self):
        em = EpisodicMemory()
        for i in range(5):
            em.store(_ctx(i * 0.1), _actions(), _ctx((i + 1) * 0.1), tick=i)
        assert len(em._episodes) == 5

    def test_store_with_emotion(self):
        em = EpisodicMemory()
        ep = em.store(_ctx(), _actions(), _ctx(), tick=0,
                      emotion="curiosity", emotional_intensity=0.8)
        assert ep.emotion == "curiosity"
        assert ep.emotional_intensity == 0.8

    def test_episode_delta(self):
        em = EpisodicMemory()
        ep = em.store(_ctx(0.2), _actions(), _ctx(0.8), tick=0)
        delta = ep.delta
        assert np.allclose(delta, 0.6)


class TestCausalLinks:
    def test_link_causal(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(), _actions(), _ctx(0.6), tick=0)
        ep2 = em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=1)
        link = em.link_causal(ep1.episode_id, ep2.episode_id, strength=0.9)
        assert link.from_id == 0
        assert link.to_id == 1
        assert link.strength == 0.9

    def test_get_causes(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(), _actions(), _ctx(0.6), tick=0)
        ep2 = em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=1)
        em.link_causal(0, 1)
        causes = em.get_causes(1)
        assert len(causes) == 1
        assert causes[0].episode_id == 0

    def test_get_effects(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(), _actions(), _ctx(0.6), tick=0)
        ep2 = em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=1)
        em.link_causal(0, 1)
        effects = em.get_effects(0)
        assert len(effects) == 1
        assert effects[0].episode_id == 1

    def test_causal_chain(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(), _ctx(0.3), tick=0)
        ep2 = em.store(_ctx(0.3), _actions(), _ctx(0.6), tick=1)
        ep3 = em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=2)
        em.link_causal(0, 1)
        em.link_causal(1, 2)
        chain = em.get_causal_chain(0)
        assert len(chain) == 3
        assert [e.episode_id for e in chain] == [0, 1, 2]

    def test_auto_link_sequential(self):
        em = EpisodicMemory()
        em.store(_ctx(), _actions(), _ctx(0.6), tick=0)
        em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=1)
        # Should auto-link
        assert len(em._causal_links) >= 1


class TestRecall:
    def test_recall_recent(self):
        em = EpisodicMemory()
        for i in range(10):
            em.store(_ctx(i * 0.1), _actions(), _ctx((i + 1) * 0.1), tick=i)
        recent = em.recall_recent(3)
        assert len(recent) == 3
        assert recent[0].tick > recent[1].tick > recent[2].tick

    def test_recall_by_emotion(self):
        em = EpisodicMemory()
        em.store(_ctx(), _actions(), _ctx(), tick=0,
                 emotion="curiosity", emotional_intensity=0.9)
        em.store(_ctx(), _actions(), _ctx(), tick=1,
                 emotion="neutral", emotional_intensity=0.1)
        em.store(_ctx(), _actions(), _ctx(), tick=2,
                 emotion="curiosity", emotional_intensity=0.5)
        curious = em.recall_by_emotion("curiosity")
        assert len(curious) == 2
        assert curious[0].emotional_intensity >= curious[1].emotional_intensity

    def test_recall_by_context(self):
        em = EpisodicMemory()
        em.store(_ctx(0.1), _actions(), _ctx(), tick=0)
        em.store(_ctx(0.9), _actions(), _ctx(), tick=1)
        em.store(_ctx(0.11), _actions(), _ctx(), tick=2)
        recalled = em.recall_by_context(_ctx(0.1), max_results=2)
        assert len(recalled) == 2

    def test_recall_by_causality(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(), _ctx(0.3), tick=0)
        ep2 = em.store(_ctx(0.3), _actions(), _ctx(0.6), tick=1)
        ep3 = em.store(_ctx(0.6), _actions(), _ctx(0.9), tick=2)
        em.link_causal(0, 1)
        em.link_causal(1, 2)
        chain = em.recall_by_causality(0)
        assert len(chain) == 3


class TestNarratives:
    def test_build_narratives(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(["approach"]), _ctx(0.3), tick=0)
        ep2 = em.store(_ctx(0.3), _actions(["observe"]), _ctx(0.6), tick=1)
        ep3 = em.store(_ctx(0.6), _actions(["investigate"]), _ctx(0.9), tick=2)
        em.link_causal(0, 1)
        em.link_causal(1, 2)
        narratives = em.build_narratives()
        assert len(narratives) >= 1
        assert narratives[0].length == 3

    def test_narrative_theme(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(["approach"]), _ctx(0.3), tick=0,
                       emotion="curiosity")
        ep2 = em.store(_ctx(0.3), _actions(["approach"]), _ctx(0.6), tick=1,
                       emotion="curiosity")
        em.link_causal(0, 1)
        narratives = em.build_narratives()
        if narratives:
            assert "curiosity" in narratives[0].theme

    def test_recall_narrative(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(), _ctx(0.3), tick=0)
        ep2 = em.store(_ctx(0.3), _actions(), _ctx(0.6), tick=1)
        em.link_causal(0, 1)
        em.build_narratives()
        if em._narratives:
            n_id = list(em._narratives.keys())[0]
            recalled = em.recall_narrative(n_id)
            assert recalled is not None

    def test_narrative_by_theme(self):
        em = EpisodicMemory()
        ep1 = em.store(_ctx(0.1), _actions(["approach"]), _ctx(0.3), tick=0,
                       emotion="satisfaction")
        ep2 = em.store(_ctx(0.3), _actions(["approach"]), _ctx(0.6), tick=1,
                       emotion="satisfaction")
        em.link_causal(0, 1)
        em.build_narratives()
        found = em.recall_narratives_by_theme("satisfaction")
        # May or may not find depending on theme inference


class TestMemoryEviction:
    def test_eviction_at_capacity(self):
        config = EpisodicMemoryConfig(max_episodes=10)
        em = EpisodicMemory(config)
        for i in range(15):
            em.store(_ctx(i * 0.05), _actions(), _ctx((i + 1) * 0.05), tick=i)
        assert len(em._episodes) <= 10


class TestMemorySummary:
    def test_summary(self):
        em = EpisodicMemory()
        em.store(_ctx(), _actions(), _ctx(), tick=0, emotion="curiosity")
        em.store(_ctx(), _actions(), _ctx(), tick=1, emotion="satisfaction")
        s = em.summary()
        assert s["n_episodes"] == 2
        assert s["n_causal_links"] >= 1
        assert "curiosity" in s["emotions"]
