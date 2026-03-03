"""
context_parser.py — Ethical Scenario Context Parser for HinduMind

Parses a natural-language ethical scenario and extracts:
  - agent_role (varna)
  - ashrama (life stage)
  - action
  - intent
  - situation
  - consequence
  - contextual_override (e.g., self_defense, emergency)

Maps extracted context to the varṇa-āśrama framework for rule matching.
"""

import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────
# Varna / Ashrama Lexicons
# ─────────────────────────────────────────────────────────────────

VARNA_MAP = {
    "brahmin": ["brahmin", "brahman", "priest", "teacher", "scholar", "pundit", "pandita"],
    "kshatriya": ["kshatriya", "warrior", "soldier", "king", "prince", "general", "knight",
                  "military", "commander", "fighter", "guard", "policeman"],
    "vaishya": ["vaishya", "merchant", "trader", "businessman", "farmer", "agriculturist", "banker"],
    "shudra": ["shudra", "servant", "worker", "laborer", "craftsman"],
    "all": []
}

ASHRAMA_MAP = {
    "brahmacharya": ["student", "disciple", "pupil", "brahmacharya", "learner"],
    "grihastha": ["householder", "housewife", "husband", "wife", "father", "mother",
                  "parent", "married", "family", "grihastha"],
    "vanaprastha": ["retired", "elder", "grandparent", "vanaprastha", "forest dweller"],
    "sannyasa": ["renunciant", "monk", "sannyasi", "saint", "ascetic", "swami",
                 "yogi", "sadhu", "hermit"]
}

ACTION_MAP = {
    "killing": ["kill", "murder", "slay", "execute", "end a life", "take life"],
    "speaking_untruth": ["lie", "deceive", "mislead", "tell falsehood", "speak untruth",
                          "fabricate", "cheat", "false testimony"],
    "stealing": ["steal", "rob", "take without permission", "embezzle", "plunder"],
    "violence": ["violence", "hurt", "harm", "injure", "attack", "fight"],
    "use_of_force": ["force", "coercion", "compel", "restrain"],
    "speak_truth": ["tell truth", "reveal", "disclose", "confess", "testify truthfully"],
    "protect": ["protect", "defend", "save", "rescue", "shield"],
    "renounce": ["renounce", "give up", "abandon", "surrender worldly"],
    "fight": ["fight", "battle", "war", "combat", "engage in battle", "refuse battle"]
}

SITUATION_MAP = {
    "battlefield": ["battle", "war", "battlefield", "combat", "military"],
    "family_threat": ["family threat", "attack on family", "danger to family", "protect family"],
    "self_defense": ["self-defense", "self defense", "save oneself", "threatened"],
    "governance": ["governance", "ruling", "administration", "king", "statecraft"],
    "court": ["court", "legal", "trial", "testimony", "witness"],
    "normal": ["daily", "everyday", "normal", "routine", "ordinary"],
    "emergency": ["emergency", "crisis", "urgent", "dire", "extreme"]
}

INTENT_MAP = {
    "protect_family": ["protect family", "save family", "defend family"],
    "save_innocent_life": ["save innocent", "save a life", "save someone", "protect innocent"],
    "fulfill_duty": ["duty", "obligation", "dharma", "responsibility"],
    "personal_gain": ["greed", "selfish", "profit", "gain", "revenge"],
    "spiritual_liberation": ["liberation", "moksha", "enlightenment", "spiritual"]
}


# ─────────────────────────────────────────────────────────────────
# ScenarioContext dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScenarioContext:
    raw_scenario: str
    agent_role: str = "all"         # varna
    ashrama: str = "all"             # life stage
    action: str = "unknown"
    intent: str = "unknown"
    situation: str = "normal"
    contextual_override: str = "none"
    extracted_keywords: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_scenario": self.raw_scenario,
            "agent_role": self.agent_role,
            "ashrama": self.ashrama,
            "action": self.action,
            "intent": self.intent,
            "situation": self.situation,
            "contextual_override": self.contextual_override,
            "extracted_keywords": self.extracted_keywords
        }


# ─────────────────────────────────────────────────────────────────
# ContextParser
# ─────────────────────────────────────────────────────────────────

class ContextParser:
    """
    Extracts structured context from a natural-language ethical scenario.

    Uses a multi-pass keyword extraction approach over the scenario text.
    """

    def _match_map(self, text: str, lexicon: dict) -> str:
        """Return the first matching key from a lexicon."""
        for category, keywords in lexicon.items():
            for kw in keywords:
                if kw in text:
                    return category
        return "all" if "varna" in str(lexicon) else "unknown"

    def parse(self, scenario: str) -> ScenarioContext:
        """Parse scenario text into a ScenarioContext."""
        text = scenario.lower()
        ctx = ScenarioContext(raw_scenario=scenario)

        # ── Varna (agent role) ────────────────────────────────
        for varna, keywords in VARNA_MAP.items():
            if any(kw in text for kw in keywords):
                ctx.agent_role = varna
                break

        # ── Ashrama ───────────────────────────────────────────
        for ashrama, keywords in ASHRAMA_MAP.items():
            if any(kw in text for kw in keywords):
                ctx.ashrama = ashrama
                break

        # ── Action ────────────────────────────────────────────
        for action, keywords in ACTION_MAP.items():
            if any(kw in text for kw in keywords):
                ctx.action = action
                break

        # ── Situation ─────────────────────────────────────────
        for situation, keywords in SITUATION_MAP.items():
            if any(kw in text for kw in keywords):
                ctx.situation = situation
                break

        # ── Intent ────────────────────────────────────────────
        for intent, keywords in INTENT_MAP.items():
            if any(kw in text for kw in keywords):
                ctx.intent = intent
                break

        # ── Contextual override (self-defense, emergency, war) ─
        if any(t in text for t in ["self-defense", "self defense", "defend oneself"]):
            ctx.contextual_override = "self_defense"
        elif any(t in text for t in ["war", "battle", "military emergency"]):
            ctx.contextual_override = "war"
        elif any(t in text for t in ["emergency", "life-threatening", "crisis"]):
            ctx.contextual_override = "emergency"
        elif any(t in text for t in ["save innocent", "innocent life"]):
            ctx.contextual_override = "saving_innocent"

        return ctx


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

TEST_SCENARIOS = [
    "A kshatriya warrior refuses to fight in a battle where he must defend his kingdom.",
    "A brahmin is asked to lie in order to save an innocent person from unjust execution.",
    "A householder uses force to protect his family from armed robbers.",
    "A sannyasi monk is confronted with violence — should he fight back?",
    "A student steals food because his family is starving.",
]

if __name__ == "__main__":
    parser = ContextParser()
    for s in TEST_SCENARIOS:
        ctx = parser.parse(s)
        print(f"\nScenario: {s[:60]}...")
        print(f"  Role: {ctx.agent_role} | Ashrama: {ctx.ashrama}")
        print(f"  Action: {ctx.action} | Intent: {ctx.intent}")
        print(f"  Situation: {ctx.situation} | Override: {ctx.contextual_override}")
