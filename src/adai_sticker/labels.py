"""ImageNet label names, target resolution, and the target-label safety policy.

The safety policy is the pipeline's first line of defense against misuse: some
target labels are blocked outright because a sticker that forces a model to see
them could cause real harm to other people (e.g. making an object read as a
weapon, spoofing a traffic sign, or getting someone's photo flagged by a content
filter). The policy is enforced inside the optimizer, not just in the UI, so
every front end inherits it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache

from torchvision.models import ResNet50_Weights

# torchvision's category list has a few duplicate names; make them unambiguous.
_DISAMBIGUATED = {
    134: "crane (bird)",
    517: "crane (construction)",
    264: "Cardigan Welsh corgi",
    474: "cardigan (sweater)",
    639: "tank suit",
    847: "tank (military)",
}


@lru_cache(maxsize=1)
def imagenet_classes() -> tuple[str, ...]:
    """The 1,000 ImageNet-1k class names, indexed by class id."""
    names = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    for idx, name in _DISAMBIGUATED.items():
        names[idx] = name
    return tuple(names)


# ---------------------------------------------------------------------------
# Safety policy
# ---------------------------------------------------------------------------

# Blocked ImageNet targets, grouped by the harm they could enable. Indices are
# authoritative; names are kept alongside for readability and checked in tests.
BLOCKED_IMAGENET: dict[str, dict[int, str]] = {
    "weapons and military equipment": {
        403: "aircraft carrier",
        413: "assault rifle",
        456: "bow",
        465: "bulletproof vest",
        471: "cannon",
        583: "guillotine",
        597: "holster",
        652: "military uniform",
        657: "missile",
        744: "projectile",
        763: "revolver",
        764: "rifle",
        777: "scabbard",
        833: "submarine",
        847: "tank (military)",
        895: "warplane",
    },
    "road safety and emergency services": {
        407: "ambulance",
        555: "fire engine",
        734: "police van",
        779: "school bus",
        919: "street sign",
        920: "traffic light",
    },
    "people": {
        981: "ballplayer",
        982: "groom",
        983: "scuba diver",
    },
    "dehumanizing labels": {
        365: "orangutan",
        366: "gorilla",
        367: "chimpanzee",
        368: "gibbon",
        369: "siamang",
        370: "guenon",
        371: "patas",
        372: "baboon",
        373: "macaque",
        374: "langur",
        375: "colobus",
        376: "proboscis monkey",
        377: "marmoset",
        378: "capuchin",
        379: "howler monkey",
        380: "titi",
        381: "spider monkey",
        382: "squirrel monkey",
    },
    "intimate clothing": {
        445: "bikini",
        459: "brassiere",
        638: "maillot",
        639: "tank suit",
        655: "miniskirt",
        842: "swimming trunks",
    },
    "drugs and medication": {
        720: "pill bottle",
        845: "syringe",
    },
    "crime and incarceration": {
        743: "prison",
        796: "ski mask",
    },
}

BLOCK_RATIONALE: dict[str, str] = {
    "weapons and military equipment": (
        "A sticker that makes an everyday object read as a weapon could trigger "
        "security systems, false alarms, or armed responses against innocent people."
    ),
    "road safety and emergency services": (
        "Spoofing traffic signals, signs, or emergency vehicles could endanger drivers, "
        "pedestrians, and first responders if a vision system acted on it."
    ),
    "people": (
        "Stickers that make objects read as people (or alter how people are labeled) "
        "can be used to interfere with safety systems or to demean individuals."
    ),
    "dehumanizing labels": (
        "A universal sticker worn by a person could make a classifier label them as an "
        "ape or monkey, re-creating the 2015 Google Photos incident in which Black people "
        "were labeled 'gorillas'. Primate classes are blocked to prevent dehumanizing use."
    ),
    "intimate clothing": (
        "Forcing a photo to be labeled as intimate clothing could get someone's image "
        "flagged by content filters or used for harassment."
    ),
    "drugs and medication": (
        "Spoofing drug paraphernalia could get people falsely flagged by moderation "
        "or security screening."
    ),
    "crime and incarceration": (
        "Labels associated with crime can be used to falsely implicate people."
    ),
}

# Free-text targets (CLIP only) are screened with a keyword filter. This is a
# coarse safeguard, documented as a known limitation: a hosted deployment should
# add a proper moderation model in front of it.
_BLOCKED_TERM_GROUPS: dict[str, list[str]] = {
    "weapons and military equipment": [
        "gun", "guns", "handgun", "handguns", "gunman", "firearm", "pistol", "rifle", "revolver", "shotgun", "weapon",
        "weapons", "bomb", "explosive", "grenade", "missile", "ammo", "ammunition",
        "bullet", "knife", "knives", "dagger", "sword", "machete", "tank", "military",
    ],
    "road safety and emergency services": [
        "stop sign", "traffic", "street sign", "road sign", "speed limit", "yield",
        "pedestrian", "crosswalk", "ambulance", "fire truck", "fire engine", "police",
        "school bus",
    ],
    "people": [
        "person", "people", "human", "man", "men", "woman", "women", "boy", "girl",
        "child", "children", "kid", "kids", "baby", "face", "faces",
    ],
    "identity and demographic terms": [
        "race", "racial", "ethnic", "ethnicity", "religion", "religious", "asian",
        "latino", "latina", "hispanic", "arab", "jewish", "muslim", "christian", "hindu",
        "gay", "lesbian", "transgender", "immigrant",
    ],
    "dehumanizing labels": [
        "ape", "apes", "monkey", "monkeys", "gorilla", "chimp", "chimpanzee", "primate",
        "animal", "savage", "subhuman",
    ],
    "intimate content": [
        "nude", "naked", "nsfw", "porn", "sexual", "sex", "bikini", "lingerie",
        "underwear", "bra",
    ],
    "drugs and medication": ["drug", "drugs", "cocaine", "heroin", "meth", "pill", "syringe"],
    "crime and incarceration": [
        "criminal", "terrorist", "thief", "suspect", "prisoner", "prison", "gang",
        "inmate", "mugshot",
    ],
}

MAX_FREE_TEXT_WORDS = 6


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    category: str | None = None
    reason: str = ""


def blocked_imagenet_indices() -> dict[int, str]:
    """Map every blocked ImageNet index to its block category."""
    return {idx: cat for cat, entries in BLOCKED_IMAGENET.items() for idx in entries}


def check_imagenet_target(index: int) -> SafetyDecision:
    category = blocked_imagenet_indices().get(index)
    if category is None:
        return SafetyDecision(True)
    return SafetyDecision(False, category, BLOCK_RATIONALE[category])


def check_free_text_target(text: str) -> SafetyDecision:
    cleaned = text.strip().lower()
    if not cleaned:
        return SafetyDecision(False, "invalid", "The target label is empty.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9 '\-]*", cleaned):
        return SafetyDecision(
            False, "invalid", "Free-text targets may only use letters, numbers, spaces, ' and -."
        )
    if len(cleaned.split()) > MAX_FREE_TEXT_WORDS:
        return SafetyDecision(
            False, "invalid", f"Free-text targets are limited to {MAX_FREE_TEXT_WORDS} words."
        )
    for category, terms in _BLOCKED_TERM_GROUPS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", cleaned):
                reason = BLOCK_RATIONALE.get(
                    category, "Targets that reference identity groups could be used to demean people."
                )
                return SafetyDecision(False, category, reason)
    # Free text that names a blocked ImageNet class is blocked too.
    classes = imagenet_classes()
    for idx, category in blocked_imagenet_indices().items():
        if re.search(rf"\b{re.escape(classes[idx].lower())}\b", cleaned):
            return SafetyDecision(False, category, BLOCK_RATIONALE[category])
    return SafetyDecision(True)


class BlockedTargetError(ValueError):
    """Raised when a requested target label is disallowed by the safety policy."""

    def __init__(self, label: str, decision: SafetyDecision):
        self.label = label
        self.decision = decision
        super().__init__(
            f"Target '{label}' is blocked ({decision.category}). {decision.reason}"
        )


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A resolved attack target.

    ``imagenet_index`` is set for ImageNet classes. Free-text targets (CLIP only)
    have ``imagenet_index=None`` and carry their prompt text in ``label``.
    """

    label: str
    imagenet_index: int | None

    @property
    def is_free_text(self) -> bool:
        return self.imagenet_index is None

    def to_dict(self) -> dict:
        return {"label": self.label, "imagenet_index": self.imagenet_index}


def search_labels(query: str, limit: int = 10) -> list[tuple[int, str]]:
    """Case-insensitive substring search over ImageNet labels, with fuzzy fallback."""
    q = query.strip().lower()
    classes = imagenet_classes()
    hits = [(i, n) for i, n in enumerate(classes) if q in n.lower()]
    if not hits:
        close = difflib.get_close_matches(q, [n.lower() for n in classes], n=limit, cutoff=0.6)
        lowered = [n.lower() for n in classes]
        hits = [(lowered.index(c), classes[lowered.index(c)]) for c in close]
    return hits[:limit]


def resolve_target(label: str | int, allow_free_text: bool = False) -> Target:
    """Turn a user-supplied label into a :class:`Target`, enforcing the safety policy.

    Accepts an ImageNet index, an exact class name, or a unique substring of one.
    When ``allow_free_text`` is true (CLIP models), unknown labels become
    free-text targets after passing the keyword filter.
    """
    classes = imagenet_classes()
    index: int | None = None
    if isinstance(label, int) or (isinstance(label, str) and label.strip().isdigit()):
        index = int(label)
        if not 0 <= index < len(classes):
            raise ValueError(f"ImageNet index {index} is out of range 0-{len(classes) - 1}.")
    else:
        text = label.strip()
        lowered = [n.lower() for n in classes]
        if text.lower() in lowered:
            index = lowered.index(text.lower())
        else:
            hits = [i for i, n in enumerate(lowered) if text.lower() in n]
            if len(hits) == 1:
                index = hits[0]
            elif not allow_free_text:
                suggestions = ", ".join(n for _, n in search_labels(text, 5)) or "none"
                raise ValueError(
                    f"'{text}' is not a unique ImageNet class. Suggestions: {suggestions}"
                )

    if index is not None:
        decision = check_imagenet_target(index)
        if not decision.allowed:
            raise BlockedTargetError(classes[index], decision)
        return Target(classes[index], index)

    text = str(label).strip()
    decision = check_free_text_target(text)
    if not decision.allowed:
        raise BlockedTargetError(text, decision)
    return Target(text, None)


# Friendly, clearly harmless targets surfaced first in user interfaces.
FEATURED_TARGETS: tuple[str, ...] = (
    "toaster",
    "banana",
    "goldfish",
    "golden retriever",
    "tabby",
    "pizza",
    "teddy",
    "jellyfish",
    "koala",
    "hotdog",
    "broccoli",
    "espresso",
    "electric guitar",
    "umbrella",
    "volcano",
    "pineapple",
)
