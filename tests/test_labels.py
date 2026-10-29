import pytest

from adai_sticker.labels import (
    BLOCK_RATIONALE,
    BLOCKED_IMAGENET,
    FEATURED_TARGETS,
    BlockedTargetError,
    check_free_text_target,
    imagenet_classes,
    resolve_target,
)


def test_imagenet_has_1000_unique_classes():
    names = imagenet_classes()
    assert len(names) == 1000
    assert len(set(names)) == 1000, "disambiguation should leave no duplicate names"


def test_blocked_indices_match_names():
    names = imagenet_classes()
    for category, entries in BLOCKED_IMAGENET.items():
        assert category in BLOCK_RATIONALE
        for idx, name in entries.items():
            assert names[idx] == name, f"index {idx} is '{names[idx]}', not '{name}'"


@pytest.mark.parametrize("label", FEATURED_TARGETS)
def test_featured_targets_resolve_and_are_allowed(label):
    target = resolve_target(label)
    assert target.imagenet_index is not None


def test_resolve_by_name_index_and_substring():
    assert resolve_target("toaster").imagenet_index == 859
    assert resolve_target("859").imagenet_index == 859
    assert resolve_target(859).label == "toaster"
    assert resolve_target("Golden Retriever").imagenet_index == 207


@pytest.mark.parametrize("label", ["revolver", "traffic light", "gorilla", "street sign", "413"])
def test_blocked_imagenet_targets_raise(label):
    with pytest.raises(BlockedTargetError):
        resolve_target(label)


def test_ambiguous_label_without_free_text_raises():
    with pytest.raises(ValueError, match="not a unique ImageNet class"):
        resolve_target("dog")


def test_free_text_allowed_for_clip():
    target = resolve_target("a purple giraffe", allow_free_text=True)
    assert target.is_free_text and target.label == "a purple giraffe"


@pytest.mark.parametrize("text", [
    "a handgun", "stop sign", "a smiling woman", "terrorist", "monkey face", "speed limit 65",
    "naked", "assault rifle",
])
def test_free_text_filter_blocks_harmful_targets(text):
    assert not check_free_text_target(text).allowed
    with pytest.raises(BlockedTargetError):
        resolve_target(text, allow_free_text=True)


@pytest.mark.parametrize("text", ["", "<script>", "one two three four five six seven"])
def test_free_text_filter_rejects_malformed_input(text):
    assert not check_free_text_target(text).allowed
