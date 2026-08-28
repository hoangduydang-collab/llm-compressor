"""Repair of generation configs that transformers' strict validation rejects.

This exists because of a concrete, expensive failure: GLM-5.2 ships a
``generation_config.json`` written by transformers 5.12.0 with ``top_p: 0.95``
and no ``do_sample``. transformers 5.14 validates strictly inside
``save_pretrained``, at the very end -- so both GLM-5.2 distributed PTQ smoke
arms completed calibration and compression, then died writing the checkpoint.
About 11 GPU-hours, no artifact.

The tests pin three things: that GLM's exact config is repaired, that the repair
is the semantics-preserving one (enable sampling rather than delete the sampling
parameter), and that an unrepairable config raises instead of being silently
mangled.
"""

import pytest

from pipeline.quantize import repair_generation_config


class FakeGenerationConfig:
    """Stand-in for transformers' GenerationConfig.

    Mirrors only what the repair touches: attribute access and a strict
    ``validate``. Using the real class would tie these tests to one
    transformers version, and the point is to survive version changes.
    """

    def __init__(self, **kwargs):
        self.do_sample = kwargs.pop("do_sample", False)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.temperature = kwargs.pop("temperature", 1.0)
        self._extra_problem = kwargs.pop("_extra_problem", None)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.validate_calls = 0

    def validate(self, strict=False):
        self.validate_calls += 1
        if self._extra_problem is not None:
            raise ValueError(f"GenerationConfig is invalid: \n- {self._extra_problem}")
        if not self.do_sample:
            for name, default in (
                ("top_p", 1.0), ("top_k", 50), ("typical_p", 1.0),
                ("min_p", None), ("temperature", 1.0),
            ):
                value = getattr(self, name)
                if value != default and value is not None:
                    raise ValueError(
                        "GenerationConfig is invalid: \n"
                        f"- `{name}`: `do_sample` is not set to `True`. However, "
                        f"`{name}` is set to `{value}`"
                    )


class FakeModel:
    def __init__(self, generation_config):
        self.generation_config = generation_config


def glm52_config():
    """GLM-5.2's shipped config, verbatim from the Hub (transformers_version 5.12.0)."""
    return FakeGenerationConfig(
        top_p=0.95,
        temperature=1.0,
        pad_token_id=154820,
        eos_token_id=[154820, 154827, 154829],
    )


# --------------------------------------------------------------------------
# the incident
# --------------------------------------------------------------------------


def test_glm52_shipped_config_is_repaired():
    config = glm52_config()
    with pytest.raises(ValueError, match="do_sample"):
        config.validate(strict=True)

    changes = repair_generation_config(FakeModel(config))

    assert changes, "GLM-5.2's config needed repair but none was reported"
    config.validate(strict=True)  # must not raise now


def test_repair_enables_sampling_rather_than_deleting_top_p():
    """Dropping top_p would silently change the checkpoint's decoding."""
    config = glm52_config()
    repair_generation_config(FakeModel(config))

    assert config.do_sample is True
    assert config.top_p == 0.95, "top_p must be preserved, not removed"
    assert config.temperature == 1.0


def test_repair_reports_what_it_changed_and_why():
    """The checkpoint's provenance depends on this being visible, not silent."""
    changes = repair_generation_config(FakeModel(glm52_config()))
    assert len(changes) == 1
    message = changes[0]
    assert "do_sample" in message
    assert "top_p" in message, f"should name the offending param: {message}"


# --------------------------------------------------------------------------
# no-op cases
# --------------------------------------------------------------------------


def test_already_valid_config_is_untouched():
    config = FakeGenerationConfig(do_sample=True, top_p=0.95)
    assert repair_generation_config(FakeModel(config)) == []
    assert config.do_sample is True


def test_all_default_config_needs_no_repair():
    config = FakeGenerationConfig()
    assert repair_generation_config(FakeModel(config)) == []
    assert config.do_sample is False, "must not enable sampling gratuitously"


def test_greedy_config_stays_greedy():
    """A model that genuinely wants greedy decoding must not be switched."""
    config = FakeGenerationConfig(do_sample=False, top_p=1.0, temperature=1.0)
    assert repair_generation_config(FakeModel(config)) == []
    assert config.do_sample is False


def test_model_without_generation_config_is_a_noop():
    class Bare:
        pass

    assert repair_generation_config(Bare()) == []


def test_none_generation_config_is_a_noop():
    assert repair_generation_config(FakeModel(None)) == []


# --------------------------------------------------------------------------
# other sampling params
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, named",
    [
        ({"top_k": 20}, "top_k"),
        ({"typical_p": 0.9}, "typical_p"),
        ({"min_p": 0.05}, "min_p"),
        ({"temperature": 0.7}, "temperature"),
        ({"top_p": 0.8, "top_k": 40}, "top_k"),
    ],
)
def test_other_sampling_params_also_trigger_the_repair(kwargs, named):
    config = FakeGenerationConfig(**kwargs)
    changes = repair_generation_config(FakeModel(config))
    assert config.do_sample is True
    assert named in changes[0], f"{named} not named in {changes[0]}"
    config.validate(strict=True)


# --------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------


def test_unrepairable_config_raises_rather_than_shipping_silently():
    """Better to refuse at load than to die at the end of a multi-hour save."""
    config = FakeGenerationConfig(_extra_problem="something we cannot fix")
    with pytest.raises(RuntimeError, match="cannot be made to pass"):
        repair_generation_config(FakeModel(config))


def test_unrepairable_error_reports_the_remaining_problem():
    config = FakeGenerationConfig(_extra_problem="mystery constraint violated")
    with pytest.raises(RuntimeError) as excinfo:
        repair_generation_config(FakeModel(config))
    assert "mystery constraint violated" in str(excinfo.value)


def test_validation_is_rechecked_after_repair():
    """The repair must prove itself, not assume success."""
    config = glm52_config()
    repair_generation_config(FakeModel(config))
    # once to detect, once to confirm
    assert config.validate_calls >= 2
