"""Unit tests for mesh_tetgen_error_translation.translate_tetgen_failure -
extracted out of mesh_tetgen_core.fill_core_volume's except-block, isolated
from tetgen itself which is never invoked to produce these exceptions
(they're constructed directly to exercise the translation logic)."""

import pytest

from autoflowcfd.grid.mesh_gen.mesh_tetgen_error_translation import translate_tetgen_failure


class TestTranslateTetgenFailure:
    def test_self_intersection_error_is_translated_with_guidance(self):
        original = RuntimeError("Self-intersection detected at facet 42")
        translated = translate_tetgen_failure(original)
        assert translated is not None
        assert "fewer/thinner BL layers" in str(translated)
        assert "Self-intersection detected at facet 42" in str(translated)

    def test_removevertexbyflips_error_is_translated_with_guidance(self):
        original = RuntimeError("removevertexbyflips() failed")
        translated = translate_tetgen_failure(original)
        assert translated is not None
        assert "internal robustness limit" in str(translated)

    def test_internal_tetgen_error_phrase_is_also_matched(self):
        original = RuntimeError("Internal TetGen error occurred")
        translated = translate_tetgen_failure(original)
        assert translated is not None
        assert "internal robustness limit" in str(translated)

    def test_unrecognized_error_returns_none(self):
        original = RuntimeError("some completely unrelated tetgen failure")
        assert translate_tetgen_failure(original) is None

    def test_matching_is_case_insensitive(self):
        original = RuntimeError("SELF-INTERSECTION at facet 7")
        assert translate_tetgen_failure(original) is not None
