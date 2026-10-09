"""mesh_tetgen_error_translation.translate_tetgen_failure 的单元测试——
它是从 mesh_tetgen_core.fill_core_volume 的 except 块里抽出来的，与 tetgen
本身隔离：从不调用 tetgen 来产生这些异常（直接构造异常来测翻译逻辑）。
"""

import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_error_translation import translate_tetgen_failure


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
