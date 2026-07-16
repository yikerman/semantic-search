import py3langid

from semsearch.cli.ingest.extract import LANGUAGE_SAMPLE_CHARS, detect_language


def test_detect_language_classifies_article_text():
    assert (
        detect_language(
            "This article explains how to configure a Linux desktop and its keyboard "
            "shortcuts. The instructions include practical examples for new users."
        )
        == "en"
    )
    assert (
        detect_language(
            "Cet article explique comment configurer un bureau Linux et ses raccourcis "
            "clavier. Les instructions proposent des exemples pratiques aux debutants."
        )
        == "fr"
    )
    assert (
        detect_language(
            "这篇文章介绍如何配置 Linux 桌面和键盘快捷键，并为新用户提供实用的操作示例。"
        )
        == "zh"
    )


def test_detect_language_uses_a_bounded_sample_and_normalizes_code(monkeypatch):
    seen: list[str] = []

    def classify(text: str):
        seen.append(text)
        return "EN", -1.0

    monkeypatch.setattr(py3langid, "classify", classify)

    assert detect_language("x" * (LANGUAGE_SAMPLE_CHARS + 20), title="Title") == "en"
    assert seen == [f"Title\n\n{'x' * LANGUAGE_SAMPLE_CHARS}"]
