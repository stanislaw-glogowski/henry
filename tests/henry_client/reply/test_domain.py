from henry_client.reply import ReplyChunk, ReplyContent, ReplyLine, ReplyText


def test_reply_variants_share_content_without_overlapping_types() -> None:
    chunk = ReplyChunk("chunk")
    line = ReplyLine("line")
    text = ReplyText("text")

    assert isinstance(chunk, ReplyContent)
    assert isinstance(line, ReplyContent)
    assert isinstance(text, ReplyContent)
    assert not isinstance(line, ReplyChunk)
    assert not isinstance(text, ReplyChunk)
