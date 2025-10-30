from app.services.summarizer import SummaryService


TEXT = (
    "這個翻譯系統可以協助將台語與國語轉成文字，"
    "同時自動產生摘要與關鍵字，方便志工快速整理重點。"
    "摘要演算法採用 TextRank，適合長篇訪談與會議記錄。"
)


def test_summariser_returns_sentences_and_keywords():
    summariser = SummaryService(sentences=2, keywords=3)

    result = summariser.summarise(TEXT)

    assert 0 < len(result["key_points"]) <= 2
    assert all(point.strip() for point in result["key_points"])

    keywords = result["keywords"]
    assert 0 < len(keywords) <= 3
    assert any(len(keyword) >= 1 for keyword in keywords)
