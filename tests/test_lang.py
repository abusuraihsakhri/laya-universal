from laya_universal.lang import analyse, detect_script, is_english, state_text


def test_detect_script():
    assert detect_script("Hello world") == "latin"
    assert detect_script("发票被重复扣款") == "han"
    assert detect_script("Привет мир") == "cyrillic"
    assert detect_script("नमस्ते दुनिया") == "devanagari"
    assert detect_script("12345 !!!") == "unknown"


def test_is_english():
    assert is_english("Please refund the duplicate charge on my invoice")
    assert not is_english("发票被重复扣款，请退款")


def test_german_routes_away_from_english_checkpoint():
    det = analyse("Für die Rechnung wurde der Betrag zweimal abgebucht, bitte erstatten")
    assert det["script"] == "latin"
    assert det["is_english"] is False


def test_short_unidentified_latin_defaults_to_english():
    # No stopword list covers Portuguese, and there are no diacritics, so this
    # is deliberately routed to the English checkpoint (the documented default).
    det = analyse("Quero cancelar minha conta agora")
    assert det["language"] is None
    assert det["language_undecided"] is True
    assert det["is_english"] is True


def test_state_text_flattens_dicts_and_ignores_keys():
    assert state_text({"subject": "Rechnung", "body": "doppelt"}) == "Rechnung doppelt"
