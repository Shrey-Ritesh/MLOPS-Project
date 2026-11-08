from src.settings import optional_warnings, validate_environment


def test_valid_environment():
    assert validate_environment({"GROQ_API_KEY": "gsk_test"}) == []


def test_missing_key_is_reported_by_name_only():
    problems = validate_environment({})
    assert len(problems) == 1
    assert "GROQ_API_KEY" in problems[0]


def test_placeholder_key_is_rejected():
    assert validate_environment({"GROQ_API_KEY": "your_groq_api_key_here"})


def test_key_value_never_appears_in_messages():
    secret = "gsk_super_secret_value"
    problems = validate_environment({"GROQ_API_KEY": secret, "CHROMA_PORT": "abc"})
    assert problems
    assert all(secret not in p for p in problems)


def test_chroma_port_validation():
    ok = {"GROQ_API_KEY": "k"}
    assert validate_environment({**ok, "CHROMA_PORT": "8000"}) == []
    assert validate_environment({**ok, "CHROMA_PORT": "0"})
    assert validate_environment({**ok, "CHROMA_PORT": "70000"})
    assert validate_environment({**ok, "CHROMA_PORT": "http"})


def test_missing_ocr_key_is_only_a_warning():
    env = {"GROQ_API_KEY": "k"}
    assert validate_environment(env) == []
    assert any("OCR_SPACE_API_KEY" in w for w in optional_warnings(env))
    assert optional_warnings({**env, "OCR_SPACE_API_KEY": "real"}) == []
