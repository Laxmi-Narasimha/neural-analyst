from app.services.system_info_service import build_system_info, _detect_llm_provider, _llm_ready


def test_detect_llm_provider_ollama():
    assert _detect_llm_provider("ollama/llama3") == "ollama"


def test_llm_ready_ollama_without_keys():
    assert _llm_ready("ollama/llama3", {"openai": False, "anthropic": False}) is True


def test_llm_ready_openai_requires_key():
    assert _llm_ready("gpt-4o-mini", {"openai": False}) is False
    assert _llm_ready("gpt-4o-mini", {"openai": True}) is True


def test_build_system_info_shape():
    info = build_system_info()
    assert info["deployment_mode"] in {"self_host", "saas"}
    assert "compute" in info
    assert info["compute"]["runs_on"] == "your_machine"
    assert "api_keys_present" in info
    assert "self_host" in info
    assert "talk_to_data_without_llm" in info["capabilities"]