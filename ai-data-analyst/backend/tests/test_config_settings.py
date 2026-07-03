from app.core.config import Settings


def test_settings_coerce_release_debug_flag_to_false():
    settings = Settings(debug="release")
    assert settings.debug is False


def test_settings_coerce_debug_alias_to_true():
    settings = Settings(debug="debug")
    assert settings.debug is True
