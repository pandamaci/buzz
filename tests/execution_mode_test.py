from buzz.execution_mode import is_force_cpu_enabled


def test_launcher_mode_overrides_inherited_force_cpu(monkeypatch):
    monkeypatch.setenv("BUZZ_FORCE_CPU", "true")
    monkeypatch.setenv("BUZZ_EXECUTION_MODE", "cuda")
    assert not is_force_cpu_enabled()


def test_cpu_launcher_mode_overrides_disabled_force_cpu(monkeypatch):
    monkeypatch.delenv("BUZZ_FORCE_CPU", raising=False)
    monkeypatch.setenv("BUZZ_EXECUTION_MODE", "cpu")
    assert is_force_cpu_enabled()
