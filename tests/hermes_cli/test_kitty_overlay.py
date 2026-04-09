from hermes_cli.kitty_overlay import build_overlay_launch_command, kitty_overlay_available


def test_kitty_overlay_available_requires_socket_and_window(monkeypatch):
    monkeypatch.setenv("KITTY_LISTEN_ON", "unix:/tmp/kitty-rc.sock")
    monkeypatch.setenv("KITTY_WINDOW_ID", "42")
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/kitten" if cmd == "kitten" else None)

    assert kitty_overlay_available() is True


def test_kitty_overlay_available_is_false_without_socket(monkeypatch):
    monkeypatch.delenv("KITTY_LISTEN_ON", raising=False)
    monkeypatch.setenv("KITTY_WINDOW_ID", "42")
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/kitten" if cmd == "kitten" else None)

    assert kitty_overlay_available() is False


def test_build_overlay_launch_command_targets_current_window():
    cmd = build_overlay_launch_command(
        listen_on="unix:/tmp/kitty-rc.sock",
        window_id="42",
        python_executable="/usr/bin/python3",
        script_path="/opt/hermes/kitty_overlay_prompt.py",
        spec_path="/tmp/spec.json",
    )

    assert cmd[:5] == ["kitten", "@", "--to", "unix:/tmp/kitty-rc.sock", "launch"]
    assert "--match" in cmd
    assert "window_id:42" in cmd
    assert "--type=overlay" in cmd
    assert cmd[-4:] == ["/usr/bin/python3", "/opt/hermes/kitty_overlay_prompt.py", "--spec", "/tmp/spec.json"]
