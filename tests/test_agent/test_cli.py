"""agent 层 CLI 参数解析与 payload 构建测试。"""
import pytest

from lang_agent.agent.cli import _build_payload, main


def test_build_payload_message_only():
    payload = _build_payload(message="你好", model=None, provider=None, protocol=None, thread_id=None)
    assert payload == {"message": "你好"}


def test_build_payload_with_all_fields():
    payload = _build_payload(
        message="hi", model="m", provider="p", protocol="proto", thread_id="t1"
    )
    assert payload == {
        "message": "hi",
        "model": "m",
        "provider": "p",
        "protocol": "proto",
        "thread_id": "t1",
    }


def test_main_unknown_command_exits_nonzero():
    # argparse 对未知子命令的标准行为：SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        main(["no_such_command"])
    assert exc_info.value.code == 2


def test_main_chat_requires_message():
    with pytest.raises(SystemExit) as exc_info:
        main(["chat"])
    assert exc_info.value.code == 2
