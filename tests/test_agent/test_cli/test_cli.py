"""agent 层 CLI 参数解析、payload 构建与交互模式测试（无网络依赖）。"""
import builtins
import json

import pytest

import lang_agent.agent.cli.main as cli_main
from lang_agent.agent.cli.main import _build_payload, _truncate, main


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


def test_main_chat_without_message_enters_interactive(monkeypatch):
    calls = []

    def fake_interactive(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(cli_main, "_interactive", fake_interactive)
    assert main(["chat"]) == 0
    assert len(calls) == 1


def _mock_input(monkeypatch, replies: list[str]):
    """依次返回 replies 的内容（最后给 /exit 兜底防死循环）。"""
    it = iter(replies)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            return "/exit"

    monkeypatch.setattr(builtins, "input", fake_input)


def _mock_stream(monkeypatch, sink: list[dict[str, str]]):
    def fake_stream(
        base_url: str, payload: dict[str, str], *, max_output_chars: int = cli_main.DEFAULT_MAX_OUTPUT_CHARS
    ) -> int:
        sink.append(payload)
        return 0

    monkeypatch.setattr(cli_main, "_chat_stream", fake_stream)


def test_interactive_sends_message_stream_default(monkeypatch):
    _mock_input(monkeypatch, ["你好", "/exit"])
    sent = []
    _mock_stream(monkeypatch, sent)
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (True, base_url, None))
    assert main(["chat"]) == 0
    assert len(sent) == 1
    assert sent[0]["message"] == "你好"
    assert sent[0]["thread_id"].startswith("cli-")
    assert "model" not in sent[0] and "provider" not in sent[0] and "protocol" not in sent[0]


def test_interactive_reuses_thread_id_and_honors_args(monkeypatch):
    _mock_input(monkeypatch, ["第一句", "第二句", "/exit"])
    sent = []
    _mock_stream(monkeypatch, sent)
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (True, base_url, None))
    assert main(["chat"]) == 0
    assert len(sent) == 2
    assert sent[0]["thread_id"] == sent[1]["thread_id"]
    assert sent[0]["thread_id"].startswith("cli-")

    _mock_input(monkeypatch, ["你好", "/exit"])
    sent.clear()
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (True, base_url, None))
    assert main(["chat", "--thread-id", "t9"]) == 0
    assert sent[0]["thread_id"] == "t9"


def test_interactive_exit_command_help(monkeypatch, capsys):
    _mock_input(monkeypatch, ["/help", "/unknown", "   ", "/quit"])
    sent = []
    _mock_stream(monkeypatch, sent)
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (True, base_url, None))
    assert main(["chat"]) == 0
    assert sent == []
    out = capsys.readouterr().out
    assert "/exit、/quit" in out
    assert "未知命令" in out


def test_interactive_keyboard_interrupt_exits_clean(monkeypatch):
    def raising_input(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raising_input)
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (True, base_url, None))
    assert main(["chat"]) == 0


def test_interactive_fails_when_server_unreachable(monkeypatch):
    _mock_input(monkeypatch, ["你好", "/exit"])
    monkeypatch.setattr(cli_main, "_ensure_server", lambda base_url: (False, None, None))
    assert main(["chat"]) == 1


def _mock_httpx_sse(monkeypatch, events: list[tuple[str, object]]):
    """构造一个假的 httpx.Client，按 SSE 帧序列 (event, data) 依次产出。"""
    import httpx

    class FakeResponse:
        def __init__(self):
            self.status_code = 200

        def read(self):
            return b""

        def iter_lines(self):
            for event, data in events:
                yield f"event: {event}"
                if data is not None:
                    yield f"data: {json.dumps(data, ensure_ascii=False)}"
                yield ""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self._response = FakeResponse()

        def __enter__(self):
            return self._response

        def __exit__(self, *exc):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._stream = FakeStream()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, *args, **kwargs):
            return self._stream

    monkeypatch.setattr(httpx, "Client", FakeClient)


def test_chat_stream_truncates_long_tool_result(monkeypatch, capsys):
    """验证 _chat_stream 打印 tool_result 时走 _truncate。"""
    long_content = "x" * 3000
    events = [
        ("tool_call", {"id": "c1", "name": "calculator", "arguments": {"expression": "(3+5)*7"}}),
        ("tool_result", {"tool_call_id": "c1", "name": "calculator", "content": long_content}),
        ("done", {"final_text": "结果如上"}),
    ]
    _mock_httpx_sse(monkeypatch, events)
    assert cli_main._chat_stream("http://test", {"message": "hi"}) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        f"calculator...\nInput: {json.dumps({'expression': '(3+5)*7'}, ensure_ascii=False)}\nOutput:\n"
    )
    assert f"{'x' * 500}" in out
    assert "已截断" in out
    assert "3000 字" in out


def test_chat_stream_keeps_short_tool_result(monkeypatch, capsys):
    events = [
        ("tool_call", {"id": "c1", "name": "calculator", "arguments": {"expression": "(3+5)*7"}}),
        ("tool_result", {"tool_call_id": "c1", "name": "calculator", "content": "56"}),
        ("done", {"final_text": "结果如上"}),
    ]
    _mock_httpx_sse(monkeypatch, events)
    assert cli_main._chat_stream("http://test", {"message": "hi"}) == 0
    out = capsys.readouterr().out
    assert "calculator...\nInput:" in out
    assert "Output:\n56" in out
    assert "已截断" not in out


def test_chat_stream_pairs_tool_call_with_matching_result(monkeypatch, capsys):
    """两个工具调用：结果乱序到达时也必须配到各自的 Input。"""
    events = [
        ("tool_call", {"id": "c1", "name": "calculator", "arguments": {"expression": "1+1"}}),
        ("tool_call", {"id": "c2", "name": "string_len", "arguments": {"text": "ab"}}),
        ("tool_result", {"tool_call_id": "c2", "name": "string_len", "content": "2"}),
        ("tool_result", {"tool_call_id": "c1", "name": "calculator", "content": "2"}),
        ("done", {"final_text": ""}),
    ]
    _mock_httpx_sse(monkeypatch, events)
    assert cli_main._chat_stream("http://test", {"message": "hi"}) == 0
    out = capsys.readouterr().out
    # string_len（c2）结果先到，先打印；各自 Input 必须与自己的 Output 配对
    assert out.index("string_len...") < out.index("calculator...")
    assert '"text": "ab"' in out.split("string_len...")[1].split("\n")[1]
    assert '"expression": "1+1"' in out.split("calculator...")[1].split("\n")[1]
    # 每个块的 Output 紧跟各自的 Input 之后再出现（经 Output: 行）
    assert "Output:\n2\n\ncalculator..." in out


def test_chat_stream_thinking_then_answer(monkeypatch, capsys):
    """thinking token 段后紧跟 answer 段。"""
    events = [
        ("thinking_token", {"text": "想"}),
        ("thinking_token", {"text": "一想"}),
        ("llm_token", {"text": "答"}),
        ("llm_token", {"text": "案"}),
        ("done", {"final_text": "答案"}),
    ]
    _mock_httpx_sse(monkeypatch, events)
    assert cli_main._chat_stream("http://test", {"message": "hi"}) == 0
    out = capsys.readouterr().out
    assert out.startswith("Thinking...")
    assert out.index("Thinking...") < out.index("Answer...")
    assert "想一想" in out
    assert "答案" in out


def test_chat_stream_result_after_tool_round_resumes_answer(monkeypatch, capsys):
    """工具轮之后新一轮 answer 仍打印标题。"""
    events = [
        ("thinking_token", {"text": "想"}),
        ("tool_call", {"id": "c1", "name": "calculator", "arguments": {"expression": "1+1"}}),
        ("tool_result", {"tool_call_id": "c1", "name": "calculator", "content": "2"}),
        ("thinking_token", {"text": "再想"}),
        ("llm_token", {"text": "答案是 2"}),
        ("done", {"final_text": "答案是 2"}),
    ]
    _mock_httpx_sse(monkeypatch, events)
    assert cli_main._chat_stream("http://test", {"message": "hi"}) == 0
    out = capsys.readouterr().out
    # 每轮思考都打标题；思考后紧跟 Answer...
    assert out.count("Thinking...") == 2
    assert out.count("Answer...") == 1


def test_truncate_short_content_unchanged():
    assert _truncate("short", 500) == "short"


def test_truncate_long_content_marks_and_counts():
    content = "x" * 600
    result = _truncate(content, 500)
    assert result.startswith("x" * 500)
    assert "已截断" in result
    assert "600 字" in result


def test_truncate_zero_disables():
    assert _truncate("y" * 1000, 0) == "y" * 1000


def test_main_passes_max_output_chars_arg(monkeypatch):
    """--max-output-chars 参数传到 _chat_stream（一次性模式）。"""
    seen = {}

    def fake_stream(base_url: str, payload: dict[str, str], *, max_output_chars: int) -> int:
        seen["max_output_chars"] = max_output_chars
        return 0

    monkeypatch.setattr(cli_main, "_chat_stream", fake_stream)
    assert main(["chat", "--msg", "hi", "--stream", "--max-output-chars", "123"]) == 0
    assert seen["max_output_chars"] == 123


def test_cli_interrupt_keeps_pending_in_script_mode(monkeypatch, capsys):
    monkeypatch.setattr(cli_main.sys.stdin, 'isatty', lambda: False)
    _mock_httpx_sse(monkeypatch, [('interrupt', {'agent_id': 'a', 'thread_id': 't', 'interrupts': [{'id': 'i', 'value': {'question': '审批'}}]})])
    assert cli_main._chat_stream('http://test', {'message': '问'}) == 3
    assert '--resume' in capsys.readouterr().out


def test_cli_resume_sends_answers(monkeypatch):
    sent = []
    def stream(base_url, payload, **kwargs):
        sent.append((payload, kwargs['path']))
        return 0
    monkeypatch.setattr(cli_main, '_chat_stream', stream)
    assert main(['chat', '--resume', '--agent-id', 'a', '--thread-id', 't', '--answers', '{"i":"approve"}']) == 0
    assert sent[0][0]['answers'] == {'i': 'approve'}
    assert sent[0][1] == '/chat/resume/stream'


def test_cli_corrects_provisional_final_text(monkeypatch, capsys):
    _mock_httpx_sse(monkeypatch, [('llm_token', {'text': '原文'}), ('done', {'final_text': '审核后'})])
    assert cli_main._chat_stream('http://test', {'message': '问'}) == 0
    assert capsys.readouterr().out.endswith('最终回答：\n审核后\n')
