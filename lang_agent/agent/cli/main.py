"""agent 层 CLI：纯 API 客户端。

用法：
    python -m lang_agent.agent.cli serve [--host 127.0.0.1] [--port 8000]
    python -m lang_agent.agent.cli chat --msg "计算 (3+5)*7" [--stream] [--model ...] [--thread-id ...]
    python -m lang_agent.agent.cli chat                              # 交互模式（默认流式，多轮，自动拉起服务）

工具结果打印默认截断到 500 字符（--max-output-chars 可覆盖，≤0 不截断）。
"""
import argparse
import json
import socket
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from lang_agent.agent.orchestration.config import DEFAULT_HOST, DEFAULT_PORT

if TYPE_CHECKING:
    from uvicorn import Server

# 交互模式的小命令
_EXIT_COMMANDS = ("/exit", "/quit")
_HELP_TEXT = "  /exit、/quit 退出；Ctrl+C 同效果；直接输入消息开始对话"
DEFAULT_MAX_OUTPUT_CHARS = 500  # 工具结果打印截断上限（--max-output-chars 可覆盖）


def _build_payload(
    *,
    message: str,
    model: str | None,
    provider: str | None,
    protocol: str | None,
    thread_id: str | None,
) -> dict[str, str]:
    """组装 /chat 请求体：None 的字段不传，走服务端默认值。"""
    payload: dict[str, str] = {"message": message}
    if model:
        payload["model"] = model
    if provider:
        payload["provider"] = provider
    if protocol:
        payload["protocol"] = protocol
    if thread_id:
        payload["thread_id"] = thread_id
    return payload


def _print_error(body: str) -> int:
    print(f"❌ 请求失败: {body}", file=sys.stderr)
    return 1


def _truncate(content: str, max_chars: int) -> str:
    """展示截断：超长工具结果在 CLI 打印前缩到 max_chars 并标注总长。"""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}…（共 {len(content)} 字，已截断）"


def _http_ok(base_url: str, timeout: float = 1.0) -> bool:
    """服务端就绪探测：GET /openapi.json 返回 200 即认为可用。"""
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            return client.get("/openapi.json").status_code == 200
    except httpx.HTTPError:
        return False


def _find_free_port(host: str, port: int, scan: int) -> int | None:
    """从 port 起扫描最多 scan 个端口，返回第一个能 bind 的空闲端口。"""
    for candidate in range(port, port + scan):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return None


def _ensure_server(
    base_url: str,
    *,
    timeout: float = 10.0,
    port_scan_range: int = 10,
) -> tuple[bool, str | None, "Server | None"]:
    """确保 base_url 有可用的 lang-agent 服务。返回 (ok, 实际地址, 自启实例)。

    外部已有服务 → 直接使用（不接管、退出时不关闭）；否则自动在后台拉起
    uvicorn——端口被占用时从 port+1 起扫描空闲端口，交互全程用实际地址。
    """
    if _http_ok(base_url):
        return True, base_url, None
    parsed = urlparse(base_url)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or DEFAULT_PORT
    free_port = _find_free_port(host, port, port_scan_range)
    if free_port is None:
        print(
            f"❌ 端口 {port}~{port + port_scan_range - 1} 均被占用，无法自动拉起服务；"
            f"请先运行: python -m lang_agent.agent.cli serve",
            file=sys.stderr,
        )
        return False, None, None
    from uvicorn import Config, Server  # 惰性导入：仅自动拉起时需要

    server = Server(
        Config("lang_agent.agent.server:app", host=host, port=free_port, log_level="warning")
    )
    if free_port != port:
        print(f"ℹ 端口 {port} 被占用，自动改用 http://{host}:{free_port} 启动服务")
    threading.Thread(target=server.run, daemon=True, name="lang-agent-serve").start()
    effective = f"http://{host}:{free_port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _http_ok(effective):
            return True, effective, server
        time.sleep(0.2)
    return False, None, server


def _chat(args) -> int:
    if args.message is None:
        return _interactive(args)
    payload = _build_payload(
        message=args.message,
        model=args.model,
        provider=args.provider,
        protocol=args.protocol,
        thread_id=args.thread_id,
    )
    if args.stream:
        return _chat_stream(args.base_url, payload, max_output_chars=args.max_output_chars)
    try:
        with httpx.Client(base_url=args.base_url, timeout=120) as client:
            resp = client.post("/chat", json=payload)
    except httpx.HTTPError as exc:
        print(f"❌ 无法连接 {args.base_url}: {exc}", file=sys.stderr)
        return 1
    if resp.status_code != 200:
        return _print_error(resp.text)
    data = resp.json()
    for tool_call in data.get("tool_calls", []):
        print(f"⚙ {tool_call.get('name')}({json.dumps(tool_call.get('arguments'), ensure_ascii=False)})")
    print(data["answer"])
    return 0


def _chat_stream(base_url: str, payload: dict[str, str], *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> int:
    """SSE 流式打印：Thinking/Answer 打字机 + 工具块（同一 tool_call_id 的 Input/Output 配对）。

    标题按段打印：thinking 段 → “Thinking...” 标题；工具轮之后新一轮 content
    → “Answer...” 标题；工具块无标题与内容归属问题（按 tool_call_id 配对打印）。
    """
    current: str | None = None  # 当前正在输出的段：thinking / answer / None
    saw_any = False  # 是否已输出任何内容（done 时决定是否补 final_text）
    # 工具配对：tool_call → 缓存，同 id 的 tool_result 到达时一起打印
    pending_tools: dict[str, dict[str, Any]] = {}
    try:
        with (
            httpx.Client(base_url=base_url, timeout=None) as client,
            client.stream("POST", "/chat/stream", json=payload) as resp,
        ):
            if resp.status_code != 200:
                return _print_error(resp.read().decode("utf-8", errors="replace"))
            event_type = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_type == "thinking_token":
                        if current == "answer" or current is None:
                            # 段首（含工具轮后新一轮思考）打标题；与前段分隔
                            if saw_any:
                                print()
                            print("Thinking...")
                        current = "thinking"
                        saw_any = True
                        print(data["text"], end="", flush=True)
                    elif event_type == "llm_token":
                        if current != "answer":
                            # thinking 段结束后紧接 answer；若从思考段切换，标题紧跟输出
                            if current == "thinking" or saw_any:
                                print()
                            print("Answer...")
                        current = "answer"
                        saw_any = True
                        print(data["text"], end="", flush=True)
                    elif event_type == "tool_call":
                        pending_tools[data["id"]] = data
                    elif event_type == "tool_result":
                        call = pending_tools.pop(data["tool_call_id"], None)
                        name = call["name"] if call else data.get("name")
                        args = call["arguments"] if call else None
                        if saw_any:
                            print()
                        print(f"{name}...")
                        if args is not None:
                            print(f"Input: {json.dumps(args, ensure_ascii=False)}")
                        print("Output:")
                        print(_truncate(data["content"], max_output_chars))
                        current = None
                        saw_any = True
                    elif event_type == "done":
                        if not saw_any:
                            print(data["final_text"])
                        else:
                            print()
                        return 0
                    elif event_type == "error":
                        print(f"\n❌ {data['message']}", file=sys.stderr)
                        return 1
    except httpx.HTTPError as exc:
        print(f"❌ 无法连接 {base_url}: {exc}", file=sys.stderr)
        return 1
    return 1


def _interactive(args) -> int:
    """交互模式：chat 不带 --msg 时进入，默认流式，多轮共用同一 thread_id。"""
    ok, base_url, server = _ensure_server(args.base_url)
    if not ok or base_url is None:
        return 1
    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    try:
        print(f"🤖 lang-agent 交互模式（{base_url}，会话 {thread_id}，/help 查看命令）")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            cmd = text.strip()
            if not cmd:
                continue
            if cmd in _EXIT_COMMANDS:
                break
            if cmd == "/help":
                print(_HELP_TEXT)
                continue
            if cmd.startswith("/"):
                print(f"未知命令 {cmd}（/help 查看）")
                continue
            payload = _build_payload(
                message=cmd,
                model=args.model,
                provider=args.provider,
                protocol=args.protocol,
                thread_id=thread_id,
            )
            try:
                ret = _chat_stream(base_url, payload, max_output_chars=args.max_output_chars)
                if ret != 0:
                    return ret
            except KeyboardInterrupt:
                print("\n（已中断本次生成）")
    finally:
        if server is not None:
            server.should_exit = True  # 关闭自启的服务；外部服务不动
    return 0


def _serve(args) -> int:
    import uvicorn

    uvicorn.run("lang_agent.agent.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lang-agent", description="lang-agent CLI")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="启动 FastAPI 服务")
    serve_p.add_argument("--host", default=DEFAULT_HOST)
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT)

    chat_p = sub.add_parser("chat", help="调用本地 API 对话（不带 --msg 进入交互模式）")
    chat_p.add_argument("--msg", "--message", dest="message", default=None, help="用户消息（不带则进入交互模式；交互默认流式输出）")
    chat_p.add_argument("--stream", action="store_true", help="流式输出（SSE）；交互模式恒为流式")
    chat_p.add_argument("--model", default=None, help="模型名，默认服务端配置")
    chat_p.add_argument("--provider", default=None, help="provider，默认服务端配置")
    chat_p.add_argument("--protocol", default=None, help="protocol，默认服务端配置")
    chat_p.add_argument("--thread-id", default=None, help="多轮对话线程 id（交互模式缺省自动生成 cli-xxxx）")
    chat_p.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS, dest="max_output_chars", help="工具结果展示截断上限（≤0 不截断）")
    chat_p.add_argument("--base-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "chat":
        return _chat(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
