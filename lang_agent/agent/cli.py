"""agent 层 CLI：纯 API 客户端。

用法：
    python -m lang_agent.agent.cli serve [--host 127.0.0.1] [--port 8000]
    python -m lang_agent.agent.cli chat --msg "计算 (3+5)*7" [--stream] [--model ...] [--thread-id ...]
"""
import argparse
import json
import sys

import httpx

from lang_agent.agent.config import DEFAULT_HOST, DEFAULT_PORT


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


def _chat(args) -> int:
    payload = _build_payload(
        message=args.message,
        model=args.model,
        provider=args.provider,
        protocol=args.protocol,
        thread_id=args.thread_id,
    )
    if args.stream:
        return _chat_stream(args.base_url, payload)
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


def _chat_stream(base_url: str, payload: dict[str, str]) -> int:
    """SSE 流式打印：llm_token 实时输出，tool 过程灰显，error 走 stderr 退出码 1。"""
    saw_token = False
    try:
        with httpx.Client(base_url=base_url, timeout=None) as client:
            with client.stream("POST", "/chat/stream", json=payload) as resp:
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
                        if event_type == "llm_token":
                            saw_token = True
                            print(data["text"], end="", flush=True)
                        elif event_type == "tool_call":
                            print(f"\n⚙ {data['name']}({json.dumps(data['arguments'], ensure_ascii=False)})")
                        elif event_type == "tool_result":
                            print(f"  → {data['content']}")
                        elif event_type == "done":
                            if not saw_token:
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

    chat_p = sub.add_parser("chat", help="调用本地 API 对话")
    chat_p.add_argument("--msg", "--message", required=True, dest="message", help="用户消息")
    chat_p.add_argument("--stream", action="store_true", help="流式输出（SSE）")
    chat_p.add_argument("--model", default=None, help="模型名，默认服务端配置")
    chat_p.add_argument("--provider", default=None, help="provider，默认服务端配置")
    chat_p.add_argument("--protocol", default=None, help="protocol，默认服务端配置")
    chat_p.add_argument("--thread-id", default=None, help="多轮对话线程 id")
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
