#!/usr/bin/env python3
"""DL-Math 批量课程生成器
逐节逐个文件调用 DeepSeek API 生成 markdown，每次只生成 1 个文件。
中断后重跑自动跳过已生成文件。和 web 阅读器使用完全相同的 prompt。
"""

import json, os, re, sys, time, random, argparse, subprocess
from datetime import date
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装: pip install requests")
    sys.exit(1)

# ── CONFIG ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LESSONS_DIR = PROJECT_ROOT / "lessons"
MANIFEST_PATH = PROJECT_ROOT / "manifest.json"
CLAUDE_MD_PATH = PROJECT_ROOT / "CLAUDE.md"
DECOMP_DIR = PROJECT_ROOT / "decomposition" / "chapters"

GEN_ORDER = ["主线", "历史", "代码", "习题", "镜鉴", "参考资料"]

FILE_INSTRUCTIONS = {
    "历史": '请生成【历史.md】。按照 CLAUDE.md 中"历史.md 结构"的写作纪律：画面感优先、悬念驱动、不引入技术符号、择一历史源流深入（数学史或DL史），结尾留一扇门引向主线。篇幅 5-8 min 阅读量。',
    "主线": '请生成【主线.md】。按照 CLAUDE.md 中"主线.md 三步结构"：(1) Toy Model——纯直觉模型，不写代码，用几何画面/物理类比/微型手算让读者先感受概念；(2) 正式数学讲解——动机先行，每个定义出现前先问"为什么需要它"，推导优于宣告，至少一处边界声明；(3) 概念性DL桥接——不写代码，只用数学符号描述概念在DL中的位置，结尾指向代码.md。',
    "代码": '请生成【代码.md】。按照 CLAUDE.md 中"代码.md 结构"：先完整可运行脚本（20-40行，透明数学操作，无黑盒API），然后分步逐函数拆解。每个函数按"做什么/操作/输入/输出/可调参数/对应数学"格式，每个参数讲透含义和取值范围。',
    "习题": '请生成【习题.md】。按照 CLAUDE.md 中"习题.md 结构"：概念检验（2-3题）、计算实践（2-3题，含代码验证）、DL延伸（1-2题）。每题附完整解答和思路点拨。',
    "镜鉴": '请生成【镜鉴.md】。按照 CLAUDE.md 中"镜鉴.md 结构"：数学之美、DL中的意义、人的映射、开放问题。200-400字，精炼有力，具体到概念本身的结构。',
    "参考资料": '请生成【参考资料.md】。按照 CLAUDE.md 中"参考资料.md 结构"：教材出处、延伸论文、代码库/工具、推荐阅读。按本节内容灵活选择，不为凑字数而填满。',
}

# ── FOLDER NAME ──────────────────────────────────────
def compute_folder_name(ch_num, sec_id, title):
    """复刻阅读器的文件夹命名公式"""
    slug = re.sub(r'\s+', '-', title)
    slug = re.sub(r'[^a-zA-Z0-9\-]', '', slug)[:30]
    return f"第{ch_num}节-{sec_id}-{slug}"


# ── PROMPT ──────────────────────────────────────────
def load_claude_md():
    return CLAUDE_MD_PATH.read_text(encoding="utf-8")


def build_system_prompt(claude_md):
    return (
        "你是一位深度学习数学导师。请严格遵循以下教学指令：\n\n"
        + claude_md
        + "\n\n---\n\n请只输出最终的课程内容（Markdown格式），不要包含任何元对话或解释。"
    )


def build_user_prompt(ch, sec, file_type, sec_text):
    instruction = FILE_INSTRUCTIONS[file_type]
    prompt = (
        f"{instruction}\n\n"
        f"- 章节：Ch{ch['num']} §{sec['id']} {sec['title']}\n"
        f"- 文件类型：{file_type}\n"
    )
    if sec_text:
        if file_type == "历史":
            prompt += (
                "\n以下是教材原文参考（了解本节涉及的数学概念，但历史叙事不要复述教材——"
                '去找这个概念「必须被发明」的时刻）：\n\n' + sec_text[:30000] + '\n'
            )
        else:
            prompt += (
                "\n以下是教材原文参考（请基于此内容生成，逐一覆盖每个关键定义、定理和示例，不得跳过）：\n\n"
                + sec_text[:50000] + "\n"
            )
    prompt += "\n请开始生成。"
    return prompt


def read_sec_text(sec):
    """读取教材原文"""
    path = sec.get("path", "")
    if not path:
        return ""
    full = PROJECT_ROOT / path
    if full.exists():
        return full.read_text(encoding="utf-8", errors="replace")
    return ""


# ── API ─────────────────────────────────────────────
def call_deepseek(api_key, system_prompt, user_prompt, stream=True, max_tokens=16384):
    """调用 DeepSeek API，返回 (content, truncated)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    if stream:
        return _call_stream(headers, body)
    else:
        return _call_sync(headers, body)


def _call_sync(headers, body):
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=(30, 600),
    )
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    truncated = data["choices"][0].get("finish_reason") == "length"
    return content, truncated


def _call_stream(headers, body):
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=body,
        stream=True,
        timeout=(30, 600),
    )
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")

    content = ""
    finish_reason = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                content += delta["content"]
            if "finish_reason" in chunk.get("choices", [{}])[0]:
                finish_reason = chunk["choices"][0]["finish_reason"]
        except json.JSONDecodeError:
            continue

    truncated = finish_reason == "length"
    return content, truncated


# ── RETRY ───────────────────────────────────────────
def call_with_retry(api_key, system_prompt, user_prompt, max_retries=5):
    """带重试的 API 调用"""
    for attempt in range(max_retries):
        try:
            content, truncated = call_deepseek(api_key, system_prompt, user_prompt)
            if truncated and attempt < 1:  # 截断重试一次
                time.sleep(2)
                continue
            return content, truncated
        except RuntimeError as e:
            msg = str(e)
            if "401" in msg or "402" in msg or "403" in msg:
                raise  # 不重试认证/余额错误
            if attempt < max_retries - 1:
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"  ⚠ 重试 {attempt+1}/{max_retries}，等待 {wait:.0f}s: {msg[:100]}")
                time.sleep(wait)
            else:
                raise

# ── SAVE ────────────────────────────────────────────
def save_file(folder, file_type, content):
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{file_type}.md"
    tmp = folder / f"{file_type}.md.tmp"
    tmp.write_text(content, encoding="utf-8")
    os.replace(str(tmp), str(filepath))
    return filepath


# ── GIT ─────────────────────────────────────────────
def git_commit_push(filepath, ch, sec, file_type):
    try:
        subprocess.run(
            ["git", "add", str(filepath)],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=10,
        )
        msg = f"📱 {file_type} — Ch{ch['num']} §{sec['id']} {sec['title'][:40]}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
        return True
    except Exception as e:
        print(f"  ⚠ Git 失败: {e}")
        return False


# ── SCAN ────────────────────────────────────────────
def scan_todos(manifest):
    todo = []
    for ch in manifest["chapters"]:
        for sec in ch["sections"]:
            folder_name = compute_folder_name(ch["num"], sec["id"], sec["title"])
            folder = LESSONS_DIR / folder_name
            for ft in GEN_ORDER:
                if not (folder / f"{ft}.md").exists():
                    todo.append((ch, sec, ft, folder))
    return todo


# ── MAIN ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DL-Math 批量课程生成器")
    parser.add_argument("--api-key", help="DeepSeek API Key")
    parser.add_argument("--key-file", help="包含 API Key 的文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只列出待生成文件，不调 API")
    parser.add_argument("--chapter", help="只处理指定章 (如 04)")
    parser.add_argument("--section", help="只处理指定节 (如 4.2.1)")
    parser.add_argument("--file", choices=GEN_ORDER, help="只生成指定文件类型")
    parser.add_argument("--limit", type=int, help="最多生成 N 个文件后停止")
    parser.add_argument("--delay", type=float, default=2.0, help="文件间延迟秒数 (默认 2s)")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--no-push", action="store_true", help="生成文件但不 git push")
    parser.add_argument("--frontmatter", action="store_true", help="给文件加 YAML 头")
    args = parser.parse_args()

    # ── API Key ──
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and args.key_file:
        api_key = Path(args.key_file).read_text().strip()
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        except ImportError:
            pass
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量，或用 --api-key / --key-file")
        sys.exit(1)

    # ── Load ──
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    claude_md = load_claude_md()
    system_prompt = build_system_prompt(claude_md)

    # ── Filter ──
    if args.chapter:
        manifest["chapters"] = [c for c in manifest["chapters"] if c["num"] == args.chapter]
    if args.section:
        for c in manifest["chapters"]:
            c["sections"] = [s for s in c["sections"] if s["id"] == args.section]

    # ── Scan ──
    stream = not args.no_stream
    todo = scan_todos(manifest)
    if args.file:
        todo = [(ch, sec, ft, f) for ch, sec, ft, f in todo if ft == args.file]

    total_files = sum(1 for _ in todo)  # use generator once
    todo = scan_todos(manifest)  # re-scan
    if args.file:
        todo = [(ch, sec, ft, f) for ch, sec, ft, f in todo if ft == args.file]

    print(f"📋 待生成: {len(todo)} 文件")
    print(f"📋 genOrder: {' → '.join(GEN_ORDER)}")

    if args.dry_run:
        for ch, sec, ft, folder in todo[:30]:
            print(f"  [{ft}] Ch{ch['num']} §{sec['id']} → {folder.relative_to(PROJECT_ROOT)}/{ft}.md")
        if len(todo) > 30:
            print(f"  ... 共 {len(todo)} 个文件")
        return

    # ── Generate ──
    done = 0
    for ch, sec, ft, folder in todo:
        label = f"Ch{ch['num']} §{sec['id']} [{ft}]"
        print(f"\n{'─'*60}")
        print(f"🔄 {label}")

        sec_text = read_sec_text(sec)
        user_prompt = build_user_prompt(ch, sec, ft, sec_text)

        try:
            content, truncated = call_with_retry(api_key, system_prompt, user_prompt)
        except RuntimeError as e:
            print(f"  ❌ 失败: {e}")
            continue

        if args.frontmatter:
            today = date.today().isoformat()
            title_suffix = ""
            yaml_title = sec["title"]
            if ft == "代码":
                yaml_title += " — 代码验证"
            fm = (
                f"---\nchapter: {ch['num']}\nsection: {sec['id']}\n"
                f"title: {yaml_title}\n"
                f"textbook: Mathematical Foundations for Deep Learning (Ghayoumi, 2026)\n"
                f"textbook_section: §{sec['id']} {sec['title']}\ndate: {today}\n---\n\n"
            )
            content = fm + content

        filepath = save_file(folder, ft, content)
        status = "⚠截断" if truncated else f"✅{len(content)}字"
        print(f"  {status} → {filepath.relative_to(PROJECT_ROOT)}")

        if not args.no_push:
            git_commit_push(filepath, ch, sec, ft)

        done += 1
        if args.limit and done >= args.limit:
            print(f"\n🛑 已达上限 {args.limit}，停止")
            break

        time.sleep(args.delay)

    print(f"\n✨ 完成：生成 {done} 个文件")


if __name__ == "__main__":
    main()
