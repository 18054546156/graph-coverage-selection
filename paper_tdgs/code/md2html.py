#!/usr/bin/env python
"""Minimal Markdown -> standalone HTML (stdlib only; enough for the report/handoff/proposal files).

Supports headings, paragraphs, hr, blockquotes (recursive), nested -, * and 1. lists, pipe tables,
fenced code, inline code, **bold**, [text](url).  Repo-relative links/paths in backticks stay as text.
Usage: python code/md2html.py IN.md OUT.html ["title"]
"""
import html
import re
import sys

CSS = """
body{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;max-width:1100px;margin:2em auto;
padding:0 1.2em;line-height:1.65;color:#1f2328}
h1{border-bottom:2px solid #d0d7de;padding-bottom:.3em}h2{border-bottom:1px solid #d0d7de;padding-bottom:.25em;margin-top:2em}
table{border-collapse:collapse;margin:1em 0;font-size:.92em;display:block;overflow-x:auto}
th,td{border:1px solid #d0d7de;padding:.35em .6em;vertical-align:top}th{background:#f6f8fa}tr:nth-child(even) td{background:#fbfcfd}
code{background:#f3f4f6;padding:.1em .35em;border-radius:4px;font-size:.88em}
pre{background:#f6f8fa;padding:1em;overflow-x:auto;border-radius:6px}pre code{background:none;padding:0}
blockquote{margin:1em 0;padding:.5em 1em;border-left:4px solid #8aa4c8;background:#f4f7fb;color:#33475b}
hr{border:none;border-top:1px solid #d0d7de;margin:2em 0}a{color:#0969da}
"""


def inline(s):
    codes = []

    def stash(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    p = re.sub(r"`([^`]*)`", stash, s)
    p = html.escape(p, quote=False).replace("\\*", "&#42;").replace("\\|", "|")
    p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
    p = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', p)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], p)


def split_row(line):
    line = line.strip().strip("|")
    cells, cur, i = [], "", 0
    while i < len(line):                       # split on | outside backticks, keep \| literal
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "\\|"; i += 2; continue
        if ch == "`":
            j = line.find("`", i + 1)
            if j > i:
                cur += line[i:j + 1]; i = j + 1; continue
        if ch == "|":
            cells.append(cur.strip()); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur.strip())
    return cells


LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def convert(lines):
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1; continue
        if s.startswith("```"):
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            k = len(m.group(1))
            out.append(f"<h{k}>{inline(m.group(2))}</h{k}>"); i += 1; continue
        if re.match(r"^(-{3,}|\*{3,})$", s):
            out.append("<hr>"); i += 1; continue
        if s.startswith(">"):
            j, buf = i, []
            while j < n and lines[j].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[j])); j += 1
            out.append("<blockquote>" + convert(buf) + "</blockquote>")
            i = j; continue
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            head = split_row(lines[i])
            rows, j = [], i + 2
            while j < n and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j])); j += 1
            t = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</tbody></table>")
            i = j; continue
        if LIST_RE.match(line):
            stack, j = [], i
            while j < n:
                lm = LIST_RE.match(lines[j])
                if not lm:
                    if lines[j].strip() == "" and j + 1 < n and LIST_RE.match(lines[j + 1]):
                        j += 1; continue
                    if lines[j].strip() and lines[j].startswith("  ") and stack:   # continuation line
                        out.append(" " + inline(lines[j].strip())); j += 1; continue
                    break
                ind, mk, txt = len(lm.group(1)), lm.group(2), lm.group(3)
                tag = "ol" if mk[0].isdigit() else "ul"
                while stack and ind < stack[-1][0]:
                    out.append(f"</{stack.pop()[1]}>")
                if not stack or ind > stack[-1][0]:
                    stack.append((ind, tag)); out.append(f"<{tag}>")
                elif stack[-1][1] != tag:
                    out.append(f"</{stack.pop()[1]}>"); stack.append((ind, tag)); out.append(f"<{tag}>")
                out.append(f"<li>{inline(txt)}</li>")
                j += 1
            while stack:
                out.append(f"</{stack.pop()[1]}>")
            i = j; continue
        j, buf = i, []
        while j < n and lines[j].strip() and not re.match(r"^\s*(#|>|\||```|([-*]|\d+\.)\s)", lines[j]) \
                and not re.match(r"^(-{3,})$", lines[j].strip()):
            buf.append(lines[j].strip()); j += 1
        if not buf:
            buf, j = [s], i + 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")
        i = j
    return "\n".join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    lines = open(src, encoding="utf-8").read().splitlines()
    title = sys.argv[3] if len(sys.argv) > 3 else next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), src)
    body = convert(lines)
    doc = (f'<!DOCTYPE html>\n<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" '
           f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style>'
           f'</head><body>\n<p style="color:#666;font-size:.85em">Generated from <code>{html.escape(src)}</code> '
           f'by <code>code/md2html.py</code>; the Markdown file is the source of truth.</p>\n{body}\n</body></html>\n')
    open(dst, "w", encoding="utf-8").write(doc)


if __name__ == "__main__":
    main()
