# -*- coding: utf-8 -*-
"""把一篇 Obsidian 笔记导出成可以直接发给别人的单文件 HTML + PDF。

    python export_note.py <笔记.md 或 文件名片段> [更多笔记...] [--html-only] [--keep-internal]

要点：
  * 图片以 base64 内嵌，单文件离线可读，不依赖 assets/ 目录；
  * Obsidian 短链 ![[x.png]] 按「全库按文件名检索」的语义解析，和 Obsidian 一致；
  * 默认摘掉 See Also / 备注 这类仓库内务小节（--keep-internal 保留）；
  * 只读源笔记，只往 <vault>/_export/ 写产物，不碰 index.md / log.md / raw/。

退出码：0 全部成功 | 1 笔记没找到或不唯一 | 2 有缺图（产物仍已生成）
        3 没有可用浏览器，PDF 跳过（HTML 仍已生成）| 4 缺 mistune 依赖
        5 源笔记有落单的反引号 / 代码围栏（产物仍已生成，但后半篇会错位）
"""
import base64, html, mimetypes, os, re, subprocess, sys, tempfile

# Windows 控制台默认 GBK，中文路径会输出成乱码；强制 UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

IMG_EXT ={'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.avif'}
SKIP_DIRS = {'.git', '.obsidian', '.claude', '_export', 'node_modules', '.trash'}

try:
    import mistune
except ImportError:
    sys.stderr.write('缺依赖：pip install mistune\n')
    sys.exit(4)


# ---------------------------------------------------------------- vault / 定位

def find_vault(start):
    d = os.path.abspath(start if os.path.isdir(start) else os.path.dirname(start) or '.')
    while True:
        if os.path.isdir(os.path.join(d, '.obsidian')) or os.path.isdir(os.path.join(d, 'wiki')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath('.')
        d = parent


def build_index(vault):
    """文件名 -> 绝对路径。对应 Obsidian shortest-path 短链的全库检索语义。"""
    idx = {}
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            idx.setdefault(f, os.path.join(root, f))
    return idx


def resolve_note(arg, vault):
    """接受完整路径，也接受「投机采样」这样的文件名片段。"""
    if os.path.isfile(arg):
        return [os.path.abspath(arg)]
    frag = os.path.splitext(os.path.basename(arg))[0].lower()
    root_dir = os.path.join(vault, 'wiki')
    if not os.path.isdir(root_dir):
        root_dir = vault
    hits = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.lower().endswith('.md') and frag in f.lower():
                hits.append(os.path.join(root, f))
    exact = [h for h in hits if os.path.splitext(os.path.basename(h))[0].lower() == frag]
    return exact or hits


# ---------------------------------------------------------------- markdown 加工

def fence_blocks(lines):
    """按 CommonMark 规则切出代码围栏区间：[(起行, 止行 or None, 围栏字符, 长度), ...]，行号 1 起。

    关键细节：带 info string 的 ```bash **不能**充当收尾围栏，收尾必须是纯围栏字符。
    搞错这一点，遇到落单围栏就会整篇错位——该跳过的当正文处理，该处理的当代码跳过。
    """
    blocks, open_at = [], None
    for n, line in enumerate(lines, 1):
        if open_at is None:
            m = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
            if m:
                open_at = (n, m.group(1)[0], len(m.group(1)))
        else:
            n0, ch, ln = open_at
            if re.match(r'^ {0,3}%s{%d,}\s*$' % (re.escape(ch), ln), line):
                blocks.append((n0, n, ch, ln))
                open_at = None
    if open_at is not None:
        blocks.append((open_at[0], None, open_at[1], open_at[2]))
    return blocks


def fence_state(lines):
    """每行是否落在代码围栏内（围栏行本身算在内）。第二个返回值：有围栏没收尾。"""
    blocks = fence_blocks(lines)
    state = [False] * len(lines)
    for n0, n1, _, _ in blocks:
        for i in range(n0 - 1, n1 if n1 else len(lines)):
            state[i] = True
    return state, any(n1 is None for _, n1, _, _ in blocks)


# 只对 vault 内部有意义的小节，导出时默认摘掉（--keep-internal 可保留）
INTERNAL_SECTIONS = {'see also', 'see-also', '备注'}


def strip_internal(src):
    """摘掉 See Also / 备注 等仓库内务小节，返回 (正文, 摘掉了哪些)。

    See Also 指向的是对方拿不到的笔记，备注记的是本库的加工历史 —— 单独发出去
    只会干扰阅读。按标题级别界定范围：遇到同级或更高级标题即结束。
    """
    lines = src.split('\n')
    state, _ = fence_state(lines)
    out, removed, skip_level = [], [], None
    for line, in_code in zip(lines, state):
        if not in_code:
            h = re.match(r'^(#{1,6})\s+(.*?)\s*$', line)
            if h:
                level = len(h.group(1))
                title = h.group(2).strip().rstrip(':：').lower()
                if skip_level is not None and level <= skip_level:
                    skip_level = None             # 上一个待摘小节到此为止
                if skip_level is None and title in INTERNAL_SECTIONS:
                    skip_level = level
                    removed.append(h.group(2).strip())
                    continue
        if skip_level is None:
            out.append(line)
    while out and not out[-1].strip():             # 收尾的空行 / 残留分隔线
        out.pop()
    if out and re.match(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$', out[-1]):
        out.pop()
    return '\n'.join(out) + '\n', removed


def cjk_emphasis(src):
    """CommonMark 不认「中文标点 + ** + 中文字」这种收尾（Obsidian 宽松），
    会把粗体漏成裸星号。这里自行解析成 <strong>，代码块/行内代码原样跳过。"""
    lines = src.split('\n')
    state, _ = fence_state(lines)
    out = []
    for line, in_code in zip(lines, state):
        if in_code:
            out.append(line)
            continue
        codes = []

        def stash(m):
            codes.append(m.group(0))
            return '\x00%d\x00' % (len(codes) - 1)

        seg = re.sub(r'`[^`]+`', stash, line)
        seg = re.sub(r'\*\*(?=\S)(.+?)(?<=\S)\*\*',
                     lambda m: '<strong>%s</strong>' % m.group(1), seg)
        out.append(re.sub(r'\x00(\d+)\x00', lambda m: codes[int(m.group(1))], seg))
    return '\n'.join(out)


CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#6b7280;--line:#e5e7eb;--code-bg:#f6f8fa;--quote:#0969da;--accent:#0969da}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--code-bg:#161b22;--quote:#4493f8;--accent:#4493f8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
 -webkit-text-size-adjust:100%}
.wrap{max-width:860px;margin:0 auto;padding:40px 20px 80px}
h1,h2,h3,h4{line-height:1.3;margin:2em 0 .7em;font-weight:650}
h1{font-size:1.9em;margin-top:0;padding-bottom:.4em;border-bottom:1px solid var(--line)}
h2{font-size:1.45em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.2em} h4{font-size:1.05em}
p,ul,ol{margin:.9em 0} li{margin:.3em 0}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
img{max-width:100%;height:auto;display:block;margin:1.2em auto;border-radius:6px}
td img{margin:.3em 0;max-width:260px}
blockquote{margin:1.2em 0;padding:.6em 1em;border-left:4px solid var(--quote);
 background:var(--code-bg);border-radius:0 6px 6px 0}
blockquote > :first-child{margin-top:0} blockquote > :last-child{margin-bottom:0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Courier New",monospace;font-size:.88em;
 background:var(--code-bg);padding:.15em .4em;border-radius:4px}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow-x:auto;line-height:1.55}
pre code{background:none;padding:0;font-size:.85em}
.tablewrap{overflow-x:auto;margin:1.2em 0}
table{border-collapse:collapse;width:100%;font-size:.92em}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top}
th{background:var(--code-bg);font-weight:650}
hr{border:0;border-top:1px solid var(--line);margin:2.5em 0}
.wikilink{color:var(--muted);border-bottom:1px dotted var(--muted)}
.missing-img{color:#b42318;font-size:.9em}
.foot{margin-top:4em;padding-top:1.2em;border-top:1px solid var(--line);color:var(--muted);font-size:.85em}
@media print{body{background:#fff;color:#000}.wrap{max-width:none;padding:0}
 pre,blockquote,th{background:#f6f8fa}h2,h3{break-after:avoid}img,pre,table{break-inside:avoid}}
@media (max-width:600px){.wrap{padding:24px 16px 60px}body{font-size:15px}}
"""

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title><style>%s</style></head>
<body><div class="wrap">%s
<div class="foot">由个人 Wiki 笔记导出 · 图片已内嵌，单文件离线可读</div>
</div></body></html>"""


# ---------------------------------------------------------------- 源笔记体检

def lint_source(src):
    """导出前的确定性体检，只报告不修改（源笔记归用户）。返回 [(行号, 说明, 该行), ...]。

    三类问题有同一个恶劣特征：**坏的是后半篇**，扫一眼开头完全看不出来。
      * 反引号落单 —— 之后所有行内代码的配对整体错开一位，正文出现 </code>xxx<code>
        这种倒转。常见成因是收尾打成了中文引号 ‘ ’ 或 '。
      * 围栏没收尾 —— 剩下的半篇全被当成代码。
      * 围栏多余 —— 块内又冒出一道带语言标记的开围栏（```bash），说明上面那道 ``` 是
        落单的多余货，把本该是正文的几行一起吞成了代码块。注意这种情况下围栏**数量是
        偶数、也能正常配对**，只靠计数查不出来。
    """
    lines = src.split('\n')
    blocks = fence_blocks(lines)
    state, _ = fence_state(lines)
    bad = []
    for n, (line, in_code) in enumerate(zip(lines, state), 1):
        if not in_code and line.count('`') % 2:
            bad.append((n, '反引号落单，之后所有行内代码的配对会整体错开一位', line.strip()))
    for n0, n1, ch, ln in blocks:
        if n1 is None:
            bad.append((n0, '代码围栏没有收尾，后面整篇都被当成代码', lines[n0 - 1].strip()))
            continue
        inner = re.compile(r'^ {0,3}%s{%d,}\S' % (re.escape(ch), ln))
        if any(inner.match(lines[i]) for i in range(n0, n1 - 1)):
            bad.append((n0, '这道围栏是多余的，把第 %d–%d 行正文吞成了代码块' % (n0 + 1, n1 - 1),
                        lines[n0 - 1].strip()))
    bad.sort()
    return bad


def render(src, idx, keep_internal=False):
    text = open(src, encoding='utf-8').read()
    removed = []
    if not keep_internal:
        text, removed = strip_internal(text)
    images, missing = [], []

    def embed(m):
        name, width = m.group(1).strip(), m.group(2)
        if os.path.splitext(name)[1].lower() not in IMG_EXT:   # ![[另一篇笔记]] 之类的嵌入
            return '<span class="wikilink">%s</span>' % html.escape(name)
        path = idx.get(name)
        if not path:
            missing.append(name)
            return '<span class="missing-img">[缺图: %s]</span>' % html.escape(name)
        images.append(path)
        w = ' style="width:%spx"' % width if width else ''
        return '<img src="@@IMG%d@@" alt="%s"%s>' % (len(images) - 1, html.escape(name), w)

    text = re.sub(r'!\[\[([^\]|]+?)(?:\|(\d+))?\]\]', embed, text)
    # [[其它笔记]] —— 目标不在导出范围内，降级成纯文本而不是死链
    text = re.sub(r'\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]',
                  lambda m: '<span class="wikilink">%s</span>' % html.escape(m.group(2) or m.group(1)),
                  text)
    text = cjk_emphasis(text)

    md = mistune.create_markdown(
        escape=False,
        plugins=['table', 'strikethrough', 'url', 'footnotes', 'task_lists'])
    doc = PAGE % (html.escape(os.path.splitext(os.path.basename(src))[0]), CSS, md(text))
    doc = doc.replace('<table>', '<div class="tablewrap"><table>')
    doc = doc.replace('</table>', '</table></div>')

    # 最后再塞图，避免超长 base64 过一遍 markdown 解析器
    for i, path in enumerate(images):
        mime = mimetypes.guess_type(path)[0] or 'image/png'
        with open(path, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode('ascii')
        doc = doc.replace('@@IMG%d@@' % i, 'data:%s;base64,%s' % (mime, b64))
    return doc, len(images), missing, removed


# ---------------------------------------------------------------- PDF

def find_browser():
    env = os.environ
    cands = [
        env.get('BROWSER_FOR_PDF'),
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        os.path.join(env.get('LOCALAPPDATA', ''), r'Google\Chrome\Application\chrome.exe'),
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
        '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/microsoft-edge',
    ]
    return next((c for c in cands if c and os.path.isfile(c)), None)


def to_pdf(browser, html_path, pdf_path):
    """返回 (是否成功, 失败原因)。先删旧文件——否则渲染失败时，
    残留的上一次产物会被当成"成功"，用户拿到的是过期 PDF。"""
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            return False, '旧 PDF 删不掉（是不是正在别的程序里打开着？）'
    profile = tempfile.mkdtemp(prefix='noteexport-')
    tail = ['--disable-gpu', '--no-first-run', '--user-data-dir=' + profile,
            '--no-pdf-header-footer', '--print-to-pdf=' + pdf_path, html_path]
    err = '浏览器没能产出 PDF'
    for flag in ('--headless=new', '--headless'):
        try:
            subprocess.run([browser, flag] + tail, capture_output=True, timeout=300)
        except Exception as e:
            err = '调用浏览器失败：%s' % e
            continue
        if os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 1024:
            return True, ''
    return False, err


def pdf_pages(path):
    try:
        with open(path, 'rb') as fh:
            return len(re.findall(rb'/Type\s*/Page[^s]', fh.read()))
    except Exception:
        return 0


# ---------------------------------------------------------------- main

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = set(a for a in sys.argv[1:] if a.startswith('--'))
    if not args:
        sys.stderr.write(__doc__)
        return 1

    vault = find_vault(args[0] if os.path.exists(args[0]) else '.')
    outdir = os.path.join(vault, '_export')
    os.makedirs(outdir, exist_ok=True)
    idx = build_index(vault)

    notes = []
    for a in args:
        hits = resolve_note(a, vault)
        if len(hits) != 1:
            listing = '\n'.join('  ' + os.path.relpath(h, vault) for h in hits) or '  (无)'
            sys.stderr.write('「%s」%s：\n%s\n' % (
                a, '没找到对应笔记' if not hits else '匹配到多篇，请指明哪一篇', listing))
            return 1
        notes.append(hits[0])

    rc = 0
    browser = None if '--html-only' in flags else find_browser()
    print('vault    : %s' % vault)
    print('输出目录 : %s\n' % outdir)

    for src in notes:
        stem = os.path.splitext(os.path.basename(src))[0]
        html_path = os.path.join(outdir, stem + '.html')
        doc, n_img, missing, removed = render(src, idx, '--keep-internal' in flags)
        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(doc)
        print('== %s' % os.path.relpath(src, vault))
        print('   HTML  %.2f MB · 内嵌图 %d 张' % (os.path.getsize(html_path) / 1048576, n_img))
        if removed:
            print('   已摘  %s（仓库内务，对外无意义；--keep-internal 可保留）' % ' / '.join(removed))
        for n, kind, line in lint_source(open(src, encoding='utf-8').read()):
            rc = rc or 5
            print('   ⚠️ 源笔记第 %d 行：%s' % (n, kind))
            print('      %s' % (line[:88] + ('…' if len(line) > 88 else '')))
        if missing:
            rc = 2
            print('   缺图  %d 张（已在正文标红）: %s' % (len(missing), ', '.join(missing)))
        if '--html-only' in flags:
            continue
        if not browser:
            rc = rc or 3
            print('   PDF   跳过：没找到 Edge/Chrome（可设环境变量 BROWSER_FOR_PDF 指定）')
            continue
        pdf_path = os.path.join(outdir, stem + '.pdf')
        ok, why = to_pdf(browser, html_path, pdf_path)
        if ok:
            print('   PDF   %.2f MB · %d 页' % (os.path.getsize(pdf_path) / 1048576,
                                                pdf_pages(pdf_path)))
        else:
            rc = rc or 3
            print('   PDF   生成失败：%s（HTML 可用）' % why)

    gi = os.path.join(vault, '.gitignore')
    ignored = False
    if os.path.isfile(gi):
        with open(gi, encoding='utf-8', errors='ignore') as fh:
            ignored = bool(re.search(r'^_export/?\s*$', fh.read(), re.M))
    if not ignored:
        print('\n⚠️  .gitignore 里没有 `_export/`，导出产物会被 git 看见 —— 请补上这一行。')
    return rc


if __name__ == '__main__':
    sys.exit(main())
