#!/usr/bin/env bash
# 收集指定时间范围内 wiki/ 下的笔记改动，供日报 / 周报使用。
#
# 用法：
#   bash collect_diff.sh [vault根] [--since <日期>|--days <N>] [--digest]
#
#   日报（默认）： bash collect_diff.sh "D:\llm\llm"
#                  → 今天 0 点至今，逐文件列出全部新增行
#   周报        ： bash collect_diff.sh "D:\llm\llm" --days 7 --digest
#                  → 近 7 天，只给统计 + 新增章节标题，不倾泻正文
#   指定周      ： bash collect_diff.sh "D:\llm\llm" --since 2026-08-10 --digest
#
# 参数：
#   --since <日期>  起点，git 认的任何写法（2026-08-10 / "7 days ago" / midnight）
#   --days <N>      起点 = N 天前（等价于 --since "N days ago"）
#   --digest        汇总模式：按文件统计 + 按天活跃度 + 新增章节标题（周报用）
#   默认起点：full 模式 midnight，digest 模式 7 days ago
#
# 环境变量：
#   MAX_LINES  full 模式下每文件最多输出多少行（默认 120，超出显式报告剩余行数）
#              信息不够就调高重跑，例如 MAX_LINES=400
#
# 实现要点（都是踩过的坑，别改回去）：
#   * core.quotePath=false —— 否则中文文件名被转义成 \345\255\246 这种八进制，
#     拿去当路径再 diff 一次就取不到内容。
#   * 用「基线提交 → 工作区」的净 diff，不逐 commit 累加。这个 vault 每隔几分钟
#     自动 commit 一次（"vault backup: <时间戳>"），逐 commit 会把一篇笔记撕成十几段，
#     还会因为「先列文件名、再逐个 diff」中间被自动提交抢跑而拿到空结果。
#   * index.md / log.md 自动标注为噪声 —— 前者是索引条目，后者是 AI 自己的操作日志，
#     都不是用户的学习成果，绝不能写进汇报。
#
# 退出码：
#   0  有改动，已输出
#   1  该时间范围内 wiki/ 无改动（调用方应转而询问用户，别猜）
#   2  参数或环境错误

set -uo pipefail

VAULT=""
SINCE=""
DIGEST=0
MAX_LINES="${MAX_LINES:-120}"
EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --digest) DIGEST=1; shift ;;
    --since)  SINCE="${2:-}"; shift 2 ;;
    --days)   SINCE="${2:-7} days ago"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "ERROR: 未知参数 $1" >&2; exit 2 ;;
    *)  VAULT="$1"; shift ;;
  esac
done

[ -z "$VAULT" ] && VAULT="$(pwd)"
if [ -z "$SINCE" ]; then
  if [ "$DIGEST" = 1 ]; then SINCE="7 days ago"; else SINCE="midnight"; fi
fi

git_q() { git -c core.quotePath=false -C "$VAULT" "$@"; }

if ! git_q rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: '$VAULT' 不是 git 仓库" >&2
  exit 2
fi
if [ ! -d "$VAULT/wiki" ]; then
  echo "ERROR: '$VAULT/wiki' 不存在" >&2
  exit 2
fi

# ---------- 基线：范围起点之前的最后一个提交 ----------
BOUNDARY=$(git_q rev-list -1 --before="$SINCE" HEAD 2>/dev/null)
if [ -z "$BOUNDARY" ]; then
  BOUNDARY="$EMPTY_TREE"
  BOUNDARY_DESC="（仓库在该起点之前没有提交，基线取空树，即从零开始算）"
else
  BOUNDARY_DESC=$(git_q log -1 --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M' "$BOUNDARY")
fi

# 基线 → 工作区 的净 diff（含未提交改动，不含未跟踪文件）
NET_DIFF=$(git_q diff "$BOUNDARY" -- wiki/ 2>/dev/null)
UNTRACKED=$(git_q ls-files --others --exclude-standard -- wiki/ 2>/dev/null)

if [ -z "$NET_DIFF" ] && [ -z "$UNTRACKED" ]; then
  echo "NO_CHANGES: 「$SINCE」以来 wiki/ 下没有检测到改动"
  exit 1
fi

echo "=============================================="
echo "wiki/ 改动收集$([ "$DIGEST" = 1 ] && echo "（digest 汇总模式）" || echo "（full 全文模式）")"
echo "范围：「$SINCE」至今"
echo "基线：$BOUNDARY_DESC"
echo "=============================================="

# 噪声文件标注：这两个不是学习成果
noise_tag() {
  case "$1" in
    wiki/index.md) echo "   ← 噪声：全局索引条目，非学习内容" ;;
    wiki/log.md)   echo "   ← 噪声：AI 操作日志，绝不可当成用户的学习成果" ;;
    *) echo "" ;;
  esac
}

# 把 unified diff 切成「文件 → 新增行」分段
SPLIT_AWK='
function flush() {
  if (file != "" && total > shown)
    printf "…（该文件本期共新增 %d 行，已显示前 %d 行，剩余 %d 行未显示）\n", total, shown, total - shown
}
/^diff --git / {
  flush()
  # "a/PATH b/PATH"，两侧路径相同，按长度取中点，含空格的中文名也不会切错
  s = substr($0, 12); L = int((length(s) - 5) / 2); file = substr(s, 3, L)
  printf "\n--- 文件：%s ---\n", file
  total = 0; shown = 0
  next
}
/^new file mode/ { print "（★ 本期新建，以下即全文）"; next }
/^\+\+\+ / { next }
/^\+/ {
  line = substr($0, 2)
  if (line ~ /^[[:space:]]*$/) next
  total++
  if (shown < MAX) { print line; shown++ }
  next
}
END { flush() }
'

# ==================================================================
#  DIGEST 模式（周报）
# ==================================================================
if [ "$DIGEST" = 1 ]; then
  echo
  echo "【一】按文件：本期净新增行数（从多到少）"
  echo

  NEWFILES=$(git_q diff "$BOUNDARY" --diff-filter=A --name-only -- wiki/ 2>/dev/null)

  git_q diff "$BOUNDARY" --numstat -- wiki/ 2>/dev/null \
    | awk -F'\t' -v newfiles="$NEWFILES" '
        BEGIN { n = split(newfiles, a, "\n"); for (i = 1; i <= n; i++) if (a[i] != "") isnew[a[i]] = 1 }
        $1 != "-" { add[$3] += $1; total += $1 }
        END {
          for (f in add) printf "%8d\t%s\t%s\n", add[f], (f in isnew ? "★新建" : "     "), f
        }' \
    | sort -rn \
    | while IFS=$'\t' read -r n mark f; do
        printf '%8s 行  %s  %s%s\n' "$n" "$mark" "$f" "$(noise_tag "$f")"
      done

  if [ -n "$UNTRACKED" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      L=$(grep -c . "$VAULT/$f" 2>/dev/null || echo 0)
      printf '%8s 行  ★新建  %s   （尚未 git add）\n' "$L" "$f"
    done <<< "$UNTRACKED"
  fi

  echo
  echo "【二】按天：每天动过哪些笔记（用来判断哪几天在攻同一件事）"
  echo

  git_q log --since="$SINCE" --date=short --format="@@@%ad" --name-only -- wiki/ 2>/dev/null \
    | awk '
        /^@@@/ { d = substr($0, 4); next }
        /^wiki\// { if (!seen[d "|" $0]++) { files[d] = files[d] "\n    " $0; if (!(d in days)) { order[++k] = d; days[d] = 1 } } }
        END { for (i = 1; i <= k; i++) printf "  %s%s\n\n", order[i], files[order[i]] }'

  echo "【三】本期新增的章节标题（判断「学了什么」的主要线索）"
  echo

  printf '%s\n' "$NET_DIFF" | awk '
    /^diff --git / { s = substr($0, 12); L = int((length(s) - 5) / 2); file = substr(s, 3, L); printed = 0; next }
    /^\+#{1,6} / {
      if (file == "wiki/index.md" || file == "wiki/log.md") next
      if (!printed) { printf "\n--- %s ---\n", file; printed = 1 }
      print "  " substr($0, 2)
    }'

  if [ -n "$UNTRACKED" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      case "$f" in wiki/index.md|wiki/log.md) continue ;; esac
      HEADS=$(grep -E '^#{1,6} ' "$VAULT/$f" 2>/dev/null | sed 's/^/  /')
      [ -n "$HEADS" ] && { printf '\n--- %s （未 git add 的新文件）---\n' "$f"; printf '%s\n' "$HEADS"; }
    done <<< "$UNTRACKED"
  fi

  echo
  echo "----------------------------------------------"
  echo "只想细看某一篇？去掉 --digest 并把起点设成同一个，例如："
  echo "  bash \$0 \"$VAULT\" --since \"$SINCE\""
  echo "----------------------------------------------"
  exit 0
fi

# ==================================================================
#  FULL 模式（日报）
# ==================================================================
[ -n "$NET_DIFF" ] && printf '%s\n' "$NET_DIFF" | awk -v MAX="$MAX_LINES" "$SPLIT_AWK"

if [ -n "$UNTRACKED" ]; then
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    echo
    echo "--- 文件：$f ---"
    echo "（★ 全新文件，尚未 git add，以下即全文）"
    sed '/^[[:space:]]*$/d' "$VAULT/$f" 2>/dev/null \
      | awk -v MAX="$MAX_LINES" '
          { total++; if (shown < MAX) { print; shown++ } }
          END { if (total > shown)
                  printf "…（该文件共 %d 行，已显示前 %d 行，剩余 %d 行未显示）\n", total, shown, total - shown }'
  done <<< "$UNTRACKED"
fi

echo
echo "----------------------------------------------"
echo "提醒：wiki/index.md 是索引条目、wiki/log.md 是 AI 操作日志，"
echo "      两者都不是用户的学习成果，写汇报时跳过。"
echo "----------------------------------------------"
exit 0
