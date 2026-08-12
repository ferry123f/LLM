# Git 常用命令

> 一句话：Git 从零到日常的最小上手路径——**首次配置 → 初始化并关联远程 → 日常 add/commit/pull/push 循环**，外加撤销、查历史、分支、`.gitignore` 等高频救急命令。

#git

## 首次使用配置

全局身份信息，一台机器配一次即可：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

> 补充：现在 GitHub 已不支持密码推送，HTTPS 方式需用 **Personal Access Token（PAT）** 代替密码，或改用 SSH。首次 `push` 弹出的"密码"框要填 PAT，不是账号密码。

## 初始化仓库并首次提交

```bash
git init                        # 在当前目录初始化仓库
git add -A                      # 把所有文件加入暂存区
git commit -m "首次上传"         # 保存为第一个本地版本
git branch -M main              # 把主分支重命名为 main
```

## 关联远程仓库并推送

```bash
git remote add origin https://github.com/你的用户名/仓库名.git   # 关联远程仓库
git push -u origin main         # 首次推送并建立追踪（之后可直接 git push）
```

## 日常修改上传

```bash
git status                      # 查看哪些文件发生变化
git add -A                      # 把所有新增、修改和删除加入暂存区
git commit -m "说明本次修改内容"  # 把修改保存为一个本地版本
git pull --rebase origin main   # 获取远程可能已上传的新版本（rebase 方式合并）
git push origin main            # 真正上传到远程仓库
```

## 日常流程速记

- **开始工作**：`git pull --rebase`
- **完成工作**：`git add -A` → `git commit` → `git push`

## 查看状态与历史

```bash
git status                      # 当前有哪些改动（未暂存/已暂存/未跟踪）
git log --oneline --graph       # 一行一条的提交历史，带分支图
git diff                        # 看「工作区 vs 暂存区」的具体改动
git diff --staged               # 看「暂存区 vs 上次提交」的改动
```

## 撤销与回退（新手高频救急）

```bash
git restore <文件>              # 丢弃工作区对某文件的修改（还没 add 时）
git restore --staged <文件>     # 把文件移出暂存区（撤销 add，改动仍保留）
git commit --amend              # 修改「最近一次」提交的信息或补文件（未 push 时用）
git reset --soft HEAD~1         # 撤销最近一次 commit，改动退回暂存区
git reset --hard HEAD~1         # ⚠️ 彻底丢弃最近一次 commit 及其改动，不可恢复，慎用
```

> `restore` 是较新（Git 2.23+）的命令；老教程里的 `git checkout -- <文件>` 效果等价，仍可用。

## 分支操作

```bash
git branch                      # 列出本地分支
git switch -c <新分支名>         # 新建并切换到该分支（等价于 git checkout -b）
git switch main                 # 切回 main 分支
git merge <分支名>               # 把指定分支合并进当前分支
git branch -d <分支名>           # 删除已合并的分支
```

## 忽略文件：.gitignore

在仓库根目录建 `.gitignore` 文本文件，写入不想被 Git 跟踪的路径（一行一条）：

```gitignore
node_modules/     # 依赖目录
*.log             # 所有 .log 文件
.env              # 敏感配置
__pycache__/      # Python 缓存
```

> 注意：`.gitignore` 只对**尚未被跟踪**的文件生效。已经 `add`/`commit` 过的文件需先 `git rm --cached <文件>` 移出跟踪，再提交才会生效。

## pull --rebase 遇到冲突怎么办

本笔记日常流程用的是 `git pull --rebase`。若远程和本地改了同一处，rebase 会中断并提示冲突：

```bash
# 1. 打开冲突文件，手动改掉 <<<<<<< ======= >>>>>>> 标记之间的内容
git add <已解决的文件>          # 2. 标记该文件冲突已解决
git rebase --continue          # 3. 继续 rebase
# 或者放弃这次 rebase，回到 pull 之前的状态：
git rebase --abort
```

## See Also

- [[docker|Docker 常用命令]] — 同属命令速查系列
