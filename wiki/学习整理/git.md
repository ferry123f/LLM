首次使用配置
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
初始化仓库并提交：
git init
git add -A
git commit -m "首次上传"
git branch -M main
关联仓库
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main

后续使用上传自己修改
git status #查看哪些文件发生变化
git add -A #把所有新增、修改和删除加入暂存区
git commit -m "说明本次修改内容" #把修改保存为一个本地版本
git pull --rebase origin main #获取远程设备可能上传的新版本
git push origin main #真正上传到仓库 

一般使用：
开始工作：git pull --rebase
完成工作：git add -A → git commit → git push