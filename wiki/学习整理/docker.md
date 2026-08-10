# Docker 常用命令

## 先分清：镜像 vs 容器（原笔记这里最容易混）

- **镜像（image）**：只读模板，好比"安装包 / 类"。
- **容器（container）**：镜像跑起来的实例，好比"进程 / 对象"。一个镜像能跑出多个容器。

记住这条分界，命令就不会用混：
`images / pull / rmi / build` 操作**镜像**；`ps / run / exec / start / stop / rm / logs` 操作**容器**。

## 镜像操作

```bash
docker pull nginx:latest        # 拉取镜像（不写 tag 默认 latest）
docker images                   # 列出本地镜像
docker rmi nginx:latest         # 删除镜像
docker build -t myapp:1.0 .     # 用当前目录的 Dockerfile 构建镜像
docker tag myapp:1.0 myapp:latest   # 给镜像打新标签
docker push myapp:1.0           # 推送镜像到仓库
```

## 容器操作

```bash
docker run -d --name web -p 8080:80 nginx   # 从镜像"新建并启动"一个容器
docker ps                       # 查看【运行中】的容器
docker ps -a                    # 查看【所有】容器（含已停止）
docker exec -it web bash        # 进入运行中的容器开交互 shell（参数是"容器名"，不是镜像名）
docker stop web                 # 停止容器
docker start web                # 启动一个已停止的容器
docker restart web              # 重启容器
docker rm web                   # 删除容器（需先 stop）
docker rm -f web                # 强制删除运行中的容器
```

> 精简镜像（如 alpine）可能没有 bash，用 `docker exec -it web sh`。

### `docker run` 常用参数

- `-d` 后台运行（detach）
- `-it` 交互式 + 终端（跑 bash/sh 等要加）
- `--name` 给容器起名，方便后续引用
- `-p 主机端口:容器端口` 端口映射
- `-v 主机路径:容器路径` 挂载目录 / 数据卷
- `-e KEY=value` 设置环境变量
- `--rm` 容器退出后自动删除（跑一次性任务常用）
- `--restart=always` 崩溃 / 开机自动重启

## 日志、文件与状态

```bash
docker logs -f web              # 查看容器日志（-f 实时跟随）
docker cp web:/app/log.txt .    # 从容器拷文件到主机（顺序反过来即主机→容器）
docker stats                    # 实时查看各容器 CPU/内存占用
docker inspect web              # 查看容器完整配置（JSON）
```

## 清理

```bash
docker stop $(docker ps -q)     # 停止所有运行中的容器
docker container prune          # 删除所有已停止的容器
docker image prune              # 删除悬空（dangling）镜像
docker system prune -a          # 清理所有未使用的镜像/容器/网络（慎用）
```

## Docker Compose（多容器编排）

```bash
docker compose up -d            # 按 docker-compose.yml 启动全部服务（后台）
docker compose down             # 停止并移除服务
docker compose ps               # 查看 compose 管理的容器
docker compose logs -f          # 实时查看服务日志
```

## 常见易错点

- `docker ps` 查的是**容器**，查**镜像**用 `docker images`。
- `docker exec / start / stop / rm` 的参数是**容器名或容器 ID**，不是镜像名。
- `docker run` = "新建并启动"一个容器；`docker start` = "启动一个已存在的"容器，别混。
- 删镜像是 `rmi`、删容器是 `rm`，差一个字母，对象完全不同。
- `-p` 的顺序是 `主机端口:容器端口`，写反了外部就访问不到。
