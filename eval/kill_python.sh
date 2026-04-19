#!/usr/bin/env bash
#
# kill_python.sh —— 查看并强杀当前用户的所有 Python 进程（macOS 版）

echo "==== Python processes of user ${USER} ===="
# macOS 用 command / comm，而不是 cmd
ps -u "${USER}" -o pid,ppid,%cpu,%mem,command | grep '[p]ython' || echo "None"

echo "----------------------------------------"
echo "Killing them with SIGKILL (-9) ..."
# 只杀当前用户、命令行里含 python 的进程
pkill -9 -u "${USER}" -f python 2>/dev/null

echo "Done."
