#!/usr/bin/env bash
# Restart the local Koenami API + built site with the accent page (fork lab use).
cd "$(dirname "$0")/.." || exit 1
pkill -f "python server.py --port 8766" 2>/dev/null
sleep 1
nohup .venv-accent/bin/python server.py --port 8766 --static .svelte-kit/cloudflare > data/server.log 2>&1 < /dev/null &
disown
