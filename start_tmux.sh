#!/bin/bash
tmux new-session -d -s docs 'bash'
tmux send 'conda activate ../.probcell' ENTER;
tmux a
