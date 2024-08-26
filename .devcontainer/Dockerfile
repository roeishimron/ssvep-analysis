FROM mcr.microsoft.com/vscode/devcontainers/python:3

RUN apt-get update && apt-get install -y \
    build-essential libgl1-mesa-dev libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0 libx11-xcb-dev\
    libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev\
    'libxcb*[!-dev]'\
    qtwayland5