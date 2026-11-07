#!/bin/sh
# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build sfizz and liquidsfz from source in a throwaway container, and render the notes
# `python -m tools.sfz_players prepare DIR` wrote, into DIR/sfizz.wav and DIR/liquidsfz.wav.
#
#   tools/sfz_players.sh DIR
#
# Everything is built and run inside the container; the image is removed afterwards, unless KEEP=1.
set -eu

dir=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
image=halyard-sfz-players
sfizz=1.2.3
liquidsfz=0.4.1

docker build -t "$image" -f - "$here" <<EOF
FROM debian:trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch $sfizz --recurse-submodules --shallow-submodules \
      https://github.com/sfztools/sfizz.git /src/sfizz \
    && cmake -S /src/sfizz -B /src/sfizz/build -DCMAKE_BUILD_TYPE=Release -DSFIZZ_JACK=OFF \
      -DSFIZZ_RENDER=ON -DPLUGIN_LV2=OFF -DPLUGIN_VST3=OFF -DPLUGIN_AU=OFF -DPLUGIN_PUREDATA=OFF \
      -DSFIZZ_SHARED=OFF -DSFIZZ_TESTS=OFF -DSFIZZ_DEMOS=OFF -DSFIZZ_BENCHMARKS=OFF \
    && cmake --build /src/sfizz/build -j "\$(nproc)" --target sfizz_render \
    && cp "\$(find /src/sfizz/build -name sfizz_render -type f -perm -u+x | head -1)" /usr/local/bin/
RUN git clone --depth 1 --branch $liquidsfz https://github.com/swesterfeld/liquidsfz.git /src/liquidsfz
COPY liquidsfz_render.cc /src/liquidsfz_render.cc
RUN printf '#define VERSION "$liquidsfz"\n' > /src/liquidsfz/lib/config.h \
    && g++ -O2 -std=c++17 -I/src/liquidsfz/lib /src/liquidsfz_render.cc /src/liquidsfz/lib/*.cc \
      -lsndfile -lpthread -o /usr/local/bin/liquidsfz_render
EOF

frames=$(cat "$dir/frames.txt")
sfz=$(cd "$dir/kit" && ls ./*.sfz | head -1)
docker run --rm -v "$dir:/w" "$image" sh -c "
  cd /w/kit &&
  sfizz_render --sfz '$sfz' --midi /w/notes.mid --wav /w/sfizz.wav --samplerate 48000 &&
  liquidsfz_render '$sfz' /w/notes.txt $frames 48000 /w/liquidsfz.wav &&
  echo 'sfizz $sfizz and liquidsfz $liquidsfz rendered the notes'"
[ "${KEEP:-}" ] || docker rmi "$image" >/dev/null  # KEEP=1 keeps the image, to look further
