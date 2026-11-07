// Copyright (C) 2026 Vittorio Candolo
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Render notes through liquidsfz offline, for tools/sfz_players.py: liquidsfz itself ships only real-time clients.
//
//   liquidsfz_render KIT.sfz NOTES.txt FRAMES RATE OUT.wav
//
// NOTES has one "frame key velocity" per line, velocity 0 ending the note. OUT is a stereo float WAV.
// tools/sfz_players.sh builds it against liquidsfz's sources and libsndfile.
#include "liquidsfz.hh"

#include <sndfile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

struct Note
{
  long frame;
  int key;
  int velocity;
};

int
main (int argc, char **argv)
{
  if (argc != 6)
    {
      std::fprintf (stderr, "usage: liquidsfz_render KIT.sfz NOTES.txt FRAMES RATE OUT.wav\n");
      return 2;
    }
  const long frames = std::atol (argv[3]);
  const int rate = std::atoi (argv[4]);

  LiquidSFZ::Synth synth;
  synth.set_sample_rate (rate);
  if (!synth.load (argv[1]))
    {
      std::fprintf (stderr, "liquidsfz cannot load %s\n", argv[1]);
      return 1;
    }

  std::vector<Note> notes;
  std::ifstream in (argv[2]);
  for (Note n; in >> n.frame >> n.key >> n.velocity;)
    notes.push_back (n);
  std::stable_sort (notes.begin (), notes.end (), [] (const Note &a, const Note &b) { return a.frame < b.frame; });

  SF_INFO info = {};
  info.samplerate = rate;
  info.channels = 2;
  info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  SNDFILE *out = sf_open (argv[5], SFM_WRITE, &info);
  if (!out)
    {
      std::fprintf (stderr, "cannot write %s\n", argv[5]);
      return 1;
    }

  const long block = 128;
  std::vector<float> left (block), right (block), both (2 * block);
  size_t next = 0;
  for (long start = 0; start < frames; start += block)
    {
      const long n = std::min (block, frames - start);
      for (; next < notes.size () && notes[next].frame < start + n; next++)
        {
          const Note &e = notes[next];
          const uint offset = uint (e.frame - start);
          if (e.velocity > 0)
            synth.add_event_note_on (offset, 0, e.key, e.velocity);
          else
            synth.add_event_note_off (offset, 0, e.key);
        }
      float *outputs[2] = { left.data (), right.data () };
      synth.process (outputs, uint (n));
      for (long i = 0; i < n; i++)
        {
          both[2 * i] = left[i];
          both[2 * i + 1] = right[i];
        }
      sf_writef_float (out, both.data (), n);
    }
  sf_close (out);
  return 0;
}
