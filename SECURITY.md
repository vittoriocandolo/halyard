# Security

Halyard is a command-line program and a Python library. It reads patch files (TOML) and WAV files, writes WAV, SFZ and
TOML files, and `audition` opens the audio device; it never uses the network. Every file it reads is checked as it is
read, so a damaged or crafted file is refused with its reason. If you find one that is not, or anything else that could
do harm (in the program, its package, or the workflows that build and publish it), please report it privately rather
than in a public issue.

**How:** [report it privately](https://github.com/vittoriocandolo/halyard/security/advisories/new) (GitHub's private
vulnerability reporting: the repository's Security tab, "Report a vulnerability"). Say what you found, how to see it,
and what it could do. You will get an answer within a week.

Only the latest release is looked after.
