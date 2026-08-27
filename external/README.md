# External code layout

This project keeps two kinds of external source trees:

- `ol-ems/` and `python-microgrid/` are read-only upstream reference checkouts.
- `ol-ems-hkust/` and `python-microgrid-hkust/` are vendored project snapshots.

Only the `*-hkust` directories are modified for this project. They are tracked as
ordinary files by the parent repository rather than as nested Git repositories or
submodules. This makes every local modification available from a normal clone of
the parent repository without requiring writable forks of the upstream projects.

The exact upstream revisions and licenses are recorded in
`upstreams.lock.json`. The original license files remain inside both vendored
directories.
