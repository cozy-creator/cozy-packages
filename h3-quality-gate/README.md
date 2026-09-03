# tensorhub/h3-quality-gate

This is the publishable package for the bounded `h3_quality_gate` job. Its stable callable
reference is `tensorhub/h3-quality-gate@v1/h3_quality_gate`.

The package scores already-produced candidate media against one same-seed BF16 reference and a
caller-supplied, previously ratified protocol. Video and audio gate independently; missing metrics
remain unmeasured, and the job never renders media or selects a public model lane.

The implementation stays in the shared `cozy-jobs` library so metric identity, digest handling,
red arms, and the noise-floor reader have one owner. Its metric core is the portable
`cozy-eval==2.3.3` wheel. This project is the thin deployable Package surface: it supplies an exact
lock, wheel, PackageInterface, and application import. The job is CPU-only and declares an 8 GiB
host-RAM floor.
