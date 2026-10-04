# Numerical-policy version custody

PR390 reserves SDXL 2.7.0 and Anima 0.3.2 for the final production numerical
policy. This is a source reservation, not publication or qualification.

Live ordinary CLI release reads on October 4 show `paul/sdxl` latest 2.5.0 and
`paul/anima` latest 0.3.0. Packages master444ea624 reserves SDXL 2.6.0. Independent
draft [PR387](https://github.com/cozy-creator/packages/pull/387), source caef9f5,
reserves Anima 0.3.1 for the single-frame VAE implementation. That branch and its
evidence are preserved; this policy does not replace or claim its VAE results.

The initial PR390 source incorrectly used Anima 0.3.1. Its uninstalled local wheel
is retained as a superseded artifact, excluded from the candidate cohort. Final
source uses 0.3.2 and will receive a newly built wheel. No tag or published release
has been changed.

Public PyPI's `sdxl` project belongs to an unrelated API package at version 1.0.0;
PyPI's `anima` JSON endpoint returns404. These are not Cozy Hub release records
and do not establish custody for the `paul` package namespace. The current Hub
records, branch manifests and tracker reservation are the relevant authorities.
