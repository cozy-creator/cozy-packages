Continuous H3 generation now carries the previous segment's completed audio and video context into the next segment while retaining generated character and scene references. Each segment delivers only its new frames; the worker assembles the final video and continuation frame.

This integrates the existing motion-context implementation with the removal of package and SDK fingerprints. Context compatibility checks model and adapter selections only. Compact native context retains independently encoded 22-, 39-, and 56-frame windows within the 64 MiB asset limit. Continuous sequences have no eight-shot cap.

Validation: source integration and protocol review only. No local tests, lint, vet, or CI were run, as requested. Private CLI GPU qualification of both six-segment, 60-second videos remains pending the coordinated Runtime/Creator installer update. The portable lock still requires refresh against a published compatible Runtime before release.
