# atomic-physics

Reusable atomic-physics models and visualizations.

## Larmor precession

`LarmorPrecession` evaluates the exact solution of

\[
\frac{d\boldsymbol{\mu}}{dt}
= \boldsymbol{\omega} \times \boldsymbol{\mu}
\]

for a constant angular-velocity vector. The implementation uses Rodrigues'
rotation formula, so it does not accumulate the norm drift associated with a
generic numerical ODE integrator.

The reusable model is in `src/atomic_physics/larmor_precession.py`. The
Matplotlib application in `scripts/plot_larmor_precession.py` renders one full
precession period as a 3D MP4 animation.

Install the locked environment and render the video from the project root:

```bash
uv sync
uv run python scripts/plot_larmor_precession.py
```

The completed video is written to `artifacts/larmor_precession.mp4`. Change
`angular_velocity`, `initial_magnetic_moment`, the video duration, or the frame
rate near the bottom of the plotting script to customize it. FFmpeg is supplied
by the uv dependency set, so a separate system installation is not required.

`artifacts/` contains final rendered media only. Its input provenance is the
checked-out package source plus the parameters defined in the plotting script;
no source inputs are copied into the output directory, and no temporary frames
are retained.

Run the tests with:

```bash
uv run pytest
```
