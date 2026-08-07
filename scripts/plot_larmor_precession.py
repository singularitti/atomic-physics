"""Render one period of Larmor precession as an MP4 video."""

from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from atomic_physics import LarmorPrecession

OUTPUT_PATH = Path("artifacts/larmor_precession.mp4")
VIDEO_DURATION_SECONDS = 8
FRAMES_PER_SECOND = 30


def build_animation(
    model: LarmorPrecession,
    *,
    duration_seconds: int = VIDEO_DURATION_SECONDS,
    frames_per_second: int = FRAMES_PER_SECOND,
) -> tuple[Figure, FuncAnimation]:
    """Build a 3D animation covering exactly one precession period."""
    if not np.isfinite(model.period):
        msg = "a nonzero angular velocity is required for an animation"
        raise ValueError(msg)
    if duration_seconds <= 0 or frames_per_second <= 0:
        msg = "duration_seconds and frames_per_second must be positive"
        raise ValueError(msg)

    frame_count = duration_seconds * frames_per_second
    times = np.linspace(0.0, model.period, frame_count, endpoint=True)
    moments = model.magnetic_moment(times)

    figure = plt.figure(figsize=(9, 8), layout="constrained")
    axis = figure.add_subplot(projection="3d")
    axis.set_title(
        r"Larmor precession: "
        r"$d\boldsymbol{\mu}/dt="
        r"\boldsymbol{\omega}\times\boldsymbol{\mu}$",
        pad=18,
    )
    axis.set_xlabel(r"$\mu_x$")
    axis.set_ylabel(r"$\mu_y$")
    axis.set_zlabel(r"$\mu_z$")
    axis.view_init(elev=24, azim=38)
    axis.set_box_aspect((1.0, 1.0, 1.0))

    moment_norm = float(np.linalg.norm(model.initial_magnetic_moment))
    extent = max(1.0, 1.35 * moment_norm)
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)

    omega_display = model.rotation_axis * min(extent, 1.15 * moment_norm)
    axis.quiver(
        0.0,
        0.0,
        0.0,
        *omega_display,
        color="tab:blue",
        linewidth=2.5,
        arrow_length_ratio=0.1,
        label=r"$\boldsymbol{\omega}$ axis",
    )

    (full_orbit,) = axis.plot(
        moments[:, 0],
        moments[:, 1],
        moments[:, 2],
        color="0.75",
        linestyle="--",
        linewidth=1.2,
        label="orbit",
    )
    (trail,) = axis.plot([], [], [], color="tab:orange", linewidth=2.2)
    (tip,) = axis.plot([], [], [], "o", color="tab:red", markersize=6)
    time_text = axis.text2D(0.03, 0.95, "", transform=axis.transAxes)

    initial = moments[0]
    moment_arrow = axis.quiver(
        0.0,
        0.0,
        0.0,
        *initial,
        color="tab:red",
        linewidth=3.0,
        arrow_length_ratio=0.1,
        label=r"$\boldsymbol{\mu}(t)$",
    )
    axis.legend(loc="upper right")

    def update(frame: int) -> tuple[Artist, ...]:
        nonlocal moment_arrow
        moment_arrow.remove()

        moment = moments[frame]
        moment_arrow = axis.quiver(
            0.0,
            0.0,
            0.0,
            *moment,
            color="tab:red",
            linewidth=3.0,
            arrow_length_ratio=0.1,
        )
        trail.set_data(moments[: frame + 1, 0], moments[: frame + 1, 1])
        trail.set_3d_properties(moments[: frame + 1, 2])
        tip.set_data([moment[0]], [moment[1]])
        tip.set_3d_properties([moment[2]])
        time_text.set_text(f"t = {times[frame]:.2f}")
        return moment_arrow, full_orbit, trail, tip, time_text

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1_000 / frames_per_second,
        blit=False,
    )
    return figure, animation


def render_video(
    model: LarmorPrecession,
    output_path: Path,
    *,
    duration_seconds: int = VIDEO_DURATION_SECONDS,
    frames_per_second: int = FRAMES_PER_SECOND,
) -> Path:
    """Render the animation to ``output_path`` with the bundled FFmpeg."""
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, animation = build_animation(
        model,
        duration_seconds=duration_seconds,
        frames_per_second=frames_per_second,
    )
    writer = FFMpegWriter(
        fps=frames_per_second,
        metadata={
            "title": "Larmor precession",
            "artist": "atomic-physics",
        },
        bitrate=2_400,
    )
    animation.save(output_path, writer=writer, dpi=160)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    precession = LarmorPrecession(
        angular_velocity=np.array([0.0, 0.0, 1.0]),
        initial_magnetic_moment=np.array([2.0, 0.0, 1.2]),
    )
    rendered_path = render_video(precession, OUTPUT_PATH)
    print(f"Saved {rendered_path}")
