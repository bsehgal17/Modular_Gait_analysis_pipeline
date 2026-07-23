import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec


from pose_estimation.utils.create_body_connections import build_connections

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec


class VideoVelocityViewer:
    def __init__(
        self,
        video_path,
        velocity,
        fps,
        joint_enum,
        show_skeleton=False,
        keypoints=None,
        skeleton_edges=None,
        save_svg=False,
        svg_dir="svg_frames",
    ):

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError("Cannot open video")

        self.velocity = velocity
        self.fps = fps

        self.show_skeleton = show_skeleton
        self.keypoints = keypoints

        # SVG options
        self.save_svg = save_svg
        self.svg_dir = svg_dir

        if self.save_svg:
            os.makedirs(self.svg_dir, exist_ok=True)

        # ---- Build skeleton if names provided ----
        if skeleton_edges is not None and isinstance(skeleton_edges[0][0], str):
            skeleton_edges = build_connections(joint_enum, skeleton_edges)

        self.skeleton_edges = skeleton_edges or []

        # Safety check
        for edge in self.skeleton_edges:
            if not isinstance(edge[0], int):
                raise ValueError(f"Invalid skeleton edge: {edge}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ---- Handle velocity ----
        if isinstance(velocity, dict):
            self.left_vel = velocity.get("left")
            self.right_vel = velocity.get("right")

            self.time = np.arange(len(self.left_vel)) / fps

        else:
            self.left_vel = velocity
            self.right_vel = None

            self.time = np.arange(len(velocity)) / fps

        # State
        self.current_frame = 0
        self.paused = False

        # -------------------------
        # Layout
        # -------------------------

        self.fig = plt.figure(figsize=(10, 10))

        gs = GridSpec(3, 1, height_ratios=[3, 1, 0.3])

        self.ax_video = self.fig.add_subplot(gs[0])
        self.ax_plot = self.fig.add_subplot(gs[1])
        self.ax_buttons = self.fig.add_subplot(gs[2])

        self.ax_buttons.axis("off")

        plt.subplots_adjust(hspace=0.25, bottom=0.15)

        # -------------------------
        # First frame
        # -------------------------

        ret, frame = self.cap.read()

        if not ret:
            raise ValueError("Cannot read first frame")

        h, w = frame.shape[:2]

        if self.show_skeleton:
            blank = np.zeros((h, w, 3), dtype=np.uint8)

            self.im = self.ax_video.imshow(blank)

            self.ax_video.set_xlim(0, w)

            self.ax_video.set_ylim(h, 0)

            self.ax_video.set_aspect("auto")

            self.ax_video.axis("off")

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.im = self.ax_video.imshow(frame)

        # -------------------------
        # Velocity plot
        # -------------------------

        self.ax_plot.set_xlabel("Time (s)")

        self.ax_plot.set_ylabel("Velocity")

        self.ax_plot.set_title("Velocity over Time")

        if self.right_vel is not None:
            self.ax_plot.plot(self.time, self.left_vel, label="Left Velocity")

            self.ax_plot.plot(self.time, self.right_vel, label="Right Velocity")

        else:
            self.ax_plot.plot(self.time, self.left_vel, label="Velocity")

        self.ax_plot.legend()

        self.cursor = self.ax_plot.axvline(0, color="r", linestyle="--")

        # -------------------------
        # Buttons
        # -------------------------

        ax_play = self.ax_buttons.inset_axes([0.3, 0.2, 0.15, 0.6])

        ax_pause = self.ax_buttons.inset_axes([0.55, 0.2, 0.15, 0.6])

        self.btn_play = Button(ax_play, "Play")

        self.btn_pause = Button(ax_pause, "Pause")

        self.btn_play.on_clicked(self.play)

        self.btn_pause.on_clicked(self.pause)

    # =================================================
    # DRAW SKELETON (existing)
    # =================================================

    def draw_skeleton(self, frame_idx, shape):

        if not hasattr(self, "canvas"):
            self.canvas = np.zeros(shape, dtype=np.uint8)

        self.canvas[:] = 0

        canvas = self.canvas

        if self.keypoints is None:
            return canvas

        if frame_idx >= len(self.keypoints):
            return canvas

        kp = np.asarray(self.keypoints[frame_idx])

        if kp.ndim == 2 and kp.shape[1] >= 2:
            coords = kp[:, :2]

        elif kp.ndim == 2 and kp.shape[0] == 2:
            coords = kp.T

        else:
            return canvas

        # joints

        for x, y in coords:
            if x == 0 and y == 0:
                continue

            if np.isnan(x) or np.isnan(y):
                continue

            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)

        # bones

        for j1, j2 in self.skeleton_edges:
            if j1 >= len(coords) or j2 >= len(coords):
                continue

            x1, y1 = coords[j1]
            x2, y2 = coords[j2]

            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

        return canvas

    # =================================================
    # SVG EXPORT
    # =================================================

    def save_combined_svg(self, frame_idx, width, height):

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        ax_skeleton = fig.add_subplot(gs[0])
        ax_velocity = fig.add_subplot(gs[1])

        # ==========================
        # Skeleton panel
        # ==========================
        ax_skeleton.set_facecolor("white")

        kp = np.asarray(self.keypoints[frame_idx])

        if kp.ndim == 2 and kp.shape[1] >= 2:
            coords = kp[:, :2]
        elif kp.ndim == 2 and kp.shape[0] == 2:
            coords = kp.T
        else:
            return

        # bones
        for j1, j2 in self.skeleton_edges:
            if j1 >= len(coords) or j2 >= len(coords):
                continue
            x1, y1 = coords[j1]
            x2, y2 = coords[j2]
            if np.isnan([x1, y1, x2, y2]).any():
                continue
            ax_skeleton.plot([x1, x2], [y1, y2], linewidth=2)

        # joints
        valid = []
        for x, y in coords:
            if x == 0 and y == 0:
                continue
            if np.isnan(x) or np.isnan(y):
                continue
            valid.append([x, y])

        if not valid:
            plt.close(fig)
            return

        valid = np.asarray(valid)
        ax_skeleton.scatter(valid[:, 0], valid[:, 1], s=25)

        # ---- Tight bounding box around the actual keypoints, not the full frame ----
        x_min, y_min = valid.min(axis=0)
        x_max, y_max = valid.max(axis=0)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # margin proportional to the skeleton's own size, with a small floor
        # so a single-frame degenerate bbox doesn't collapse to nothing
        margin_x = max(bbox_w * 0.15, 10)
        margin_y = max(bbox_h * 0.15, 10)

        ax_skeleton.set_xlim(x_min - margin_x, x_max + margin_x)
        ax_skeleton.set_ylim(
            y_max + margin_y, y_min - margin_y
        )  # inverted y (image coords)

        ax_skeleton.set_aspect("equal", adjustable="box")
        ax_skeleton.axis("off")

        # ==========================
        # Velocity plot
        # ==========================
        ax_velocity.plot(self.time, self.left_vel, label="Left velocity")

        if self.right_vel is not None:
            ax_velocity.plot(self.time, self.right_vel, label="Right velocity")

        current_time = frame_idx / self.fps
        ax_velocity.axvline(current_time, linestyle="--")

        ax_velocity.set_xlabel("Time (s)")
        ax_velocity.set_ylabel("Velocity")
        ax_velocity.legend()

        fig.tight_layout()

        output = os.path.join(self.svg_dir, f"frame_{frame_idx:05d}.svg")
        fig.savefig(output, format="svg", bbox_inches="tight", pad_inches=0.02)

        plt.close(fig)

    def export_svg_frames(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for idx in range(self.frame_count):
            ret, frame = self.cap.read()

            if not ret:
                break

            h, w = frame.shape[:2]

            self.save_combined_svg(idx, w, h)

        print(f"Saved SVG frames to {self.svg_dir}")

    # =================================================
    # Controls
    # =================================================

    def play(self, event):
        self.paused = False

    def pause(self, event):
        self.paused = True

    # =================================================
    # Animation update
    # =================================================

    def update(self, frame_idx):

        if self.paused:
            return self.im, self.cursor

        if self.current_frame >= self.frame_count:
            self.current_frame = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        ret, frame = self.cap.read()

        if not ret:
            return self.im, self.cursor

        h, w = frame.shape[:2]

        if self.show_skeleton:
            frame = self.draw_skeleton(self.current_frame, (h, w, 3))

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.im.set_array(frame)

        t = self.current_frame / self.fps

        self.cursor.set_xdata([t, t])

        self.current_frame += 1

        return self.im, self.cursor

    # =================================================
    # RUN
    # =================================================

    def run(self):

        if self.save_svg:
            self.export_svg_frames()

            self.cap.release()

            return

        interval = int(1000 / self.fps)

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frame_count,
            interval=interval,
            blit=False,
            repeat=False,
        )

        plt.show(block=True)

        self.cap.release()
