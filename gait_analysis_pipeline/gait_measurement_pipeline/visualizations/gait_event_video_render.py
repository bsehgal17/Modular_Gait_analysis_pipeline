import cv2
import matplotlib.pyplot as plt


class GaitEventVideoRenderer:
    def render(self, video_file, result, poses, mapper):

        pose_index = mapper.build_pose_index(poses)

        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        events = []

        for side in ("left", "right"):
            data = getattr(result, side)

            for etype, frames in (
                ("FC", data.First_contact_frames),
                ("LC", data.Last_contact_frames),
            ):
                for f in frames:
                    events.append((f / result.fps, side, f, etype))

        events.sort()

        for event_time, side, frame_idx, etype in events:
            frame_no = max(0, min(int(event_time * video_fps), total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = pose_index.get(frame_idx)

            if pose:
                heel = mapper.joint(side, "heel")
                kp = pose.get_keypoint(heel.name)

                plt.figure(figsize=(6, 4))
                plt.imshow(frame)
                plt.scatter(kp.x, kp.y, s=30)
                plt.title(f"{side} {etype}")
                plt.axis("off")
                plt.show(block=False)
                plt.pause(10)
                plt.close()

        cap.release()
