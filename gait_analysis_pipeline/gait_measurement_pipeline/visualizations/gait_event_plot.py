import matplotlib.pyplot as plt


class GaitEventPlotter:
    def plot(self, time, result):

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        for i, side in enumerate(("left", "right")):
            data = getattr(result, side)
            stance = [int(x) for x in data.stance_mask]

            axes[i].plot(time, stance, label="Stance")

            axes[i].scatter(
                [time[f] for f in data.First_contact_frames],
                [1] * len(data.First_contact_frames),
                label="FC",
            )

            axes[i].scatter(
                [time[f] for f in data.Last_contact_frames],
                [0] * len(data.Last_contact_frames),
                label="LC",
            )

            axes[i].set_title(f"{side} events")
            axes[i].grid()
            axes[i].legend()

        plt.tight_layout()
        plt.show()
