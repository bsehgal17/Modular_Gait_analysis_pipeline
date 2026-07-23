import matplotlib.pyplot as plt


def plot_velocity_events(
    time, left_vel, right_vel, left_hs, right_hs, left_to, right_to
):

    plt.figure(figsize=(12, 5))

    plt.plot(time, left_vel, label="Left Foot Velocity")
    plt.plot(time, right_vel, label="Right Foot Velocity")

    plt.scatter(time[left_hs], left_vel[left_hs], marker="o", label="Left HS")
    plt.scatter(time[right_hs], right_vel[right_hs], marker="o", label="Right HS")

    plt.scatter(time[left_to], left_vel[left_to], marker="x", label="Left TO")
    plt.scatter(time[right_to], right_vel[right_to], marker="x", label="Right TO")

    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title("Velocity Based Gait Event Detection")

    plt.legend()
    plt.grid()

    plt.show()


def plot_ap_events(time, left_signal, right_signal, left_events, right_events, title):

    plt.figure(figsize=(12, 5))

    plt.plot(time, left_signal, label="Left")
    plt.plot(time, right_signal, label="Right")

    plt.scatter(
        time[left_events], left_signal[left_events], marker="o", label="Left Events"
    )
    plt.scatter(
        time[right_events], right_signal[right_events], marker="o", label="Right Events"
    )

    plt.xlabel("Time")
    plt.ylabel("AP Distance")
    plt.title(title)

    plt.legend()
    plt.grid()

    plt.show()
