from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    DetectedEvents,
)


class GaitEventDetectorCore:
    """
    Pure algorithm:
    - stance cleaning assumed already done
    - no pose / no visualization
    """

    def detect(self, stance: list[bool]) -> DetectedEvents:
        fc, lc = [], []

        for i in range(1, len(stance)):
            if not stance[i - 1] and stance[i]:
                fc.append(i)
            elif stance[i - 1] and not stance[i]:
                lc.append(i - 1)

        return DetectedEvents(fc, lc)

    def pair(self, events: DetectedEvents) -> DetectedEvents:
        paired_fc, paired_lc = [], []
        j = 0

        for fc in events.First_contact:
            while j < len(events.Last_contact) and events.Last_contact[j] <= fc:
                j += 1
            if j < len(events.Last_contact):
                paired_fc.append(fc)
                paired_lc.append(events.Last_contact[j])
                j += 1

        return DetectedEvents(paired_fc, paired_lc)
