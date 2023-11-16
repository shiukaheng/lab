import numpy as np
from bezier import Bezier

class PiecewiseBezier:
    def __init__(self, pointlist, t_min=0.0, t_max=1.0, mult_t=1.0, segments=1):
        if segments < 1:
            raise ValueError("Number of segments must be at least 1")

        self.segments = segments
        self.segment_curves = []

        # Calculate the number of points per segment (including overlapping points)
        points_per_segment = (len(pointlist) - 1) // segments + 1

        # Create each segment
        for i in range(segments):
            start_idx = i * (points_per_segment - 1)
            end_idx = start_idx + points_per_segment
            segment_points = pointlist[start_idx:end_idx]

            # Adjust time bounds for each segment
            segment_t_min = t_min + (t_max - t_min) * i / segments
            segment_t_max = t_min + (t_max - t_min) * (i + 1) / segments

            # Create Bezier segment
            self.segment_curves.append(Bezier(segment_points, segment_t_min, segment_t_max, mult_t))

    def __call__(self, t):
        if not (self.segment_curves[0].T_min_ <= t <= self.segment_curves[-1].T_max_):
            raise ValueError("Time t is out of range")

        # Determine which segment the time t belongs to
        segment_index = min(int((t - self.segment_curves[0].T_min_) / (self.segment_curves[-1].T_max_ - self.segment_curves[0].T_min_) * self.segments), self.segments - 1)
        return self.segment_curves[segment_index](t)

    def derivative(self, order):
        # Compute the derivative for each segment and return a new PiecewiseBezier
        derived_segments = [segment.derivative(order) for segment in self.segment_curves]
        return PiecewiseBezier(derived_segments, self.segment_curves[0].T_min_, self.segment_curves[-1].T_max_, self.segment_curves[0].mult_T_, self.segments)
