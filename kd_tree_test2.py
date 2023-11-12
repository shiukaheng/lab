from rtt_star import *

r = RTTStar(
    2,
    np.array([0, 0]),
    lower_bounds=np.array([-10, -10]),
    upper_bounds=np.array([10, 10]),
    collision_samples=100
)

i = 0
while True:
    i += 1
    print(i)
    r.sample()