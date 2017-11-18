import json

t_file = "time2cars_map.json"
c_file = "car_results.json"

time_map = None
car_map = None

with open(t_file, 'r') as f:
	time_map = json.load(f)[0]

s_time = float(min(time_map.keys()))

# 4 windows of 15 mins each
NUM_WINDOWS = 4
detection_window = 15 * 60
d_times = [0] * NUM_WINDOWS

s_keys = sorted(time_map.keys())

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

windows = split_list(s_keys, NUM_WINDOWS)

max_list = [max(time_map[max(window)])[1] for window in windows]
car_count_results = list()
prev_count = 0
for i in range(0, len(max_list)):
	hash_idx = max_list[i].index("#")
	num_cars = int(max_list[i][hash_idx+1::])
	car_count_results.append(num_cars - prev_count)
	prev_count = num_cars

# Printing cars every detection window.
print("Printing number of cars every 15 mins and total cars until then.\n")
rolling_sum = 0
print("#Cars in Window", "#Cars since start")
for num in car_count_results:
	rolling_sum += num
	print(num, rolling_sum)

print("\nTotal cars =", rolling_sum)