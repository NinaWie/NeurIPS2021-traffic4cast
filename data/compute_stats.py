from util.h5_util import write_data_to_h5, load_h5_file
import json
import os
import numpy as np

BASE_FOLDER = "raw"

year = 2019
with open(f"weekday2dates_{year}.json", "r") as outfile:
    weekday_dict = json.load(outfile)

all_data = ["BERLIN", "CHICAGO", "MELBOURNE", "ISTANBUL"]  # ["ANTWERP", "BANGKOK", "BARCELONA", "MOSCOW"]
# "NEWYORK", "VIENNA" spatial transfer
for city in all_data:
    for weekday in range(7):
        if os.path.exists(f"stats/{city}_{year}_{weekday}.h5"):
            print("exists")
            continue
        dates_per_weekday = weekday_dict[str(weekday)]
        print("will process nr of files", len(dates_per_weekday))
        mean_weekday = np.zeros((len(dates_per_weekday), 288, 495, 436, 8)).astype(np.uint8)
        for i, date in enumerate(dates_per_weekday):
            fpath = f"{BASE_FOLDER}/{city}/training/{date}_{city}_8ch.h5"
            print(i, fpath)
            mean_weekday[i] = load_h5_file(fpath)
        mean_weekday = np.mean(mean_weekday, axis=0).astype(np.uint8)
        write_data_to_h5(mean_weekday, f"stats/{city}_{year}_{weekday}.h5")
        print("SAVED FILE", f"stats/{city}_{year}_{weekday}.h5")

# # FOR RUNNING AVG
# mean_weekday = np.zeros((288, 495, 436, 8))  # .astype(np.int) # len(dates_per_weekday),
# for i, date in enumerate(dates_per_weekday):
#     fpath = f"{BASE_FOLDER}/{city}/training/{date}_{city}_8ch.h5"
#     print(i, fpath)
#     # mean_weekday[i] = load_h5_file(fpath)
#     new_data = load_h5_file(fpath).astype(float)
#     mean_weekday = mean_weekday * (i / (i + 1)) + new_data / (i + 1)