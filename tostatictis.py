import glob
import json
import os
import re
from statistics import mean

out_dir = "/mnt/cfs/weibo_comments/processed"
out_log_dir = "/mnt/cfs/weibo_comments/processed_log"

total_convs_num = 0
twopersons = 0
multipersons = 0
suitable_convs = 0
for file in glob.glob(out_dir + '/**.json'):
    convs = json.load(open(file, 'r'))[0]
    total_convs_num += len(convs)
    for line in convs:
        if len(set(line['senders'])) == 2:
            twopersons += 1
            if len(line['senders']) > 2:
                suitable_convs += 1
        else:
            multipersons += 1

print(total_convs_num)
print(twopersons)
print(multipersons)

# total_convs_num = 0
# twoperson_multirounds = []
# twoperson_multirounds_per = []
# twopersons = []
# multipersons = []
# for log_file in glob.glob(out_log_dir + '/**.log'):
#     log = open(log_file).readlines()
#     if len(log) == 2:
#         twoperson_multirounds_per.extend(re.findall(r"多轮两人对话占比: ([0-9.]+)", log[1]))
#         twopersons.extend(re.findall(r"两人对话数目: ([0-9.]+)", log[1]))
#         multipersons.extend(re.findall(r"多人对话数目: ([0-9.]+)", log[1]))
#         twoperson_multirounds.append(float(twoperson_multirounds_per[-1]) * int(twopersons[-1]))
#     else:
#         twoperson_multirounds_per.extend(re.findall(r"多轮两人对话占比: ([0-9.]+)", log[-1]))
#         twopersons.extend(re.findall(r"两人对话数目: ([0-9.]+)", log[-1]))
#         multipersons.extend(re.findall(r"多人对话数目: ([0-9.]+)", log[-1]))
#         twoperson_multirounds.append(float(twoperson_multirounds_per[-1]) * int(twopersons[-1]))

# twoperson_multirounds_per = [float(i) for i in twoperson_multirounds_per]
# twopersons = [int(i) for i in twopersons]
# multipersons = [int(i) for i in multipersons]


# total_convs_num = sum(twopersons) + sum(multipersons)

# print(total_convs_num)
# print(mean(twoperson_multirounds_per))
# print(sum(twoperson_multirounds))
# print(sum(twopersons))


# out_dir = "/mnt/cfs/weibo_comments/formatted"
# out_log_dir = "/mnt/cfs/weibo_comments/log"

# total_convs_num = 0
# for file in glob.glob(out_dir + '/**.json'):
#     total_convs_num += len(json.load(open(file, 'r')))

# print(total_convs_num)
