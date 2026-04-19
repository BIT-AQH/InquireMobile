import os
import json

threshold = 0.1
num = 0
num_error = 0
num_type_error = 0
action_map = {}
# tag_map = {}
action_pair_map = {}
error_list = []

ds_map = {
    "input_text": "type",
    "scroll": "swipe",
    "navigate_back": "system_button",
    "open_app": "open"
}

def stat_error(d, action, tags):
    global num_error
    num_error += 1
    action_map[action][1] += 1
    if tags:
        for t in tags:
            tag_map[t][1] += 1
    error_list.append((d[0], d[1]))

w = 1080
h = 2400
for line in open('/tmp/saves/pred.tsv'):
    t = line.strip().split('\t')
    num += 1
    resized_width = int(t[1])
    resized_height = int(t[2])
    label = json.loads(t[3])
    pred = json.loads(t[4])['arguments']
    action = label['action_type']
    action2 = pred['action']
    key = action + '_' + action2
    action_pair_map[key] = action_pair_map.get(key, 0) + 1

    # tags = t[2].split(',')
    # for tag in tags:
    #     if tag not in tag_map:
    #         tag_map[tag] = [0, 0]
    #     tag_map[tag][0] += 1
    tags = None

    if action not in action_map:
        action_map[action] = [0, 0]
    action_map[action][0] += 1

    if action2 != ds_map.get(action, action):
        stat_error(t, action, tags)
        num_type_error += 1
        continue
    if action in ['click', 'long_press']:
        x1 = label['x']
        y1 = label['y']
        x2, y2 = pred['coordinate']
        x2 = round(x2*w/resized_width)
        y2 = round(y2*h/resized_height)

        if abs(x1-x2)/w >= threshold or abs(y1-y2)/h >= threshold:
            stat_error(t, action, tags)
            continue
    if action == 'open_app':
        x1 = label['app_name'].strip().lower()
        x2 = pred['text'].strip().lower()
        if x1 != x2:
            stat_error(t, action, tags)
            continue
    if action == 'input_text':
        x1 = label['text'].strip().lower()
        x2 = pred['text'].strip().lower()
        if x1 != x2:
            stat_error(t, action, tags)
            continue
    if action == 'scroll':
        x1, y1 = pred['coordinate']
        x2, y2 = pred['coordinate2']
        d = ''
        if x1 < x2:
            d = 'right'
        elif x1 > x2:
            d = 'left'
        elif y1 < y2:
            d = 'down'
        elif y1 > y2:
            d = 'up'
        if label['direction'].strip().lower() != d:
            stat_error(t, action, tags)
            continue
    if action == 'navigate_back':
        if pred['button'].lower() != 'back':
            stat_error(t, action, tags)
            continue

print(num)
print(num_error)
print(num_type_error)
print(action_map)
# print(tag_map)
print(json.dumps(action_pair_map, indent=4))

for action in action_map:
    print(action, action_map[action], round(action_map[action][1]*1.0/action_map[action][0], 4))