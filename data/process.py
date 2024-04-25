import json
import pickle

def convert_jsonl(input_file, output_file):
    converted_data = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}")
                print("Problematic line:", line)
                continue

            dialog_key = f"dialog{idx}"
            converted_data[dialog_key] = {"dialog": [], "label": []}
            dialog = converted_data[dialog_key]["dialog"]
            label = converted_data[dialog_key]["label"]

            current_dialog = {}
            current_label = {}

            for item in data:
                if 'role' not in item:
                    continue

                role = item["role"]
                utterances = item["utterance"]
                actions = item["action"]

                if role == "CLIENT" or role == "SERVER":
                    if role not in current_dialog:
                        current_dialog[role] = []
                        current_label[role] = []
                    current_dialog[role].extend(utterances)
                    current_label[role].extend(actions)

                # Check if both CLIENT and SERVER have spoken, then add to dialog
                if "CLIENT" in current_dialog and "SERVER" in current_dialog:
                    dialog.append(current_dialog)
                    label.append(current_label)
                    current_dialog = {}
                    current_label = {}

            # Adding any remaining conversation
            if "CLIENT" in current_dialog or "SERVER" in current_dialog:
                dialog.append(current_dialog)
                label.append(current_label)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

def modify_json(file_path):
    # 读取原始的 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历所有的 dialog 数据组
    for key, value in data.items():
        dialog = value.get("dialog", [])
        labels = value.get("label", [])

        # 创建新的 dialog 和 label 列表
        new_dialog = [{"SERVER": ["你好，有什么需要帮助的吗"], "CLIENT": dialog[0]["CLIENT"]}]
        new_labels = [{"SERVER": ["暖场"], "CLIENT": labels[0]["CLIENT"]}]

        # 遍历原有的 dialog 和 label，重新排列 SERVER 和 CLIENT
        for i in range(0, len(dialog) - 1):
            # 添加当前的 SERVER 和其标签
            new_dialog.append({"SERVER": dialog[i]["SERVER"], "CLIENT": []})
            new_labels.append({"SERVER": labels[i]["SERVER"], "CLIENT": []})

            # 添加下一个 CLIENT 和其标签
            new_dialog[-1]["CLIENT"] = dialog[i + 1]["CLIENT"]
            new_labels[-1]["CLIENT"] = labels[i + 1]["CLIENT"] if i + 1 < len(labels) else []

        # 更新 dialog 和 label 数据
        data[key] = {"dialog": new_dialog, "label": new_labels}

    # 将修改后的数据保存到新的 JSON 文件中
    with open('temp2.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_as_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    convert_jsonl('kst63.jsonl','temp1.json')
    modify_json('temp1.json')
    save_as_pickle(read_json('temp2.json'),'kst63.pkl')