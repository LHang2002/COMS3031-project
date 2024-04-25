def extract_data(read_path,data_content,write_path):
    data_list = []
    # 打开文件并逐行读取
    with open(read_path, 'r') as file:
        for line in file:
            # 检查'ori_da': 是否在当前行中
            if data_content in line:
                # 提取'ori_da': 后面的内容
                # 由于数据格式可能不规则，这里我们尽量通用地提取字符串，直到遇到下一个逗号
                start_index = line.find(data_content) + len(data_content)
                end_index = line.find(',', start_index)
                # 如果这一行的'ori_da': 后面没有其他数据，即没有逗号，则提取到行末
                if end_index == -1:
                    end_index = len(line)
                data = line[start_index:end_index].strip().strip("'\"")
                data_list.append(data)

    # 将提取的内容写入新文件，每项数据一行
    with open(write_path, 'w') as new_file:
        for data in data_list:
            new_file.write(data + '\n')

if __name__ == "__main__":
    # 使用示例
    extract_data("gdp63.txt","'ori_da':","ori_da.txt")
    extract_data("gdp63.txt","'new_da':","gdp_da.txt")
    extract_data("raw_prompt.txt","'new_da':","gpt_da.txt")
    extract_data("gdp63.txt","'ori_resp':","ori_resp.txt")
    extract_data("gdp63.txt","'new_resp':","gdp_resp.txt")
    extract_data("raw_prompt.txt","'new_resp':","gpt_resp.txt")