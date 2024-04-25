def extract_all_data(read_path,write_path):
    # 打开原始文件
    with open(read_path, 'r') as file:
        # 读取文件的唯一一行
        line = file.readline()

    # 使用Python的字符串方法find来查找所有的'winner':出现的位置
    start = 0
    winners = []

    while True:
        start = line.find("'winner':", start)
        if start == -1:  # 如果没有找到更多的'winner':, 跳出循环
            break
        # 找到了一个'winner':, 把它加入到winners列表中
        end = line.find("}", start)  # 假设每个'winner':后面都跟着一个逗号，这样我们就可以定位到这一段的结尾
        winners.append(line[start+10:end])
        start = end  # 更新start位置，为下一次查找做准备

    # 将找到的所有'winner':写入到新的txt文件中
    with open(write_path, 'w') as file:
        for winner in winners:
            file.write(winner + '\n')  # 每写入一个'winner':就换行


if __name__ == "__main__":
    extract_all_data('gdp_raw.txt','gdp_raw_win.txt')
    extract_all_data('raw_human.txt','raw_human_win.txt')
