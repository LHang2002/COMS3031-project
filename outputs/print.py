import pickle

def print_pkl_to_readable_file(pkl_file_path, output_file_path):
    # Load the data from the .pkl file
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    
    # Print the loaded data to an output file
    with open(output_file_path, 'w') as output_file:
        print(data, file=output_file)

if __name__ == "__main__":
    # 示例文件路径，您需要根据实际情况修改这些路径
    pkl_file_path = 'gdp63.pkl'
    output_file_path = 'gdp63.txt'
    
    # 调用函数将.pkl文件的内容打印到一个可读的文件中
    print_pkl_to_readable_file(pkl_file_path, output_file_path)
