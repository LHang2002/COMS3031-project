import re
import json

def func1():
    def replace_escaped_quotes(input_string):
        # 使用正则表达式匹配 \" 并替换为 "
        return re.sub(r'\\\"', '\"', input_string)

    with open('kst24k_annotated.jsonl','r',encoding='utf-8') as fin,\
        open('afterdecode.jsonl','w',encoding='utf-8') as fout:
        lines=fin.readlines()
        for line in lines:
            data=replace_escaped_quotes(line)
            fout.write(data[1:-2])
            fout.write('\n')

def func2():
    def filter_jsonl_with_regex():
    # 编译一个正则表达式来匹配包含"label":0或"label":1的行
        pattern = re.compile(r'"label":\s*(0|1)')
        
        with open('afterdecode.jsonl', 'r', encoding='utf-8') as infile, open("output.jsonl", 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 使用正则表达式检查每一行
                if pattern.search(line):
                    # 如果匹配，则写入到输出文件
                    outfile.write(line)
    filter_jsonl_with_regex()

def func3():
    def split_jsonl_to_multiline():
        with open("output.jsonl", 'r', encoding='utf-8') as infile, open("data.jsonl", 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    # 将当前行解析为JSON对象
                    data = json.loads(line)
                    # 美化JSON对象，indent参数指定缩进量，这将使得JSON对象分布在多行
                    pretty_json = json.dumps(data, indent=4,ensure_ascii=False)
                    # 将美化后的JSON写入输出文件，并在每个JSON对象后添加一个换行符以分隔
                    outfile.write(pretty_json + '\n\n')
                except json.JSONDecodeError:
                    # 如果当前行不是有效的JSON，打印错误信息（可选）
                    print("Invalid JSON detected and skipped.")
    split_jsonl_to_multiline()


if __name__ == '__main__':
    func2()