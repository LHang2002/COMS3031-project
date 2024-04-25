import json
import os
from gpt import GPT
with open("kst24k.jsonl", 'r') as file:
    json_lines = file.readlines()

gpt = GPT(model_name='gpt-4-turbo', user_name='ZhangSan', use_vpn=False, p_key="sk-xxx")
role = '''
        1.我想让你扮演数据标注的角色，任务是对一组数据中的utterance进行标注

        2.如果utterance对应的role是SERVER，需要对其action进行重新标注；
        （1）标注类型有19种，包括：继续对话、确认咨询问题、暖场、应对拒诊、询问需求、询问基本信息、询问疾病信息、询问就诊史、诊断分析、描述疾病特性、确认治疗意愿、描述治疗方案、描述医院信息、介绍价格、答疑、套电、邀约、留下联系方式、确定联系方式。
        （2）每种标注类型的定义详细说明如下：
        {
            "继续对话": "指维持对话的持续性，鼓励对方继续交谈，可能通过提问、回应或引入新话题实现。",
            "确认咨询问题": "明确对方提出的咨询内容，确保理解准确，可能需要通过重述或澄清问题来实现。",
            "暖场": "通过友好、非正式的交谈降低对话氛围的紧张感，建立舒适的对话环境。",
            "应对拒诊": "在面对患者拒绝就诊或治疗的情况下，采取的应对策略，可能包括询问原因、提供额外信息或安抚。",
            "询问需求": "明确对方在咨询或治疗中的具体需求，以便提供更准确的帮助或信息。",
            "询问基本信息": "收集对方的基本个人信息，如年龄、性别、职业、地址等，这些信息有助于了解背景和提供个性化的服务。",
            "询问疾病信息": "获取关于患者当前疾病状况的详细信息，如症状、发病时间等。",
            "询问就诊史": "了解患者以往的就医经历，包括过去的诊断、治疗和就诊记录。",
            "诊断分析": "基于收集的信息对疾病进行分析，提出可能的诊断。",
            "描述疾病特性": "向患者说明其疾病的特点，包括病因、症状、发展过程等。",
            "确认治疗意愿": "确认患者是否愿意接受建议的治疗方案或进一步的医疗操作。",
            "描述治疗方案": "向患者详细介绍治疗方法、计划、预期效果以及可能的风险。",
            "描述医院信息": "提供有关医院或诊所的相关信息，如位置、服务、专业领域等。例如SERVER的话语中出现<address>等",
            "介绍价格": "告知患者有关治疗或服务的费用信息。",
            "答疑": "回答患者的疑问，提供必要的信息和解释。",
            "套电": "在电话交谈中，通过提问或讨论，试图获取更多的信息，通常用于销售或市场调查。",
            "邀约": "邀请患者到医院或诊所进行面诊或进一步的咨询。",
            "留下联系方式": "提供或请求对方的联系方式，以便后续沟通或跟进。例如SERVER的话语中出现<mobile>等",
            "确定联系方式": "确认双方用于后续沟通的最佳联系方式和时间。例如复述CLIENT的联系方式"
        }

        3.如果utterance对应的role是CLIENT，需要对其action进行标注；
        （1） 标注种类有9种，包括：咨询问题、回复疾病信息、回复基本信息、回复就诊史、确认就诊时间、提供联系方式、暖场、继续对话、询问医院信息。
        （2）每种标注类型的定义详细说明如下：
        {
            "咨询问题": "指向对方提出具体的咨询请求，可能涉及健康问题、医疗程序、医院服务等。",
            "回复疾病信息": "针对对方关于疾病状况的提问提供回答，可能包括症状描述、疾病进展、治疗反应等。",
            "回复基本信息": "回答关于个人基本信息的查询，例如年龄、性别、职业等。例如CLIENT的话语中出现<name>或<address>等",
            "回复就诊史": "提供有关过去的医疗记录或就诊经历的信息。",
            "确认就诊时间": "确定或核实预约的就诊时间，以便于患者和医疗机构的时间安排。",
            "提供联系方式": "CLIENT的话语中有<mobile>",
            "暖场": "通过轻松友好的对话方式减轻紧张或不舒服的氛围，使对话更加流畅。例如CLIENT的话语中出现<org>等",
            "继续对话": "通过提出新的问题或话题，或是对当前话题提供更深入的见解，以保持对话的连续性。",
            "询问医院信息": "提出关于医院位置、医院设施、专家团队、服务范围等医院信息的问题。"
        }

        4.输入样例如下：[{"role": "CLIENT", "utterance": ["疤痕增生怎么办", "我有疤痕增生怎么办", "用什么药呢"], "action": []}, {"role": "SERVER", "utterance": ["这在我们皮肤专科医院很常见", "我先了解下基本情况，便于分析指导，请问性别，年龄？"], "action": []}, {"role": "CLIENT", "utterance": ["你们医院在哪里"], "action": []}, {"role": "SERVER", "utterance": ["我们在美兰区海府路上", "你留个可以接收短息电话，系统自动发到手机上详细地址 ，方便查看", "您是在海口这边吗？"], "action": []}, {"role": "CLIENT", "utterance": ["疤痕增生6年了", "<name>"], "action": []}, {"role": "SERVER", "utterance": ["嗯"], "action": []}, {"role": "CLIENT", "utterance": ["能修复吗"], "action": []}, {"role": "SERVER", "utterance": ["疤痕是如何形成的？", "目前有扩大的迹象吗？"], "action": []}, {"role": "CLIENT", "utterance": ["有"], "action": []}, {"role": "SERVER", "utterance": ["疤痕在什么部位？多大面积？"], "action": []}, {"role": "CLIENT", "utterance": ["我去点痣点出来的"], "action": []}, {"role": "SERVER", "utterance": ["可以治好的"], "action": []}, {"role": "CLIENT", "utterance": ["后来打针越来越大"], "action": []}, {"role": "SERVER", "utterance": ["嗯嗯，你有微信么？我加一下，可以发个疤痕情况的图片，我再帮你详细分析下"], "action": []}, {"role": "CLIENT", "utterance": ["<mobile>"], "action": []}, {"role": "SERVER", "utterance": ["好的", "您通过一下"], "action": []}, {"label": 1}]

        5.输出样例如下：[{"role":"CLIENT","utterance":["疤痕增生怎么办","我有疤痕增生怎么办","用什么药呢"],"action":["咨询问题","咨询问题","咨询问题"]},{"role":"SERVER","utterance":["这在我们皮肤专科医院很常见","我先了解下基本情况，便于分析指导，请问性别，年龄？"],"action":["继续对话","询问基本信息"]},{"role":"CLIENT","utterance":["你们医院在哪里"],"action":["询问医院信息"]},{"role":"SERVER","utterance":["我们在美兰区海府路上","你留个可以接收短息电话，系统自动发到手机上详细地址 ，方便查看","您是在海口这边吗？"],"action":["描述医院信息","套电","询问基本信息"]},{"role":"CLIENT","utterance":["疤痕增生6年了","<name>"],"action":["回复疾病信息","回复基本信息"]},{"role":"SERVER","utterance":["嗯"],"action":["继续对话"]},{"role":"CLIENT","utterance":["能修复吗"],"action":["咨询问题"]},{"role":"SERVER","utterance":["疤痕是如何形成的？","目前有扩大的迹象吗？"],"action":["询问疾病信息","询问疾病信息"]},{"role":"CLIENT","utterance":["有"],"action":["回复疾病信息"]},{"role":"SERVER","utterance":["疤痕在什么部位？多大面积？"],"action":["询问疾病信息"]},{"role":"CLIENT","utterance":["我去点痣点出来的"],"action":["回复疾病信息"]},{"role":"SERVER","utterance":["可以治好的"],"action":["诊断分析"]},{"role":"CLIENT","utterance":["后来打针越来越大"],"action":["回复疾病信息"]},{"role":"SERVER","utterance":["嗯嗯，你有微信么？我加一下，可以发个疤痕情况的图片，我再帮你详细分析下"],"action":["套电"]},{"role":"CLIENT","utterance":["<mobile>"],"action":["提供联系方式"]},{"role":"SERVER","utterance":["好的","您通过一下"],"action":["继续对话","继续对话"]},{"label":1}]

        6.我将发送一个输入样例，你按照以上的方法帮我进行标注并输出为输出样例格式。
        
        7.注意事项：在标注的时候utterance里面一句话只对应一个action，如果出现多个，则选择最准确的那个。
        '''

for j in range(284,300):
    gpt = GPT(model_name='gpt-4', user_name='ZhangSan', use_vpn=False, p_key="sk-nRjm3MuSfJ9yoDZu987f9f8c0b014052B463046c0587B01c")
    input =  str(json_lines[j])
        
    flag, response = gpt.call(input,role=role)

    # 将每个response作为JSON对象添加到列表中
    summary_data = response

    # 将summary_data保存到JSON文件中
    with open('kst24k_annotated.jsonl', 'a') as file:
        json.dump(summary_data, file, ensure_ascii=False)
        file.write('\n')
