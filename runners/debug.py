import numpy as np
import logging
import pickle
import argparse
import numpy as np
import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["OPENAI_API_BASE"] = "https://xxx"
import sys
sys.path.append("..")

from tqdm.auto import tqdm
from core.gen_models_kst import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players_kst import (
    ClientModel, ServerModel, KstSystemPlanner,
    ServerChatModel, ClientChatModel, KstChatSystemPlanner
)
from core.game_kst import KstGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples_kst import EXP_DIALOG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def init_models(cmd_args):
    if cmd_args.llm in ['code-davinci-002']:
        backbone_model = OpenAIModel(cmd_args.llm)
        SysModel = ServerModel
        UsrModel = ClientModel
        SysPlanner = KstSystemPlanner
    elif cmd_args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = ServerChatModel
        UsrModel = ClientChatModel
        SysPlanner = KstChatSystemPlanner
    elif cmd_args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = ServerChatModel
        UsrModel = ClientChatModel
        SysPlanner = KstChatSystemPlanner
    elif cmd_args.llm == 'Baichuan2-7B-Chat':
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = ServerChatModel
        UsrModel = ClientChatModel
        SysPlanner = KstChatSystemPlanner
    return backbone_model, SysModel, UsrModel, SysPlanner


def get_system_user_planner(backbone_model, SysModel, UsrModel, SysPlanner, sys_da, user_da, exp_1):
    system = SysModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,  # for MCTS open loop
            "return_full_text": False,
        }
    )
    user = UsrModel(
        user_da,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,
            "repetition_penalty": 1.0,
            "do_sample": True,  # for MCTS open loop
            "return_full_text": False,
        },
        backbone_model=backbone_model,
        conv_examples=[exp_1]
    )

    planner = SysPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1]
    )
    return system, user, planner


def load_dialogue_data(dialogue_data_file="/home/linyihang/COMS3031-project/data/kst.pkl"):
    with open(dialogue_data_file, "rb") as f:
        all_dialogs = pickle.load(f)
    return all_dialogs

def create_debug_data(mcts_policy, mcts_policy_next_da, dialog_planner):
    """
    创建并返回调试数据字典。

    参数:
    - mcts_policy: MCTS策略的概率列表。
    - mcts_policy_next_da: MCTS策略下一个对话行为的标识。
    - dialog_planner: 对话规划器实例。

    返回:
    - 一个包含调试信息的字典。
    """
    debug_data = {
        "probs": mcts_policy,
        "da": mcts_policy_next_da,
        "search_tree": {
            "Ns": dialog_planner.Ns,
            "Nsa": dialog_planner.Nsa,
            "Q": dialog_planner.Q,
            "P": dialog_planner.P,
            "Vs": dialog_planner.Vs,
            "realizations": dialog_planner.realizations,
            "realizations_Vs": dialog_planner.realizations_Vs,
            "realizations_Ns": dialog_planner.realizations_Ns,
        },
    }
    return debug_data

def create_comparison_data(did, context, human_resp, next_sys_da, mcts_pred_rep, mcts_policy_next_da, debug_data):
    """
    创建并返回用于比较的数据字典。

    参数:
    - did: 对话ID。
    - context: 对话上下文。
    - human_resp: 人类响应。
    - next_sys_da: 下一个系统对话行为。
    - mcts_pred_rep: MCTS预测的响应。
    - mcts_policy_next_da: MCTS策略下一个对话行为。
    - debug_data: 调试数据。

    返回:
    - 一个包含比较信息的字典。
    """
    cmp_data = {
        'did': did,
        'context': context,
        'ori_resp': human_resp,
        'ori_da': next_sys_da,
        'new_resp': mcts_pred_rep,
        'new_da': mcts_policy_next_da,
        "debug": debug_data,
    }
    return cmp_data


def configure_mcts_args(cmd_args):
    """
    根据命令行参数配置并返回MCTS的参数。

    参数:
    - cmd_args: 命令行参数对象。

    返回:
    - 配置后的MCTS参数的dotdict对象。
    """
    args = dotdict({
        "cpuct": 1.0,
        "num_MCTS_sims": cmd_args.num_mcts_sims,
        "Q_0": cmd_args.Q_0,
        "max_realizations": cmd_args.max_realizations,
    })
    return args


def main(cmd_args):
    game_ontology = KstGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = KstGame.SYS
    user_name = KstGame.USR
    num_dialogs = cmd_args.num_dialogs
    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

    backbone_model, SysModel, UsrModel, SysPlanner = init_models(cmd_args)
    system, user, planner = get_system_user_planner(backbone_model, SysModel, UsrModel, SysPlanner, sys_da, user_da,
                                                    exp_1)
    game = KstGame(system, user)
    all_dialogs = load_dialogue_data()
    args = configure_mcts_args(cmd_args)
    output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
    num_done = 0
    pbar = tqdm(total=num_dialogs, desc="evaluating")
    for did in all_dialogs.keys():
        if num_done == num_dialogs:
            break
        print("evaluating dialog id: ", did)
        context = ""
        dialog = all_dialogs[did]
        print("dialog: ", dialog)
        state = game.init_dialog()
        for t, turn in enumerate(dialog["dialog"]):

            # 判定对话结束的特殊条件
            if len(turn["CLIENT"]) == 0:  # ended
                break
            # also skip last turn as there is no evaluation
            if t == len(dialog["dialog"]) - 1:
                break

            usr_utt = " ".join(turn["CLIENT"]).strip()
            usr_da = dialog["label"][t]["CLIENT"][-1]
            print("usr_utt: ", usr_utt)
            print("usr_da: ", usr_da)
            print("state: ", state.to_string_rep(True,True))
            print("mapped_usr_da: ", usr_da)
            # 用户提供了联系方式，也会结束
            if usr_da == KstGame.U_PROVIDE_CONTACT:
                break

            # map sys as well
            sys_utt = " ".join(turn["SERVER"]).strip()
            sys_da = set(dialog["label"][t]["SERVER"])
            print("sys_utt: ", sys_utt)
            print("sys_da: ", sys_da)

            intersected_das = sys_da.intersection(system.dialog_acts)
            print("intersected_das: ")
            print(intersected_das)
            if len(intersected_das) == 0:
                sys_da = "继续对话"
            else:
                sys_da = list(intersected_das)[-1]

            state.add_single(KstGame.SYS, sys_da, sys_utt)
            state.add_single(KstGame.USR, usr_da, usr_utt)
            print("state: ", state.to_string_rep(True,True))
            # 这里将会加入标准的上文。

            # update context for evaluation
            context = f"""
            {context}
            Server: {sys_utt}
            Client: {usr_utt}
            """
            context = context.replace('\t', '').strip()
            print(context)
            # 这里初始化完成，开始进入MCTS过程。

            # mcts policy
            if isinstance(backbone_model, OpenAIModel):
                backbone_model._cached_generate.cache_clear()
            dialog_planner = OpenLoopMCTS(game, planner, args)
            print("searching")
            print(args.num_MCTS_sims)

            for i in tqdm(range(args.num_MCTS_sims)):
                dialog_planner.search(state)
                print(state.to_string_rep())

            mcts_policy = dialog_planner.get_action_prob(state)
            mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]

            # # fetch the generated utterance from simulation
            mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

            # next ground truth utterance
            human_resp = " ".join(dialog["dialog"][t + 1]["SERVER"]).strip()
            next_sys_das = set(dialog["label"][t + 1]["SERVER"])
            next_intersected_das = next_sys_das.intersection(system.dialog_acts)
            if len(next_intersected_das) == 0:
                next_sys_da = "继续对话"
            else:
                next_sys_da = list(next_intersected_das)[-1]

            # logging for debug
            debug_data = create_debug_data(mcts_policy, mcts_policy_next_da, dialog_planner)
            # update data
            cmp_data = create_comparison_data(did, context, human_resp, next_sys_da, mcts_pred_rep, mcts_policy_next_da, debug_data)
            output.append(cmp_data)

            if cmd_args.debug:
                print(context)
                print("human resp: ", human_resp)
                print("human da: ", next_sys_da)
                print("mcts resp: ", mcts_pred_rep)
                print("mcts da: ", mcts_policy_next_da)
        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)
        num_done += 1
        pbar.update(1)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="outputs/debug.pkl", help='output file')
    parser.add_argument('--llm', type=str, default="gpt-3.5-turbo", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo","Baichuan2-7B-Chat"], help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=10, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.25, help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.parse_args()
    cmd_args = parser.parse_args()
    print("saving to", cmd_args.output)

    main(cmd_args)
