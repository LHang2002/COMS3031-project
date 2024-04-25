import numpy as np
import logging
import pickle
import argparse
import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["OPENAI_API_BASE"] = "https://xxx"
from tqdm.auto import tqdm
from core.gen_models_kst import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players_kst import (
	ClientModel, ServerModel, KstSystemPlanner,
	ServerChatModel, ClientChatModel, KstChatSystemPlanner
)
from core.game_kst import KstGame
from core.helpers import DialogSession
from utils.prompt_examples_kst import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	system_name = KstGame.SYS
	user_name = KstGame.USR
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	if cmd_args.llm == 'code-davinci-002':
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

	game_ontology = KstGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']

	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1]
	)
	user = UsrModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
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
	game = KstGame(system, user)

	with open("/home/linyihang/COMS3031-project/data/kst63.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = 63

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models
	bad_dialogs = []
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			print("skipping dialog id: ", did)
			continue
		if num_done == num_dialogs:
			break

		print("evaluating dialog id: ", did)
		context = ""
		no_error = True
		dialog = all_dialogs[did]
		
		state = game.init_dialog()
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["CLIENT"]) == 0:  # ended
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["CLIENT"]).strip()
			usr_da = dialog["label"][t]["CLIENT"][-1]

			# map to our dialog act
			if usr_da == "咨询问题":
				usr_da = KstGame.U_INQUIRY_QUESTION
			elif usr_da == "回复疾病信息":
				usr_da = KstGame.U_REPLY_DISEASE_INFO
			elif usr_da == "回复基本信息":
				usr_da = KstGame.U_REPLY_BASIC_INFO
			elif usr_da == "回复就诊史":
				usr_da = KstGame.U_REPLY_MEDICAL_HISTORY
			elif usr_da == "确认就诊时间":
				usr_da = KstGame.U_CONFIRM_APPOINTMENT_TIME
			elif usr_da == "提供联系方式":
				usr_da = KstGame.U_PROVIDE_CONTACT
			elif usr_da == "暖场":
				usr_da = KstGame.U_WARM_UP
			elif usr_da == "询问医院信息":
				usr_da = KstGame.U_INQUIRE_HOSPITAL_INFO
			else:
				usr_da = KstGame.U_CONTINUE_CONVERSATION

			# game ended
			if usr_da == KstGame.U_PROVIDE_CONTACT:
				break

			# map sys as well
			sys_utt = " ".join(turn["SERVER"]).strip()
			sys_da = set(dialog["label"][t]["SERVER"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da = "继续对话"
			else:
				sys_da = list(intersected_das)[-1]
			
			state.add_single(KstGame.SYS, sys_da, sys_utt)
			state.add_single(KstGame.USR, usr_da, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			SERVER: {sys_utt}
			Client: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# mcts policy
			prior, v = planner.predict(state)
			greedy_policy = system.dialog_acts[np.argmax(prior)]
			try:
				next_best_state = game.get_next_state(state, np.argmax(prior))
			except Exception as e:
				bad_dialogs.append(did)
				no_error = False
				raise e
			greedy_pred_resp = next_best_state.history[-2][2]

			# next ground truth utterance
			human_resp = " ".join(dialog["dialog"][t + 1]["SERVER"]).strip()
			next_sys_das = set(dialog["label"][t+1]["SERVER"])
			next_intersected_das = next_sys_das.intersection(system.dialog_acts)
			if len(next_intersected_das) == 0:
				next_sys_da = "继续对话"
			else:
				next_sys_da = list(next_intersected_das)[-1]

			# logging for debug
			debug_data = {
				"prior": prior,
				"da": greedy_policy,
				"v": v
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': greedy_pred_resp,
				'new_da': greedy_policy,
				"debug": debug_data,
			}
			output.append(cmp_data)
		
		if no_error:
			with open(cmd_args.output, "wb") as f:
				pickle.dump(output, f)
			pbar.update(1)
			num_done += 1
	pbar.close()
	print(bad_dialogs)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--llm', type=str, default="gpt-3.5-turbo", choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt","Baichuan2-7B-Chat"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences to generate. -1 for no limit')
	parser.add_argument('--output', type=str, default="outputs/raw_prompt63.pkl", help='output file')
	parser.parse_args()
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)

	main(cmd_args)