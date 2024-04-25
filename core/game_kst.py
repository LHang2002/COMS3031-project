import numpy as np
import logging

from core.gen_models_kst import DialogModel
from core.helpers import DialogSession
from abc import ABC, abstractmethod
from typing import List


logger = logging.getLogger(__name__)


class DialogGame(ABC):
	def __init__(self, 
			system_name:str, system_agent:DialogModel, 
			user_name: str, user_agent:DialogModel):
		self.SYS = system_name
		self.system_agent = system_agent
		self.USR = user_name
		self.user_agent = user_agent
		return

	@staticmethod
	@abstractmethod
	def get_game_ontology() -> dict:
		"""returns game related information such as dialog acts, slots, etc.
		"""
		raise NotImplementedError

	def init_dialog(self) -> DialogSession:
		# [(sys_act, sys_utt, user_act, user_utt), ...]
		return DialogSession(self.SYS, self.USR)

	def get_next_state(self, state:DialogSession, action) -> DialogSession:
		next_state = state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		
		# state in user's perspective
		user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None)  # user just reply
		next_state.add_single(state.USR, user_da, user_resp)
		return next_state
	
	def get_next_state_batched(self, state:DialogSession, action, batch=3) -> List[DialogSession]:
		all_next_states = [state.copy() for _ in range(batch)]

		sys_utts = self.system_agent.get_utterance_batched(state.copy(), action, batch)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		for i in range(batch):
			all_next_states[i].add_single(state.SYS, sys_da, sys_utts[i])
		
		# state in user's perspective
		user_das, user_resps = self.user_agent.get_utterance_w_da_from_batched_states(all_next_states, None)  # user just reply
		for i in range(batch):
			all_next_states[i].add_single(state.USR, user_das[i], user_resps[i])
		return all_next_states

	def display(self, state:DialogSession):
		string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
		print(string_rep)
		return

	@abstractmethod
	def get_dialog_ended(self, state) -> float:
		"""returns 0 if not ended, then (in general) 1 if system success, -1 if failure 
		"""
		raise NotImplementedError


class KstGame(DialogGame):
	SYS = "Server"
	USR = "Client"

	# 定义系统对话行为
	S_CONTINUE_DIALOG = "继续对话"
	S_CONFIRM_INQUIRY = "确认咨询问题"
	S_WARMING_UP = "暖场"
	S_REFUSE = "应对拒诊"
	S_INQUIRY_DEMAND = "询问需求"
	S_INQUIRY_BASIC_INFO = "询问基本信息"
	S_INQUIRY_DISEASE_INFO = "询问疾病信息"
	S_INQUIRY_MEDICAL_HISTORY = "询问就诊史"
	S_DIAGNOSTIC_ANALYSIS = "诊断分析"
	S_DESCRIBE_DISEASE_FEATURES = "描述疾病特性"
	S_CONFIRM_TREATMENT_INTENTION = "确认治疗意愿"
	S_DESCRIBE_TREATMENT_PLAN = "描述治疗方案"
	S_DESCRIBE_HOSPITAL_INFO = "描述医院信息"
	S_INTRODUCE_PRICE = "介绍价格"
	S_ANSWER_QUESTION = "答疑"
	S_TRICK_CALL = "套电"
	S_INVITATION = "邀约"
	S_LEAVE_CONTACT = "留下联系方式"
	S_CONFIRM_CONTACT = "确定联系方式"

    # 定义用户对话行为
	U_INQUIRY_QUESTION = "咨询问题"
	U_REPLY_DISEASE_INFO = "回复疾病信息"
	U_REPLY_BASIC_INFO = "回复基本信息"
	U_REPLY_MEDICAL_HISTORY = "回复就诊史"
	U_CONFIRM_APPOINTMENT_TIME = "确认就诊时间"
	U_PROVIDE_CONTACT = "提供联系方式"
	U_WARM_UP = "暖场"
	U_CONTINUE_CONVERSATION = "继续对话"
	U_INQUIRE_HOSPITAL_INFO = "询问医院信息"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel, 
			max_conv_turns=15):
		super().__init__(KstGame.SYS, system_agent, KstGame.USR, user_agent)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
		return {
			"system": {
				"dialog_acts": [
					KstGame.S_CONTINUE_DIALOG, KstGame.S_CONFIRM_INQUIRY, KstGame.S_WARMING_UP, 
					KstGame.S_REFUSE, KstGame.S_INQUIRY_DEMAND, KstGame.S_INQUIRY_BASIC_INFO,
					KstGame.S_INQUIRY_DISEASE_INFO, KstGame.S_INQUIRY_MEDICAL_HISTORY, KstGame.S_DIAGNOSTIC_ANALYSIS,
					KstGame.S_DESCRIBE_DISEASE_FEATURES, KstGame.S_CONFIRM_TREATMENT_INTENTION, KstGame.S_DESCRIBE_TREATMENT_PLAN,
					KstGame.S_DESCRIBE_HOSPITAL_INFO, KstGame.S_INTRODUCE_PRICE, KstGame.S_ANSWER_QUESTION, KstGame.S_TRICK_CALL,
					KstGame.S_INVITATION, KstGame.S_LEAVE_CONTACT, KstGame.S_CONFIRM_CONTACT
				],
			},
			"user": {
				"dialog_acts": [
					KstGame.U_CONFIRM_APPOINTMENT_TIME,
					KstGame.U_CONTINUE_CONVERSATION,KstGame.U_INQUIRE_HOSPITAL_INFO,
					KstGame.U_INQUIRY_QUESTION,KstGame.U_PROVIDE_CONTACT,
					KstGame.U_REPLY_BASIC_INFO,KstGame.U_REPLY_DISEASE_INFO,
					KstGame.U_REPLY_MEDICAL_HISTORY,KstGame.U_WARM_UP
                ]
            }
        }

	def get_dialog_ended(self, state) -> float:
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended due to maximum conversation turns reached")
			return -1.0
		for (_, da, _) in state:
			if da == KstGame.U_PROVIDE_CONTACT or da == KstGame.U_CONFIRM_APPOINTMENT_TIME:
				logger.info("Dialog ended with successful outcome")
				return 1.0
		return 0.0