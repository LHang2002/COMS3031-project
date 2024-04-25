import logging
import numpy as np

from typing import List, Tuple
from core.helpers import DialogSession
from core.gen_models_kst import GenerationModel, DialogModel
from core.game_kst import KstGame
from abc import ABC, abstractmethod
from collections import Counter


logger = logging.getLogger(__name__)


class DialogPlanner(ABC):
	@abstractmethod
	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		pass

	@abstractmethod
	def predict(self, state) -> "Tuple[np.ndarray, float]":
		# returns a prob and value
		pass


class KstSystemPlanner(DialogPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__()
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
		self.user_dialog_acts = user_dialog_acts
		self.user_max_hist_num_turns = user_max_hist_num_turns  # used in heuristic function
		self.conv_examples = conv_examples
		self.generation_model = generation_model
		self.smoothing = 1.0
		self.task_prompt = f"""
		以下是医疗咨询场景的背景信息。
		在这个场景中，Server是一位医疗专业人员，如医生或护士，其角色是向寻求医疗帮助的Client提供医疗建议、诊断健康问题以及讨论治疗方案。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		在对话过程中，Server可以选择以下行动：
		{" ".join([f"[{da}]" for da in dialog_acts])}
		以下是一个Server与Client在医疗咨询背景下的示例对话，其中Server正在处理Client的健康问题并建议可能的治疗方案，并让Client留下联系方式。
		{self.process_exp()}
		以下是另一位Server与寻求医疗建议的Client之间的新对话。
		"""

		self.task_prompt = self.task_prompt.replace("\t", "").strip()

		self.inf_args = {
			"max_new_tokens": 128,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return

	def process_exp(self, keep_sys_da=True, keep_user_da=False):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da) + "\n"
		return prompt_exps.strip()

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == KstGame.S_WARMING_UP or da == KstGame.S_CONTINUE_DIALOG else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			Server:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{state.to_string_rep(keep_sys_da=True)}
			Server:
			"""
		prompt = prompt.replace("\t", "").strip()
		logger.debug(prompt)
		data = self.generation_model.generate(prompt, **self.inf_args)
		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == KstGame.USR)
		prompt = f"""
		以下是医疗咨询场景的背景信息。
		在这个场景中，Server（医疗专业人员）正在与Client（患者）进行对话，以提供医疗建议、诊断健康问题或讨论治疗方案。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		Client可以在对话中选择以下行为来回应Server：
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		以下是Server与Client在医疗咨询背景下的对话，其中Server正在处理Client的健康问题并建议可能的治疗方案。
		{self.process_exp(keep_sys_da=False, keep_user_da=True)}
		以下是另一位Server与寻求医疗建议的Client之间的新对话。
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
		Server: 您有微信么？我加一下，根据您的症状，我再帮你详细分析下。
		Client:
		"""
		prompt = prompt.replace("\t", "").strip()

		inf_args = {
			"max_new_tokens": 128,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.generate(prompt, **inf_args)
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"Client prompt: {prompt}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == KstGame.U_PROVIDE_CONTACT or da == KstGame.U_CONFIRM_APPOINTMENT_TIME:
				score.append(1.0)
			elif da == KstGame.U_REPLY_DISEASE_INFO or da == KstGame.U_REPLY_MEDICAL_HISTORY:
				score.append(0.5)
			elif da == KstGame.U_INQUIRY_QUESTION or da == KstGame.U_REPLY_BASIC_INFO or da == KstGame.U_INQUIRE_HOSPITAL_INFO:
				score.append(0)
			elif da == KstGame.U_WARM_UP or da == KstGame.U_CONTINUE_CONVERSATION:
				score.append(-0.5)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)
	

class KstChatSystemPlanner(KstSystemPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__(
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns,
			generation_model, conv_examples
		)
		self.task_prompt = f"""
		在Kst医疗咨询场景中，医疗专业人员（Server）与患者（Client）之间进行对话，目的是提供医疗建议、讨论健康问题和治疗方案。
		医疗专业人员需要有效沟通，以帮助患者更好地理解他们的健康状况和可用的治疗选项。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		作为医疗专业人员，你可以在对话中选择以下行为：
		{" ".join([f"[{da}]" for da in dialog_acts])}
		以下是医疗专业人员和患者之间关于医疗咨询的示例对话。
		""".replace("\t", "").strip()
		self.new_task_prompt = "以下是一位新的医疗专业人员（你）和一位患者之间的新对话。"
		self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

		self.inf_args = {
			"max_new_tokens": 128,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return
	
	def process_chat_exp(self, 
			new_task_prompt,
			assistant_role=KstGame.SYS,
			keep_sys_da=True, keep_user_da=False):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp, keep_sys_da, keep_user_da, assistant_role)
			prompt_exps.append({
				"role":"system", "content": new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self,
			exp:DialogSession, 
			keep_sys_da, keep_user_da,
			assistant_role=KstGame.SYS,
			max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# Kst dataset starts with the system/Server
		assert(exp[0][0] == KstGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		# init with user
		# if assistant_role == KstGame.SYS:
		# 	if keep_user_da:
		# 		prompt_messages.append({
		# 			"role": "user",
		# 			"content": f"{KstGame.USR}: [{KstGame.U_CONTINUE_CONVERSATION}] Hello.".strip()
		# 		})
		# 	else:
		# 		prompt_messages.append({
		# 			"role": "user",
		# 			"content": f"{KstGame.USR}: Hello.".strip()
		# 		})
		# all the rest
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			# if assistant is the Server, then current data is also Server -> then it is of role "system"
			if role == KstGame.SYS:
				if keep_sys_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == KstGame.SYS:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			else:
				if keep_user_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == KstGame.USR:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			
			prompt_messages.append({
				"role": prompt_role,
				"content": content
			})
		return prompt_messages

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == KstGame.S_WARMING_UP else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{KstGame.USR}: Hello.'})
		else:
			assert(state[-1][0] == KstGame.USR)
			messages += self.__proccess_chat_exp(state, keep_sys_da=True, keep_user_da=False)
		# produce a response
		data = self.generation_model.chat_generate(messages, **self.inf_args)

		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == KstGame.USR)

		user_task_prompt = f"""
		您是一位寻求医疗帮助的患者（Client）。一位医疗专业人员（Server）正在尝试为您提供医疗咨询、诊断疾病或讨论治疗方案。
		您可以在对话中选择以下行为来回应医疗专业人员：
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		以下是一位医疗专业人员与您（患者）之间的新对话。
		""".replace("\t", "").strip()
		user_new_task_prompt = "以下是一位医疗专业人员与您（患者）之间的新对话。"

		messages = [
			{'role': 'system', 'content': user_task_prompt},
			*self.process_chat_exp(new_task_prompt=user_new_task_prompt, assistant_role=KstGame.USR, keep_sys_da=False, keep_user_da=True),
			{'role': 'system', 'content': user_new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, assistant_role=KstGame.USR, keep_sys_da=False, keep_user_da=True)
		x={"generated_text":messages[-1]["content"]}
		data=[x]
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"Client prompt: {messages}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == KstGame.U_PROVIDE_CONTACT or da == KstGame.U_CONFIRM_APPOINTMENT_TIME:
				score.append(1.0)
			elif da == KstGame.U_REPLY_DISEASE_INFO or da == KstGame.U_REPLY_MEDICAL_HISTORY:
				score.append(0.5)
			elif da == KstGame.U_INQUIRY_QUESTION or da == KstGame.U_REPLY_BASIC_INFO or da == KstGame.U_INQUIRE_HOSPITAL_INFO:
				score.append(0)
			elif da == KstGame.U_WARM_UP or da == KstGame.U_CONTINUE_CONVERSATION:
				score.append(-0.5)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)


class ServerModel(DialogModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.max_hist_num_turns = max_hist_num_turns
		# prompts and DAs
		self.da_prompts_mapping = {
			KstGame.S_CONTINUE_DIALOG:"指维持对话的持续性，鼓励对方继续交谈，可能通过提问、回应或引入新话题实现。",
			KstGame.S_CONFIRM_INQUIRY:"明确对方提出的咨询内容，确保理解准确，可能需要通过重述或澄清问题来实现。",
			KstGame.S_WARMING_UP:"通过友好、非正式的交谈降低对话氛围的紧张感，建立舒适的对话环境。",
			KstGame.S_REFUSE:"在面对患者拒绝就诊或治疗的情况下，采取的应对策略，可能包括询问原因、提供额外信息或安抚。",
			KstGame.S_INQUIRY_DEMAND:"明确对方在咨询或治疗中的具体需求，以便提供更准确的帮助或信息。",
			KstGame.S_INQUIRY_BASIC_INFO:"收集对方的基本个人信息，如年龄、性别、职业、地址等，这些信息有助于了解背景和提供个性化的服务。",
			KstGame.S_INQUIRY_DISEASE_INFO:"获取关于患者当前疾病状况的详细信息，如症状、发病时间等。",
			KstGame.S_INQUIRY_MEDICAL_HISTORY:"了解患者以往的就医经历，包括过去的诊断、治疗和就诊记录。",
			KstGame.S_DIAGNOSTIC_ANALYSIS:"基于收集的信息对疾病进行分析，提出可能的诊断。",
			KstGame.S_DESCRIBE_DISEASE_FEATURES:"向患者说明其疾病的特点，包括病因、症状、发展过程等。",
			KstGame.S_CONFIRM_TREATMENT_INTENTION:"确认患者是否愿意接受建议的治疗方案或进一步的医疗操作。",
			KstGame.S_DESCRIBE_TREATMENT_PLAN:"向患者详细介绍治疗方法、计划、预期效果以及可能的风险。",
			KstGame.S_DESCRIBE_HOSPITAL_INFO:"提供有关医院或诊所的相关信息，如位置、服务、专业领域等。例如SERVER的话语中出现<address>等",
			KstGame.S_INTRODUCE_PRICE:"告知患者有关治疗或服务的费用信息。",
			KstGame.S_ANSWER_QUESTION:"回答患者的疑问，提供必要的信息和解释。",
			KstGame.S_TRICK_CALL:"在电话交谈中，通过提问或讨论，试图获取更多的信息，在该场景下指试图获取CLIENT的联系方式",
			KstGame.S_INVITATION:"邀请患者到医院或诊所进行面诊或进一步的咨询。",
			KstGame.S_LEAVE_CONTACT:"提供或请求对方的联系方式，以便后续沟通或跟进。例如SERVER的话语中出现<mobile>等",
			KstGame.S_CONFIRM_CONTACT:"确认双方用于后续沟通的最佳联系方式和时间。例如复述CLIENT的联系方式"
					}
		# only allow da that has the mapping
		self.dialog_acts = [da for da in dialog_acts if da in self.da_prompts_mapping]
		
		logger.debug(self.dialog_acts)
		self.task_prompt = f"""
		以下是医疗咨询场景的背景信息。
		在这个场景中，医疗专业人员（Server）与患者（Client）之间进行交流，目的是为了提供医疗建议、诊断健康问题或讨论治疗方案。
		医疗专业人员需要与患者进行有效的沟通，以帮助他们更好地理解自己的健康状况和可能的治疗选择。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		以下是医疗专业人员与患者之间的一段示例对话，其中医疗专业人员正在处理患者的健康问题并提出治疗建议。
		{self.process_exp()}
		以下是另一位医疗专业人员与患者之间的新对话。
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level
			"return_full_text": False,
			**inference_args
		}
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_exp(exp) + "\n"
		return prompt_exps.strip()

	def __proccess_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		prompt_exp = ""
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			
			if role == KstGame.SYS:
				prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
			else:
				prompt_exp += f"{role}: {utt}\n"
		return prompt_exp.strip()
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		# planner gives an action, state is history, you need to produce a response accrd to the action
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			{da_prompt}
			Server:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{self.__proccess_exp(state, max_hist_num_turns=self.max_hist_num_turns)}
			{da_prompt}
			Server:
			"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		sys_resp = self.backbone_model._cleaned_resp(data, prompt)[0]  # TODO
		return sys_resp

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError
	

class ServerChatModel(ServerModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__(
			dialog_acts=dialog_acts,
			backbone_model=backbone_model,
			max_hist_num_turns=max_hist_num_turns,
			conv_examples=conv_examples,
			inference_args=inference_args
		)
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level, unless you do OpenLoop search
			"return_full_text": False,
			**inference_args
		}
		self.task_prompt = """
		在Kst医疗咨询场景中，医疗专业人员（Server）与患者（Client）之间进行对话，目的是为了提供医疗建议、诊断健康问题或讨论治疗方案。
		医疗专业人员需要与患者进行有效沟通，以帮助他们更好地理解自己的健康状况和可能的治疗选择。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		以下是医疗专业人员与患者之间的一段示例对话。
		""".replace("\t", "").strip()
		self.new_task_prompt = "以下是一位医疗专业人员（您）与另一位患者之间的新对话。\n医疗专业人员向患者问好。"
		self.prompt_examples = self.process_chat_exp()
		return

	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# Kst dataset starts with the system
		assert(exp[0][0] == KstGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		
		next_sys_da = KstGame.S_WARMING_UP
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == KstGame.SYS:
				prompt_messages.append({
					"role": "assistant",
					"content": f"{role}: {utt}".strip()
				})
			else:
				if i+1 < len(exp.history):
					next_sys_da = exp[i+1][1]
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}\n{self.da_prompts_mapping[next_sys_da]}".strip()
					})
				else:
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}".strip()
					})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		return self.get_utterance_batched(state, action, batch=1)[0]
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3) -> List[str]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{KstGame.USR}: Hello.\n{da_prompt}'})
		else:
			assert(state[-1][0] == KstGame.USR)
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
		gen_args = {
			**self.inference_args,
			"num_return_sequences": batch,  # this will be changed to n inside chat_generate
		}
		data = self.backbone_model.chat_generate(messages, **gen_args)
		sys_resps = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{KstGame.SYS}:", user_role=f"{KstGame.USR}:"
		)
		return sys_resps

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError


class ClientModel(DialogModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		# prompts
		self.task_prompt = f"""
		以下是关于医疗咨询场景的背景信息。
		在这个场景中，医疗专业人员（Server）正试图向寻求帮助的患者（Client）提供医疗建议、诊断疾病或讨论治疗方案。
		除此之外，医疗专业人员（Server）希望能够获取患者（Client）的联系方式，以进行下一步诊断。
		患者在对话中可以选择以下行为来回应医疗专业人员：
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		以下是一位医疗专业人员与一位患者关于医疗咨询的示例对话。
		{self.process_exp()}
		以下是另一位医疗专业人员与患者之间的新对话。
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = inference_args
		return
	
	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
		return prompt_exps.strip()
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == KstGame.SYS)
		prompt = f"""
		{self.task_prompt}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
		Client:
		"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
		return user_resp

	def get_utterance_w_da(self, state:DialogSession, action=None) -> "Tuple[str, str]":
		user_resp = self.get_utterance(state, action)
		# extract da
		start_idx = user_resp.find("[")
		end_idx = user_resp.find("]")
		if start_idx == -1 or end_idx == -1:
			da = KstGame.U_CONTINUE_CONVERSATION
		else:
			da = user_resp[start_idx+1:end_idx]
			user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
			if da not in self.dialog_acts:
				da = KstGame.U_CONTINUE_CONVERSATION
		return da, user_resp


class ClientChatModel(ClientModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__(
			dialog_acts=dialog_acts,
			inference_args=inference_args,
			backbone_model=backbone_model,
			conv_examples=conv_examples,
			max_hist_num_turns=max_hist_num_turns
		)
		self.inference_args = inference_args
		self.task_prompt = f"""
		您是一位寻求医疗帮助的患者（Client）。一位医疗专业人员（Server）正试图为您提供医疗建议、诊断健康问题或讨论治疗方案。
		在对话过程中，您可以选择以下行为来回应医疗专业人员：
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		以下是一位医疗专业人员与某位患者之间的示例对话。
		""".replace("\t", "").strip()
		self.new_task_prompt = "以下是一位医疗专业人员与您（患者）之间的新对话。您可以选择是否接受所提供的医疗建议或者是否留下联系方式以方便下一步的治疗诊断。"
		self.heuristic_args: dict = {
			"max_hist_num_turns": 2,
			"example_pred_turn": [[0, 2, 3, 4]]
		}
		self.prompt_examples = self.process_chat_exp()
		return
	
	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# Kst dataset starts with the system
		assert(exp[0][0] == KstGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == KstGame.SYS:
				prompt_messages.append({
					"role": "user",
					"content": f"{role}: {utt}".strip()
				})
			else:
				prompt_messages.append({
					"role": "assistant",  # assistant is the user simulator
					"content": f"{role}: [{da}] {utt}".strip()
				})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == KstGame.SYS)  # next turn is user's turn
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)

		# produce a response
		data = self.backbone_model.chat_generate(messages, **self.inference_args)
		user_resp = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{KstGame.USR}:", user_role=f"{KstGame.SYS}:"
		)[0]
		return user_resp
	
	def get_utterance_from_batched_states(self, states:List[DialogSession], action=None) -> List[str]:
		assert(all([state[-1][0] == KstGame.SYS for state in states]))
		all_prompts = []
		for state in states:
			messages = [
				{'role': 'system', 'content': self.task_prompt},
				*self.prompt_examples,
				{'role': 'system', 'content': self.new_task_prompt}
			]
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
			all_prompts.append(messages)
		# produce a response
		datas = self.backbone_model.chat_generate_batched(all_prompts, **self.inference_args)
		user_resps = []
		for data in datas:
			user_resp = self.backbone_model._cleaned_chat_resp(
				data, assistant_role=f"{KstGame.USR}:", user_role=f"{KstGame.SYS}:"
			)
			user_resps.append(user_resp[0])
		return user_resps
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		gen_user_resps = self.get_utterance_from_batched_states(states, action)
		das = []
		user_resps = []
		# extract da
		for user_resp in gen_user_resps:
			start_idx = user_resp.find("[")
			end_idx = user_resp.find("]")
			if start_idx == -1 or end_idx == -1:
				da = KstGame.U_CONTINUE_CONVERSATION
			else:
				da = user_resp[start_idx+1:end_idx]
				user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
				if da not in self.dialog_acts:
					da = KstGame.U_CONTINUE_CONVERSATION
			das.append(da)
			user_resps.append(user_resp)
		return das, user_resps

	def __process_heuristics_chat_exp(self, dialog:DialogSession):
		if len(dialog) == 0:
			return []
		# assumes you start with the system
		# and ends with a user utterance to predict
		assert(dialog[0][0] == KstGame.SYS)
		assert(dialog[-1][0] == KstGame.USR)

		prompt_messages = []
		input_context = []
		answer_da = dialog[-1][1]
		for i, (role, da, utt) in enumerate(dialog):
			# if assistant is the Server, then current data is also Server -> then it is of role "system"
			# treat this as a task
			content = f"{role}: {utt}".strip()
			input_context.append(content)
		input_context.append(f"{dialog.USR} feeling:")

		prompt_q = "\n".join(input_context)
		prompt_messages.append({
			"role": 'user',
			"content": prompt_q
		})
		prompt_messages.append({
			"role": 'assistant',
			"content": f"{answer_da}"
		})
		return prompt_messages
	
	def __truncate_heuristics_dialog(self, dialog:DialogSession, pred_end_idx=-1):
		max_history_length = self.heuristic_args['max_hist_num_turns']
		if pred_end_idx == -1:
			pred_end_idx = len(dialog.history) - 1
		new_sys_start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
		new_history = []
		for j, (role, da, utt) in enumerate(dialog):
			if j >= new_sys_start_idx:
				new_history.append((role, da, utt))
			if j == pred_end_idx:
				# user's utternace to predict
				break
		new_dialog_session = DialogSession(dialog.SYS, dialog.USR).from_history(new_history)
		return new_dialog_session
	
	def process_heurstics_chat_exp(self, new_task_prompt: str):
		prompt_exps = []
		for i, exp in enumerate(self.conv_examples):
			pred_end_turns: List[int] = self.heuristic_args['example_pred_turn'][i]
			# make a new dialogue session until that pred_idx with max max_history_length turns
			for pred_end_turn in pred_end_turns:
				pred_end_idx = pred_end_turn * 2 + 1
				new_dialog_session = self.__truncate_heuristics_dialog(exp, pred_end_idx)
				prompt_exps += self.__process_heuristics_chat_exp(new_dialog_session)
				prompt_exps.append({
					"role":"system", "content": new_task_prompt
				})
		return prompt_exps[:-1]

	def predict_da(self, state:DialogSession, never_end=True) -> str:
		# never_end=True  during real chat, let user choose to terminate, not this function
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == KstGame.USR)

		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
			{'role': 'system', 'content': self.new_task_prompt}
		]
		new_dialog_session = self.__truncate_heuristics_dialog(state, -1)
		messages += self.__process_heuristics_chat_exp(new_dialog_session)[:-1]

		# majority vote, same as value function
		inf_args = {
			"max_new_tokens": 128,
			"temperature": 0.7,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 5,
		}
		datas = self.backbone_model.chat_generate(messages, **inf_args)
		# process into das
		sampled_das: list = []
		for resp in datas:
			user_da = resp['generated_text'].strip()
			if user_da not in self.dialog_acts:
				sampled_das.append(KstGame.U_CONTINUE_CONVERSATION)
			if never_end:
				sampled_das.append(user_da)
			else:
				sampled_das.append(user_da)
		logger.info(f"sampled das: {sampled_das}")
		# majority vote
		counted_das = Counter(sampled_das)
		user_da = counted_das.most_common(1)[0][0]
		return user_da