import logging
import random

from abc import ABC, abstractmethod
from typing import List
from core.gen_models_kst import GenerationModel


logger = logging.getLogger(__name__)


class RespRanker(ABC):
	@abstractmethod
	def evaluate(self, context, resp_a, resp_b):
		"""
		Compare two responses and return the preference.
		"""
		raise NotImplementedError


class KstEvaluator(RespRanker):
	def __init__(self, gen_model: GenerationModel):
		super().__init__()
		self.gen_model = gen_model
		self.inference_args = {
			"max_tokens": 128,
			"temperature": 0.7,
			"echo": False,
			"n": 5,
			"stop": ""
		}
	
	def evaluate(self, context, resp_a, resp_b):
		do_swap = False
		if random.random() < 0.5:
			do_swap = True
			resp_a, resp_b = resp_b, resp_a
		prompt = f"""
		以下是关于Kst医疗咨询场景的背景信息。
		在这个场景中，医疗专业人员（Server）与患者（Client）进行对话，目的是为了提供医疗建议、诊断健康问题或讨论治疗方案。
		医疗专业人员需要与患者进行有效的沟通，以帮助他们更好地理解自己的健康状况和可能的治疗选择。
		在咨询过程中，医疗专业人员希望能够留下患者联系方式，以更好地服务患者。
		以下是医疗专业人员与患者之间的一段对话。医疗专业人员正在试图帮助患者理解其健康状况并提出建议。
		{context}
		以下哪个回应更能帮助医疗专业人员有效地向患者提供医疗建议并留下患者联系方式？
		A. Server：{resp_a}
		B. Server：{resp_b}
		C. 无法判断。
		您可以选择A、B或C。
		您的选择：
		""".replace('\t', '').strip()
		logger.debug(f"prompt: {prompt}")
		resps = self.gen_model.generate(prompt, **self.inference_args)
		choices, rationales = self._process_resps(resps)
		preference = self._majority_vote(choices, do_swap)
		return preference, {'choices': choices, 'rationales': rationales, 'do_swap': do_swap}

	def _process_resps(self, resps:List[dict]):
		choices = []
		rationales = []
		for resp in resps:
			gen = resp['generated_text'].strip()
			
			if len(gen) == 0:
				print("Empty response")
				choice = 'c'
			else:
				choice = gen[0].lower()
			
			if choice not in ['a', 'b', 'c']:
				print(f"Invalid choice: {choice}")
				choice = 'c'
			choices.append(choice)
			# see if there is a rationale  # just dump the entire response
			rationale = gen
			rationales.append(rationale)
		return choices, rationales

	def _majority_vote(self, resps:List[str], do_swap=False):
		# if there is a majority vote between A=0 and B=1, return the majority vote
		# otherwise, return C=2
		a_cnt = 0
		b_cnt = 0
		for resp in resps:
			if resp == 'a':
				a_cnt += 1
			elif resp == 'b':
				b_cnt += 1
		if a_cnt > b_cnt:
			return 0 if not do_swap else 1
		elif b_cnt > a_cnt:
			return 1 if not do_swap else 0
		return 2