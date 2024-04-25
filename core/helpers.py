class DialogSession():
	# 初始化DialogSession类
	def __init__(self, sys_name, user_name) -> None:
		self.SYS = sys_name
		self.USR = user_name
		self.history: list = []  # [(role, da, utt), ....]
		return
	
	# 对话历史赋值
	def from_history(self, history):
		self.history = history
		return self
	
	# 分割对话历史
	def to_string_rep(self, keep_sys_da=False, keep_user_da=False, max_turn_to_display=-1):
		history = ""
		# 分割的轮数
		num_turns_to_truncate = 0
		# 最大展示轮数
		if max_turn_to_display > 0:
			num_turns_to_truncate = max(0, len(self.history) // 2 - max_turn_to_display)
		for i, (role, da, utt) in enumerate(self.history):
			if (i // 2) < num_turns_to_truncate:
				continue
			if i % 2 == 0:
				assert(role == self.SYS)
				if keep_sys_da:
					history += f"{role}: [{da}] {utt}\n"
				else:
					history += f"{role}: {utt}\n"
			else:
				assert(role == self.USR)
				if keep_user_da:
					history += f"{role}: [{da}] {utt}\n"
				else:
					history += f"{role}: {utt}\n"
		return history.strip()

	# 复制DialogSession
	def copy(self):
		new_session = DialogSession(self.SYS, self.USR)
		new_session.from_history(self.history.copy())
		return new_session

    # 增加对话历史
	def add_single(self, role, da, utt):
		if len(self.history) % 2 == 0:
			assert(role == self.SYS)
		else:
			assert(role == self.USR)
		self.history.append((role, da, utt))
		return
	
	# 获取对话内容
	def get_turn_utt(self, turn, role):
		if role == self.SYS:
			return self.history[turn * 2][-1]
		else:
			return self.history[turn * 2 + 1][-1]
	
	# 迭代
	def __iter__(self):
		return iter(self.history)

	# 返回对话历史轮数
	def __len__(self):
		return len(self.history) //  2  # number of turns

	# 返回指定对话历史
	def __getitem__(self, index):
		return self.history[index]

	# 判断两个对话历史是否相同
	def __eq__(self, __o: object) -> bool:
		# 判断是否为同一类型
		if not isinstance(__o, DialogSession):
			return False
		return self.history == __o.history