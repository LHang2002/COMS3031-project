from core.game_kst import KstGame

EXP_DIALOG = [
    # 
	(KstGame.SYS, KstGame.U_WARM_UP,				"你好，有什么需要帮助的吗",),
	(KstGame.USR, KstGame.U_INQUIRY_QUESTION,		"您好，我想咨询一下，我的头发突然之间掉了一大块，这是怎么回事呢？",),
	(KstGame.SYS, KstGame.S_INQUIRY_BASIC_INFO,		"能告诉我您的性别和年龄吗？",),
	(KstGame.USR, KstGame.U_REPLY_BASIC_INFO,		"我是男性，45岁。",),
	(KstGame.SYS, KstGame.S_INQUIRY_DISEASE_INFO,	"这个情况持续了多久？目前有几块脱发区域？",),
	(KstGame.USR, KstGame.U_REPLY_DISEASE_INFO,		"大概两到三个月了，就一块。",),
	(KstGame.SYS, KstGame.S_DIAGNOSTIC_ANALYSIS,	"嗯，根据您的描述，初步判断可能是斑秃。",),
	(KstGame.USR, KstGame.U_INQUIRY_QUESTION,		"请问这与压力有关吗？还有，它还会再长出来吗？",),
	(KstGame.SYS, KstGame.S_ANSWER_QUESTION,		"斑秃的成因复杂，可能与多种因素有关。您能告诉我您的微信号吗？我们加个微信，我可以更详细地帮您分析。",),
	(KstGame.USR, KstGame.U_PROVIDE_CONTACT,		"好的，<mobile>",),
]