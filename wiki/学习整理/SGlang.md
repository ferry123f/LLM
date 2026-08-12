分层理解： 服务层 调度层 模型层
![sglang-architecture.svg|700](https://raw.githubusercontent.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/0a0ba58aae3eccded83c77967f0b1185a018acd7/sglang/code-walk-through/sglang-architecture.svg)
主进程：HTTP服务+tokenizer
子进程：scheduler
子进程：detokenizer
scheduler.py:
由run_scheduler_process L4574r启动 调用 run_event_loop， 有两种模式：
events_loop_normal和events_loop_overlap，下面以前者为例
由一个主循环组成：
while true：
	rece_requests:从两个 ZMQ 通道（用户 + RPC）一次性把当前所有能读到的消息拉出来（_pull_raw_reqs()非阻塞），做多 rank 广播（_broadcast_reqs_across_ranks）和解包，返回一个 list。
	processs_input_requests:
	get_next_batch_to_run:
	result = self.run_batch(batch):
	self.process_batch_result(batch, result):大体可以分为process_batch_result_decode和process_batch_result_prefill，前者是把 decode 出来的新 token 归位到每个用户，结束的用户释放 KV；后者是记第 1 个 token； 核心是把 KV 写进 RadixCache ，两者都会调用output_streamer.stream_output把处理后的结果发送给另一个进程Detokenizer