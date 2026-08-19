DP（数据并行）：每个GPU都拥有完整的模型副本，然后，将训练数据划分成多个小批次（mini-batch），每个批次分配给不同的GPU进行处理。
![[Pasted image 20260819145231.png|543]]
zero：通过对模型副本中的优化器状态、梯度和参数进行切分，来实现减少对内存的占用
DDP（分布式数据并行）：传统DP一般用于**单机多卡**场景。而DDP能多机也能单机。这依赖于**Ring-AllReduce**，它由百度最先提出，可以有效解决数据并行中通信负载不均（Server存在瓶颈）的问题。
pp（流水线并行）：流水线并行，是将**模型的不同层**（单层，或连续的多层）分配到不同的GPU上，按顺序处理数据，实现流水线式的并行计算。
![[Pasted image 20260819145833.png]]
流水并行有点像串行。每个GPU需要等待前一个GPU的计算结果，可能会导致大量的GPU资源浪费（图中黄色部分bubble）。图b将mini-batch的数据进一步切分成micro-batch数据。当GPU 0处理完一个micro-batch数据后，紧接着开始处理下一个micro-batch数据，以此来减少GPU的空闲时间。图c在一个micro-batch完成前向计算后，提前调度，完成相应的反向计算，这样就能释放部分显存，用以接纳新的数据，提升整体训练性能。![[Pasted image 20260819145822.png]]TP（张量并行）：将模型的张量（如权重矩阵）按维度切分到不同的GPU上运行的并行方式。张量切分方式分为按行进行切分和按列进行切分，分别对应行并行（Row Parallelism）（权重矩阵按行分割）与列并行（Column Parallelism）（权重矩阵按列分割）。![[Pasted image 20260819150229.png]]每个节点处理切分后的子张量。最后，通过集合通信操作（如All-Gather或All-Reduce）来合并结果。适合单个张量过大的情况，可以**显著减少单个节点的内存占用**。但是当切分维度较多的时候，**通信开销比较大**。而且，张量并行的实现过程较为复杂，需要仔细设计切分方式和通信策略。
EP（专家并行）：MoE（混合专家模型）中的一种并行计算策略。它通过将专家（子模型）分配到不同的GPU上，实现计算负载的分布式处理，提高计算效率。最终结果按照all-to-all的方式通信。
通信原语：
一对多：
broadcast（广播）
![[Pasted image 20260819150921.png|599]]
Scatter（划分并散布）
![[Pasted image 20260819150939.png|593]]
多对一：
Reduce（规约）：在集合通信里，它表示“规约”运算，是**一系列简单运算操作**（包括：SUM、MIN、MAX、PROD、LOR等）的统称。![[Pasted image 20260819151058.png|595]]
Gather（反向的Scatter）：![[Pasted image 20260819151015.png|584]]
多对多：
All Reduce：Dp中常用
![[Pasted image 20260819151243.png|581]]
All Gather：可以理解为先Gather再Broadcast![[Pasted image 20260819151215.png|597]]
Reduce Scatter：先归约（Reduce），再分散（Scatter）
![[Pasted image 20260819151346.png]]
All-to-All：将节点i的发送缓冲区中的第j块数据发送给节点j。节点j将接收到的来自节点i的数据块，放在自身接收缓冲区的第i块位置。
![[Pasted image 20260819151512.png]]Ring All reduce：
	![[Pasted image 20260819152041.png|386]]![[Pasted image 20260819152055.png|279]]