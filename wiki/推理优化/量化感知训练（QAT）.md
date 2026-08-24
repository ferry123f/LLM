量化：
以FP32到INT8举例，量化核心思想就是将浮点数区间的参数映射到INT8的离散空间中
r=S（q-z）
其中，r为FP32的浮点数，q是INT8的量化值，z是zero-point（零点），S是缩放因子
![[Pasted image 20260824151034.png]]
要确定S和z两个参数
s=rmax-rmin/qmax-qmin
z=0-rmin/s +qmin
训练后量化(PTQ)