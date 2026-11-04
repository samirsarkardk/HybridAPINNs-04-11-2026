# HybridXPINNs-27-10-2026

# This is done with HybridXPINNs algorithm
The governing equation is

$$
\frac{\partial u}{\partial t}
+c\frac{\partial u}{\partial x}=0.
$$ 


where 

$$ c = 0.5 $$

with the initial conditions

$$
u(x,0)=
\begin{cases}
1, & \text{when} -0.2 \le x \le 0.2\\
0, & \text{otherwise}.
\end{cases}
$$


Here in the strip I want to apply fully different PINN network.
