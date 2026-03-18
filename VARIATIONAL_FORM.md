# `hdg_conv_diff_fenicsx_0_9_fix.py` 的变分形式

本文档直接对应脚本 `hdg_conv_diff_fenicsx_0_9_fix.py` 中的有限元空间、双线性形式和线性形式。

## 1. 强形式

在区域 $\Omega = (0,1)^2$ 上，程序求解的是稳态对流-扩散方程

$$
\boldsymbol{w}\cdot\nabla u - \nabla\cdot(\kappa \nabla u) = f \qquad \text{in } \Omega,
$$

并在边界上施加 Dirichlet 条件

$$
u = u_D \qquad \text{on } \partial\Omega.
$$

其中：

- 扩散系数是常数 $\kappa = 10^{-3}$；
- 对流速度场为
  $$
  \boldsymbol{w}(x,y) =
  \begin{pmatrix}
  \sin(\pi x)\sin(\pi y) \\
  \cos(\pi x)\cos(\pi y)
  \end{pmatrix};
  $$
- 精确解取为
  $$
  u_e(x,y) = \sin(3\pi x)\cos(2\pi y);
  $$
- 右端由精确解反代得到
  $$
  f = \boldsymbol{w}\cdot\nabla u_e - \nabla\cdot(\kappa \nabla u_e).
  $$

## 2. 离散空间

脚本采用 HDG 的两场写法：

- 单元内部未知量 $u_h \in V_h$；
- 骨架/迹未知量 $\bar u_h \in \bar V_h$。

对应离散空间为

$$
V_h = \{ v_h\in L^2(\Omega): v_h|_K \in \mathbb{P}_k(K),\ \forall K\in\mathcal{T}_h \},
$$

$$
\bar V_h = \{ \bar v_h\in L^2(\mathcal{F}_h): \bar v_h|_F \in \mathbb{P}_k(F),\ \forall F\in\mathcal{F}_h \},
$$

其中 $\mathcal{T}_h$ 是网格剖分，$\mathcal{F}_h$ 是全部 facet 的集合。测试函数记为

$$
(v_h, \bar v_h) \in V_h \times \bar V_h.
$$

## 3. 数值通量与参数

程序中定义：

- 单元尺寸 $h_K$ 由 `CellDiameter(msh)` 给出；
- 外法向量为 $\boldsymbol{n}$；
- 惩罚参数
  $$
  \gamma = \frac{16k^2}{h_K};
  $$
- 对流上风参数
  $$
  \lambda =
  \begin{cases}
  0, & \boldsymbol{w}\cdot\boldsymbol{n} > 0, \\
  1, & \boldsymbol{w}\cdot\boldsymbol{n} \le 0.
  \end{cases}
  $$

因此程序使用的上风迹值可以写成

$$
\widehat{u}_h = u_h - \lambda (u_h - \bar u_h).
$$

也就是：

- 出流面 ($\boldsymbol{w}\cdot\boldsymbol{n}>0$) 取 $\widehat{u}_h = u_h$；
- 入流面 ($\boldsymbol{w}\cdot\boldsymbol{n}\le 0$) 取 $\widehat{u}_h = \bar u_h$。

## 4. 双线性形式

程序中的双线性形式由扩散项和对流项两部分构成。

### 4.1 扩散部分

对所有 $(u_h,\bar u_h),(v_h,\bar v_h) \in V_h\times\bar V_h$，扩散双线性形式为

$$
\begin{aligned}
a_{\mathrm{diff}}\big((u_h,\bar u_h),(v_h,\bar v_h)\big)
={}& \sum_{K\in\mathcal{T}_h} \int_K \kappa\nabla u_h\cdot\nabla v_h\,dx \\
&- \sum_{K\in\mathcal{T}_h} \int_{\partial K} \kappa (u_h-\bar u_h)\, \nabla v_h\cdot\boldsymbol{n}\,ds \\
&- \sum_{K\in\mathcal{T}_h} \int_{\partial K} (\kappa\nabla u_h\cdot\boldsymbol{n})(v_h-\bar v_h)\,ds \\
&+ \sum_{K\in\mathcal{T}_h} \int_{\partial K} \gamma\kappa (u_h-\bar u_h)(v_h-\bar v_h)\,ds.
\end{aligned}
$$

### 4.2 对流部分

程序中的对流部分是

$$
\begin{aligned}
a_{\mathrm{adv}}\big((u_h,\bar u_h),(v_h,\bar v_h)\big)
={}& -\sum_{K\in\mathcal{T}_h} \int_K (\boldsymbol{w}u_h)\cdot\nabla v_h\,dx \\
&+ \sum_{K\in\mathcal{T}_h} \int_{\partial K}
(\boldsymbol{w}\cdot\boldsymbol{n})\,\widehat{u}_h\,(v_h-\bar v_h)\,ds,
\end{aligned}
$$

其中

$$
\widehat{u}_h = u_h - \lambda (u_h-\bar u_h).
$$

### 4.3 总双线性形式

所以脚本中的总双线性形式就是

$$
a\big((u_h,\bar u_h),(v_h,\bar v_h)\big)
= a_{\mathrm{diff}}\big((u_h,\bar u_h),(v_h,\bar v_h)\big)
+ a_{\mathrm{adv}}\big((u_h,\bar u_h),(v_h,\bar v_h)\big).
$$

## 5. 线性形式

脚本右端项写成

$$
L(v_h,\bar v_h)
= \sum_{K\in\mathcal{T}_h} \int_K f v_h\,dx
+ \int_{\mathcal{F}_h} 0\cdot \bar v_h\,ds.
$$

也就是说，真正起作用的是体积分项

$$
L(v_h,\bar v_h)=\sum_{K\in\mathcal{T}_h} \int_K f v_h\,dx.
$$

第二项只是为了和 block mixed-domain 形式保持一致而保留的零项。

## 6. 离散变分问题

因此，该程序求解的离散问题可以表述为：

> 求 $(u_h,\bar u_h)\in V_h\times\bar V_h$，并满足边界上的迹变量
> $$
> \bar u_h = u_D \quad \text{on } \partial\Omega,
> $$
> 使得对任意 $(v_h,\bar v_h)\in V_h\times\bar V_h$ 都有
> $$
> a\big((u_h,\bar u_h),(v_h,\bar v_h)\big)=L(v_h,\bar v_h).
> $$

## 7. 与代码的对应关系

代码中的关键对应关系如下：

- `V`, `Vbar`, `W`：分别对应 $V_h$、$\bar V_h$ 和乘积空间；
- `u, ubar`：对应试函数 $(u_h,\bar u_h)$；
- `v, vbar`：对应测试函数 $(v_h,\bar v_h)$；
- `gamma = 16.0 * k**2 / h`：对应罚参数 $\gamma$；
- `lmbda = conditional(gt(dot(w, n), 0), 0, 1)`：对应上风参数 $\lambda$；
- `a`：对应总双线性形式；
- `L`：对应右端线性形式；
- `bc = fem.dirichletbc(u_bc, dofs)`：对应在边界上对 $\bar u_h$ 施加 Dirichlet 条件。

---

# `hdg_stokes_fenicsx_0_9.py` 的强形式与 HDG 变分形式

本文档这一部分对应脚本 `hdg_stokes_fenicsx_0_9.py` 当前实现的二维 Stokes 问题。

## 8. Stokes 强形式

在区域 $\Omega=(0,1)^2$ 上，程序求解稳态 Stokes 方程

$$
-\nu \Delta \boldsymbol{u} + \nabla p = \boldsymbol{f}
\qquad \text{in } \Omega,
$$

$$
\nabla\cdot \boldsymbol{u} = 0
\qquad \text{in } \Omega,
$$

并在边界上施加速度 Dirichlet 条件

$$
\boldsymbol{u} = \boldsymbol{u}_D
\qquad \text{on } \partial\Omega.
$$

程序里取粘性系数

$$
\nu = 1,
$$

真解采用

$$
\begin{aligned}
 u_1 &= -x^2(x-1)^2y(y-1)(2y-1), \\
 u_2 &= x(x-1)(2x-1)y^2(y-1)^2, \\
 p &= x^6-y^6,
\end{aligned}
$$

并令右端由真解反代得到：

$$
\boldsymbol{f} = -\nu\,\nabla\cdot(\nabla \boldsymbol{u}) + \nabla p.
$$

## 9. Stokes 的 HDG 离散空间

程序使用四个离散未知量：

- 单元内部速度 $\boldsymbol{u}_h \in V_h$；
- facet 速度迹 $\bar{\boldsymbol{u}}_h \in \bar V_h$；
- 单元内部压力 $p_h \in Q_h$；
- facet 压力迹 $\bar p_h \in \bar Q_h$。

对应空间写成

$$
V_h = \{ \boldsymbol{v}_h\in [L^2(\Omega)]^d : \boldsymbol{v}_h|_K \in [\mathbb{P}_k(K)]^d,\ \forall K\in\mathcal{T}_h \},
$$

$$
\bar V_h = \{ \bar{\boldsymbol{v}}_h\in [L^2(\mathcal{F}_h)]^d : \bar{\boldsymbol{v}}_h|_F \in [\mathbb{P}_k(F)]^d,\ \forall F\in\mathcal{F}_h \},
$$

$$
Q_h = \{ q_h\in L^2(\Omega) : q_h|_K \in \mathbb{P}_{k-1}(K),\ \forall K\in\mathcal{T}_h \},
$$

$$
\bar Q_h = \{ \bar q_h\in L^2(\mathcal{F}_h) : \bar q_h|_F \in \mathbb{P}_k(F),\ \forall F\in\mathcal{F}_h \}.
$$

因此测试函数为

$$
(\boldsymbol{v}_h, \bar{\boldsymbol{v}}_h, q_h, \bar q_h)
\in V_h\times\bar V_h\times Q_h\times\bar Q_h.
$$

## 10. Stokes 的 HDG 双线性形式

### 10.1 速度扩散部分

程序中的速度双线性形式为

$$
\begin{aligned}
a_h\big((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h),(\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h)\big)
={}& \sum_{K\in\mathcal{T}_h}\int_K \nu\,\nabla \boldsymbol{u}_h : \nabla \boldsymbol{v}_h\,dx \\
&- \sum_{K\in\mathcal{T}_h}\int_{\partial K} \nu\,(\boldsymbol{u}_h-\bar{\boldsymbol{u}}_h)\cdot \partial_n \boldsymbol{v}_h\,ds \\
&- \sum_{K\in\mathcal{T}_h}\int_{\partial K} \nu\,\partial_n \boldsymbol{u}_h\cdot (\boldsymbol{v}_h-\bar{\boldsymbol{v}}_h)\,ds \\
&+ \sum_{K\in\mathcal{T}_h}\int_{\partial K} \nu\,\frac{\alpha}{h_K}(\boldsymbol{u}_h-\bar{\boldsymbol{u}}_h)\cdot(\boldsymbol{v}_h-\bar{\boldsymbol{v}}_h)\,ds.
\end{aligned}
$$

其中程序中取

$$
\alpha = 16k^2.
$$

### 10.2 压力-速度耦合部分

按照程序里的写法，压力-速度耦合双线性形式为

$$
\begin{aligned}
b_h\big((\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h),(p_h,\bar p_h)\big)
={}& -\sum_{K\in\mathcal{T}_h}\int_K p_h\,\nabla\cdot\boldsymbol{v}_h\,dx \\
&+ \sum_{K\in\mathcal{T}_h}\int_{\partial K}(\boldsymbol{v}_h\cdot\boldsymbol{n})\,\bar p_h\,ds.
\end{aligned}
$$

同样，连续性方程对应的离散形式写成

$$
\begin{aligned}
b_h\big((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h),(q_h,\bar q_h)\big)
={}& -\sum_{K\in\mathcal{T}_h}\int_K q_h\,\nabla\cdot\boldsymbol{u}_h\,dx \\
&+ \sum_{K\in\mathcal{T}_h}\int_{\partial K}(\boldsymbol{u}_h\cdot\boldsymbol{n})\,\bar q_h\,ds.
\end{aligned}
$$

### 10.3 压力小惩罚项

为了避免压力常数模态带来的奇异性，程序额外加入一个很小的压力惩罚项：

$$
\epsilon_p \sum_{K\in\mathcal{T}_h}\int_K p_h q_h\,dx,
$$

其中程序里取

$$
\epsilon_p = 10^{-12}.
$$

这一步不是连续 Stokes 模型本身的一部分，而是数值实现中为了固定 pressure gauge 而加入的小正则项。

### 10.4 总双线性形式

因此，程序中的总双线性形式可写成

$$
\mathcal{A}_h
\big((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h,p_h,\bar p_h),
(\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h,q_h,\bar q_h)\big)
= a_h\big((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h),(\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h)\big)
+ b_h\big((\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h),(p_h,\bar p_h)\big)
+ b_h\big((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h),(q_h,\bar q_h)\big)
+ \epsilon_p \sum_{K\in\mathcal{T}_h}\int_K p_h q_h\,dx.
$$

## 11. Stokes 的线性形式

程序右端写成

$$
\mathcal{L}_h(\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h,q_h,\bar q_h)
= \sum_{K\in\mathcal{T}_h}\int_K \boldsymbol{f}\cdot\boldsymbol{v}_h\,dx,
$$

其余关于 $\bar{\boldsymbol{v}}_h$、$q_h$、$\bar q_h$ 的项都被写成零项，只是为了保持 block mixed-domain 形式的完整结构。

## 12. Stokes 的 HDG 离散变分问题

因此，当前脚本求解的离散问题可以表述为：

> 求
> $$
> (\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h,p_h,\bar p_h)
> \in V_h\times\bar V_h\times Q_h\times\bar Q_h,
> $$
> 并满足边界上的速度迹条件
> $$
> \bar{\boldsymbol{u}}_h = \boldsymbol{u}_D \qquad \text{on } \partial\Omega,
> $$
> 使得对任意
> $$
> (\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h,q_h,\bar q_h)
> \in V_h\times\bar V_h\times Q_h\times\bar Q_h
> $$
> 都有
> $$
> \mathcal{A}_h((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h,p_h,\bar p_h),
> (\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h,q_h,\bar q_h))
> = \mathcal{L}_h(\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h,q_h,\bar q_h).
> $$

## 13. 与 `hdg_stokes_fenicsx_0_9.py` 代码的对应关系

- `V`, `Vbar`, `Q`, `Qbar`：分别对应 $V_h$、$\bar V_h$、$Q_h$、$\bar Q_h$；
- `u_h`, `ubar_h`, `p_h`, `pbar_h`：对应四个试函数；
- `v_h`, `vbar_h`, `q_h`, `qbar_h`：对应四个测试函数；
- `a`：对应速度扩散型 HDG 双线性形式；
- `b_vp`：对应 $b_h((\boldsymbol{v}_h,\bar{\boldsymbol{v}}_h),(p_h,\bar p_h))$；
- `b_uq`：对应 $b_h((\boldsymbol{u}_h,\bar{\boldsymbol{u}}_h),(q_h,\bar q_h))$；
- `epsilon_p`：对应小压力惩罚参数；
- `A_form`：对应当前 Stokes 的总双线性形式；
- `L_form`：对应当前 Stokes 的右端线性形式；
- `velocity_bc = fem.dirichletbc(ubar_bc, velocity_dofs)`：对应边界上对速度迹 $\bar{\boldsymbol{u}}_h$ 施加 Dirichlet 条件。
