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
