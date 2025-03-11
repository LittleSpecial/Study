# 1.Open-Reasoner-Zero
## 2025.3.11

[Github](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)\
Github地址：https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero\

## 强化学习算法：
### GAE(广义优势估计 General Advantage Estimation):

#### 1.GAE的作用

GAE 通过结合多个 n-step 优势估计，并使用指数加权平均（由超参数 λ 控制），在 **偏差（bias）** 和 **方差（variance）** 之间提供了权衡。这种方法在强化学习中用于更稳定和高效的策略梯度估计。

#### 2.公式

GAE 计算优势函数的方法如下：

\[
\hat{A}_t = \delta_t + (\gamma\lambda) \delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1} \delta_{T-1}
\]

其中：

- \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) 是 **时序差分（TD，Temporal Difference）残差**，用于衡量当前状态值估计与实际的误差。\({R}_t\)是实际的累计奖励，\(V(s_t)\)是当前状态的value function，\(\gamma V(s_{t+1})\)是从下一个状态估计的未来回报。
- 相比于 Monte Carlo 方法（必须等整个轨迹结束后才能计算累积奖励），TD 学习可以在每个时间步就进行更新，使得学习更加高效，适用于在线学习。
- \(\gamma\) 是 **折扣因子（discount factor）**，决定了未来奖励相对于即时奖励的重要性。
- \(\lambda\) 是 **GAE 估计中的平衡参数**，当 \(\lambda \rightarrow 1\) 时，GAE 近似于蒙特卡洛方法（较低偏差但高方差）；当 \(\lambda \rightarrow 0\) 时，GAE 退化为一阶 TD 误差（较高偏差但低方差）。


#### 3.具体解释

GAE旨在减少策略梯度方差的同时，保持较低的偏差
\({A}_t = {R}_t - V({s_t})\)

- \({R}_t\)是实际的累计奖励

- \(V(s_{t})\)是状态值函数(Value Function)
### PPO(Proximal Policy Optimization)

#### Algorithm Target

- **策略优化目标**（Policy Objective）：更新策略模型参数 **θ**，最大化期望奖励。
- **值函数优化目标**（Value Objective）：更新值函数模型参数 **φ**，最小化值函数损失。

#### 目标函数

##### **(1) 策略优化目标函数**
PPO 通过 **裁剪（Clipping）策略比率** 避免策略更新过大，提高训练稳定性：
\[
J_{PPO} (\theta) = \mathbb{E}_{t, s_t, a_t \sim \pi_{\theta_{old}}} \left[ 
\min \left( 
\frac{\pi_{\theta} (a_t | s_t)}{\pi_{\theta_{old}} (a_t | s_t)} \hat{A}_t, 
\text{clip} \left( 
\frac{\pi_{\theta} (a_t | s_t)}{\pi_{\theta_{old}} (a_t | s_t)}, 1 - \epsilon, 1 + \epsilon
\right) \hat{A}_t
\right) 
\right]
\]
- **\(\pi_{\theta}(a_t | s_t)\) 是当前策略，\(\pi_{\theta_{old}}(a_t | s_t)\) 是旧策略**。
- **裁剪参数 \(\epsilon\)** 限制策略更新的幅度，防止剧烈变化导致训练不稳定。
- **\(\hat{A}_t\) 是优势函数（Advantage Function）**，衡量当前动作的相对价值。
- **调整策略以选择最优的动作，以 最大化长期累计奖励（也就是让总收益尽可能高）。**

##### **(2) 值函数优化目标**
值函数损失最小化：
\[
J_{value} (\phi) = \frac{1}{2} \mathbb{E}_{t, s_t, a_t \sim \pi_{\theta_{old}}} \left[ 
(V_{\phi} (s_t) - R_t)^2
\right]
\]
- **\(V_{\phi}(s_t)\) 是当前值函数估计**。
- **\(R_t\) 是折扣回报（Discounted Return）：**
  \[
  R_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
  \]
  其中，\(\gamma\) 是折扣因子。
- **评估当前状态的长期收益。 **

---

## Key Findings

### Minimal Reward Function Design
一个**简单的基于规则的奖励函数**不仅足够有效，而且是**最优的**，因为**最小化设计**可以避免潜在的奖励篡改（reward hacking）。值得注意的是，即使是未对齐的基础模型也能快速适应所需格式，这表明该任务是**相对简单的**，无需复杂的奖励工程（reward engineering）。

### Loss Function

训练过程**无需依赖 KL 正则化技术**（例如 KL 形状奖励和 KL 损失），不同于 RLHF（基于人类反馈的强化学习）社区的主流方法以及Reasoner 模型。这种方法的稳定性为未来**大规模强化学习**提供了良好的潜力。

KL 正则化（Kullback-Leibler Regularization）是一种常见的技术，主要用于**约束策略更新幅度**，防止强化学习中的策略发生剧烈变化。它通常通过**KL 散度（Kullback-Leibler Divergence）**来衡量新旧策略之间的差异，并在损失函数中加入相应的正则项。

#### **KL 散度公式**
KL 散度用于衡量两个概率分布 \( P \) 和 \( Q \) 之间的差异，其数学定义如下：
\[
D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
\]
在强化学习中，\( P \) 代表旧策略 \( \pi_{\text{old}}(a | s) \)，\( Q \) 代表新策略 \( \pi_{\theta}(a | s) \)，KL 散度可以衡量**新旧策略的相似性**。

#### **KL 正则化的作用**
- **防止策略崩溃**：避免新策略偏离旧策略过远，从而降低训练不稳定性。
- **提高训练稳定性**：通过控制策略更新的幅度，使策略优化过程更平稳。
- **常见的 KL 约束方式**：
  - **KL 惩罚项**：直接在损失函数中加入 \( \beta D_{\text{KL}}(\pi_{\theta} \| \pi_{\text{old}}) \)。
  - **KL 约束优化**：如 TRPO（Trust Region Policy Optimization），确保 KL 散度不超过设定阈值。
  - **PPO（Proximal Policy Optimization）**：使用**裁剪（Clipping）**技术，间接约束 KL 散度。

### Scale up Training Data
**数据的数量和多样性扩展**对于 Reasoner-Zero 的训练至关重要。仅在**有限的学术数据集**（如 MATH）上训练会导致模型**快速达到性能瓶颈**，而我们精心构建的大规模、多样化数据集，使模型可以**持续扩展**，并且在**训练集和测试集上均未出现饱和现象**。 

## 实验过程

### 3.1 训练细节与超参数

- **模型初始化**：
  - 策略网络（policy network）和评论网络（critic network）均采用 **Qwen-2.5 基础模型**（7B 和 32B 版本）。
  - 价值头（value head）**随机初始化**于 \( U(-\sqrt{5}, \sqrt{5}) \)，且**无偏置项**。
  - **策略网络与评论网络不共享权重**。

- **优化器与学习率**：
  - 采用 **AdamW** 优化器，超参数设置为 \( \beta = [0.9, 0.95] \)，无权重衰减（weight decay）。
  - 学习率：
    - **策略网络**：\( 1 \times 10^{-6} \)
    - **评论网络**：\( 5 \times 10^{-6} \)
  - **学习率调度**：恒定学习率 + **线性预热 50 步**。

- **训练过程**：
  - 采用 **sample packing** 技术，每个生成步骤：
    - **128 个不同的 prompt**
    - **每个 prompt 生成 64 个响应**
    - 生成时温度（temperature）和 top-p 采样均设为 **1.0**。
  - **策略网络采用严格的 on-policy 优化**，即**每次生成对应一次优化更新**。
  - **评论网络支持 off-policy 更新**，**每次迭代执行 12 次 mini-batch 训练**。
  - 训练中应用 **batch 级别的优势归一化（advantage normalization）**。

- **稳定性与正则化**：
  - **未使用 KL 正则化或熵奖励（entropy bonus）**。
  - 证明了 **标准 PPO（vanilla PPO）可实现稳定训练**，无需额外稳定化技术。

- **评测**：
  - 评测模型的 **推理能力**，使用数据集：
    - **数学推理**：GPQA DIAMOND、AIME2024、AIME2025、MATH500
    - **编程能力**：LIVECODEBENCH
    - **综合能力**：MMLU、MMLU_PRO
  - 主要指标：**每个问题 16 个样本的平均准确率**。

#### 相关概念

##### **1. Policy Network（策略网络） & Critic Network（评论网络）**
在强化学习中，策略网络（Policy Network）和评论网络（Critic Network）是用于优化智能体决策的两个核心部分：

- **策略网络（Policy Network）**：
  - 负责 **决策**，即在某个状态 \( s \) 下选择动作 \( a \)。
  - 其核心目标是**最大化期望奖励**，即学习一个最优策略 \( \pi_{\theta}(a | s) \)。
  - 在 PPO 等策略梯度方法中，策略网络通过 **梯度上升** 来优化策略，使其获得更高的累积奖励。

- **评论网络（Critic Network）**：
  - 负责 **评估策略的好坏**，即预测**当前状态的值函数 \( V(s) \)**。
  - 计算**时序差分误差（TD Error）**：
    \[
    \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
    \]
  - 通过 **梯度下降** 使得估计值更接近真实的折扣回报，提供更稳定的优化信号。
  - **策略网络依赖评论网络的反馈来进行优化**。

---

##### **2. 线性预热（Linear Warm-up）**
- 训练初期，直接使用较大学习率可能会导致不稳定甚至发散。
- **线性预热** 指的是：
  - 训练前 **N** 个优化步骤中，学习率 **从一个较小值逐渐升高到目标学习率**。
  - 这样可以**稳定训练过程**，防止参数发生剧烈变化。

---

##### **3. AdamW 优化器**
AdamW（Adam with Weight Decay）是 Adam 优化器的改进版本：

- **Adam（Adaptive Moment Estimation）**：
  - 结合 **动量优化（Momentum）** 和 **自适应学习率（Adaptive Learning Rate）**，可适应不同梯度规模，提高收敛速度。
  - 更新规则：
    \[
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
    \]
    \[
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    \]
    \[
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t
    \]
  - 其中 \( \beta_1 \) 和 \( \beta_2 \) 控制一阶、二阶动量的权重，默认设为 0.9 和 0.999。

- **AdamW 改进点**：
  - 在 Adam 的基础上，**去掉了 L2 正则化对动量项的影响**，而是**直接在权重更新时引入权重衰减（Weight Decay）**：
    \[
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t - \lambda \theta_{t-1}
    \]
  - 这样可以**更好地防止过拟合**，适用于大规模神经网络训练。

---

##### **4. 生成时温度（Temperature）和 Top-p 采样**
在文本生成任务中，模型的输出通常是一个概率分布，我们可以通过不同的方法来控制生成的多样性：

- **温度（Temperature）**：
  - 控制输出分布的平滑程度：
    \[
    P(a) = \frac{\exp(\frac{\log P(a)}{T})}{\sum_{b} \exp(\frac{\log P(b)}{T})}
    \]
  - \( T \) 越大，**概率分布更平滑**，采样更随机，生成内容更加多样化。
  - \( T \) 越小，**分布更陡峭**，更倾向于选择高概率的词，生成内容更确定性。
  - \( T = 1.0 \) 是标准设置，\( T < 1.0 \) 时生成更确定，\( T > 1.0 \) 时更随机。

- **Top-p 采样（Nucleus Sampling）**：
  - 传统的 Top-k 采样方法是选择**前 k 个概率最高的词**进行采样，而 **Top-p 采样** 采用的是 **累积概率截断**：
    1. 计算所有可能输出词的概率，并按概率降序排序。
    2. **截取** 一个最小的候选集合，使其**累积概率至少达到 p**。
    3. 在该集合中**进行随机采样**。

  - **效果**：
    - 避免了 Top-k 固定 k 值的局限，使得采样更加动态。
    - 对于确定性高的场景（如数学题解答），Top-p 采样可以减少生成错误，提高准确性。
    - \( p = 1.0 \) 相当于不裁剪，\( p \) 越小，生成越保守。

---


