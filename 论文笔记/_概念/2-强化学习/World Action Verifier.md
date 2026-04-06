# World Action Verifier

World Action Verifier (WAV) 是一个自改进的 World Model 框架，通过 Forward-Inverse 不对称性实现模型自我改进。

## 核心机制

- **State Plausibility**: 检查预测的状态是否合理（从无 action 视频语料生成多样 subgoal）
- **Action Reachability**: 检查 action 是否能到达该状态（用稀疏 inverse model 推断）
- **循环一致性验证**: 通过验证 state plausibility 和 action reachability 的循环一致性来发现预测错误

## 优势

- **不依赖大量 action-labeled 数据**: 利用更容易获取的 action-free 视频语料
- **样本效率**: 2x 更高的样本效率
- **下游性能提升**: 下游 policy 性能提升 18%

## 相关链接

- [[World Model]]
- [[JEPA]]
