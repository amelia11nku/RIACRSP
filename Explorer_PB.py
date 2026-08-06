# -*- coding: utf-8 -*-
"""
Explorer_QLearning.py
探索种群的 12 种扰动算子集与 Q-Learning 逻辑
"""
import random
import math
import numpy as np
from typing import List, Tuple

class ExplorerPerturbation:
    def __init__(self, instance, op_index_to_id, op_id_to_index):
        self.num_machines = instance.num_machines
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.num_jobs = instance.num_jobs
        self.processing_times = instance.processing_times
        self.priority_dict = instance.priority_dict
        self.op_index_to_id = op_index_to_id
        self.op_id_to_index = op_id_to_index
        
    def _is_valid_OS(self, OS_indices: List[int]) -> bool:
        """快速检验 OS precedence"""
        expected = set(self.op_index_to_id)
        if len(OS_indices) != len(expected) or set(OS_indices) != expected:
            return False
        pos = {op: i for i, op in enumerate(OS_indices)}
        for op in OS_indices:
            op_id = self.op_index_to_id[op]
            for pred in self.priority_dict.get(op_id, set()):
                if pos.get(self.op_id_to_index[pred], -1) >= pos[op]:
                    return False
        return True

    def _reinsert_ops(self, OS_indices: List[int], ops_to_reinsert_ids: List[str]) -> List[int]:
        """Randomly reinsert selected operations with precedence guaranteed by construction."""
        selected = {
            self.op_id_to_index[op_id]
            for op_id in ops_to_reinsert_ids
            if op_id in self.op_id_to_index
        }
        if not selected:
            return OS_indices.copy()

        nodes = list(OS_indices)
        remaining = [op_idx for op_idx in nodes if op_idx not in selected]
        successors = {op_idx: set() for op_idx in nodes}
        in_degree = {op_idx: 0 for op_idx in nodes}

        def add_edge(left, right):
            if right not in successors[left]:
                successors[left].add(right)
                in_degree[right] += 1

        # Preserve the relative order of all operations that were not selected.
        for left, right in zip(remaining, remaining[1:]):
            add_edge(left, right)

        # Enforce every technological precedence edge, including transitive effects.
        for op_id, predecessors in self.priority_dict.items():
            op_idx = self.op_id_to_index[op_id]
            for predecessor in predecessors:
                add_edge(self.op_id_to_index[predecessor], op_idx)

        available = [op_idx for op_idx in nodes if in_degree[op_idx] == 0]
        new_OS = []
        while available:
            op_idx = random.choice(available)
            available.remove(op_idx)
            new_OS.append(op_idx)
            for successor in successors[op_idx]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    available.append(successor)

        if len(new_OS) != len(nodes):
            raise ValueError("Reinsertion constraint graph contains a cycle.")
        return new_OS

    def perturb(self, action: int, MS: List[int], TW: List[int], TF: List[int], OS: List[int], cp_ops=None):
        """原位修改为主的扰动路由，返回拷贝的新解"""
        nMS, nTW, nTF, nOS = MS.copy(), TW.copy(), TF.copy(), OS.copy()
        n_ops = len(OS)
        
        if action == 0:  # MS_s
            # 随机选一个工序，随机选一个可用机器替换
            idx = random.randrange(n_ops)
            alts = list(self.processing_times.get(self.op_index_to_id[OS[idx]], {}).keys())
            if alts: nMS[OS[idx]-1] = int(random.choice(alts))
            
        elif action == 1:  # MS_m
            # 随机选一个作业，随机选一个工序，随机选一个可用机器替换
            job = random.randint(1, self.num_jobs)
            for i in range(n_ops):
                op_id = self.op_index_to_id[i + 1]
                if int(op_id.split('_')[0][1:]) == job:
                    alts = list(self.processing_times.get(op_id, {}).keys())
                    if alts: nMS[i] = int(random.choice(alts))
                    
        elif action == 2:  # MS_l
            # 选一批关键路径上的工序，随机选一个可用机器替换
            if cp_ops:
                k = max(2, math.ceil(0.5 * len(cp_ops)))
                for op in random.sample(cp_ops, min(k, len(cp_ops))):
                    alts = list(self.processing_times.get(op.op_id, {}).keys())
                    if alts: nMS[self.op_id_to_index[op.op_id]-1] = int(random.choice(alts))
                    
        elif action == 3:  # TW_s
            idx = random.randrange(n_ops)
            nTW[OS[idx]-1] = random.randint(1, self.num_agvs_w)
            
        elif action == 4:  # TW_m
            job = random.randint(1, self.num_jobs)
            for i in range(n_ops):
                if int(self.op_index_to_id[i + 1].split('_')[0][1:]) == job:
                    nTW[i] = random.randint(1, self.num_agvs_w)
                    
        elif action == 5:  # TW_l
            if cp_ops:
                k = max(2, math.ceil(0.5 * len(cp_ops)))
                for op in random.sample(cp_ops, min(k, len(cp_ops))):
                    nTW[self.op_id_to_index[op.op_id]-1] = random.randint(1, self.num_agvs_w)
                    
        elif action == 6:  # TF_s
            idx = random.randrange(n_ops)
            nTF[OS[idx]-1] = random.randint(1, self.num_agvs_f)
            
        elif action == 7:  # TF_m
            job = random.randint(1, self.num_jobs)
            for i in range(n_ops):
                if int(self.op_index_to_id[i + 1].split('_')[0][1:]) == job:
                    nTF[i] = random.randint(1, self.num_agvs_f)
                    
        elif action == 8:  # TF_l
            if cp_ops:
                k = max(2, math.ceil(0.5 * len(cp_ops)))
                for op in random.sample(cp_ops, min(k, len(cp_ops))):
                    nTF[self.op_id_to_index[op.op_id]-1] = random.randint(1, self.num_agvs_f)

        elif action == 9:  # OS_s
            for _ in range(5):
                idx1, idx2 = random.sample(range(n_ops), 2)
                op1_id, op2_id = self.op_index_to_id[nOS[idx1]], self.op_index_to_id[nOS[idx2]]
                if op1_id.split('_')[0] != op2_id.split('_')[0]:
                    nOS[idx1], nOS[idx2] = nOS[idx2], nOS[idx1]
                    if self._is_valid_OS(nOS): break
                    nOS[idx1], nOS[idx2] = nOS[idx2], nOS[idx1] # revert
                    
        elif action == 10: # OS_m
            job = random.randint(1, self.num_jobs)
            job_ops = [self.op_index_to_id[i+1] for i in range(n_ops) if int(self.op_index_to_id[i+1].split('_')[0][1:]) == job]
            nOS = self._reinsert_ops(nOS, job_ops)
            
        elif action == 11: # OS_l
            if cp_ops:
                k = max(2, math.ceil(0.5 * len(cp_ops)))
                chosen_ids = [op.op_id for op in random.sample(cp_ops, min(k, len(cp_ops)))]
                nOS = self._reinsert_ops(nOS, chosen_ids)

        return nMS, nTW, nTF, nOS

class ExplorerQLearning:
    def __init__(self, n_actions=12, alpha=0.15, gamma=0.3, epsilon=0.25):
        # 3 (quality) * 3 (novelty) * 4 (distance source: MS, TW, TF, OS) = 36 states
        self.n_states = 36 
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)

    def calculate_layer_distances(self, x: Tuple, target: Tuple, n_ops: int) -> Tuple[float, float, float, float]:
        """计算个体与目标（gbest 或 elite）在四层编码上的独立距离"""
        if target is None: return 0.0, 0.0, 0.0, 0.0
        t_MS, t_TW, t_TF, t_OS = target[2], target[3], target[4], target[5]
        
        d_MS = sum(a != b for a, b in zip(x[0], t_MS)) / n_ops
        d_TW = sum(a != b for a, b in zip(x[1], t_TW)) / n_ops
        d_TF = sum(a != b for a, b in zip(x[2], t_TF)) / n_ops
        d_OS = sum(a != b for a, b in zip(x[3], t_OS)) / n_ops
        
        return d_MS, d_TW, d_TF, d_OS

    def calculate_mixed_layer_distances1(self, x: Tuple, gbest: Tuple, elites: List[Tuple], n_ops: int, w1: float = 0.5, w2: float = 0.5) -> Tuple[float, float, float, float]:
        """计算个体相对于 gbest 和 群体代表(elites) 的混合新颖性距离"""
        # 1. 计算到 gbest 的绝对距离
        d_MS_g, d_TW_g, d_TF_g, d_OS_g = self.calculate_layer_distances(x, gbest, n_ops)
        if not elites:
            return d_MS_g, d_TW_g, d_TF_g, d_OS_g
            
        # 2. 计算到群体代表 (elites) 的平均距离
        d_MS_p, d_TW_p, d_TF_p, d_OS_p = 0.0, 0.0, 0.0, 0.0
        valid_count = 0
        
        for elite in elites:
            elite_x = elite[2:]
            # 排除自我对比
            if x is elite_x or x == elite_x:
                continue
            e_MS, e_TW, e_TF, e_OS = self.calculate_layer_distances(x, elite, n_ops)
            d_MS_p += e_MS
            d_TW_p += e_TW
            d_TF_p += e_TF
            d_OS_p += e_OS
            valid_count += 1
            
        if valid_count == 0:
            return d_MS_g, d_TW_g, d_TF_g, d_OS_g
            
        # 3. 返回加权混合后的距离
        return (
            w1 * d_MS_g + w2 * (d_MS_p / valid_count),
            w1 * d_TW_g + w2 * (d_TW_p / valid_count),
            w1 * d_TF_g + w2 * (d_TF_p / valid_count),
            w1 * d_OS_g + w2 * (d_OS_p / valid_count)
        )
    
    def calculate_mixed_layer_distances(self, x: Tuple, elites: List[Tuple], n_ops: int) -> Tuple[float, float, float, float]:
        """Calculate layer-wise novelty distances to the representative elite set."""
        if not elites:
            return 0.0, 0.0, 0.0, 0.0

        d_MS, d_TW, d_TF, d_OS = 0.0, 0.0, 0.0, 0.0
        valid_count = 0

        for elite in elites:
            elite_x = elite[2:]
            if x is elite_x or x == elite_x:
                continue

            e_MS, e_TW, e_TF, e_OS = self.calculate_layer_distances(x, elite, n_ops)
            d_MS += e_MS
            d_TW += e_TW
            d_TF += e_TF
            d_OS += e_OS
            valid_count += 1

        if valid_count == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            d_MS / valid_count,
            d_TW / valid_count,
            d_TF / valid_count,
            d_OS / valid_count,
        )

    def get_state(self, mk: float, q1: float, q2: float, dist: float, d1: float, d2: float, d_MS: float, d_TW: float, d_TF: float, d_OS: float) -> int:
        """
        【统一状态计算入口】
        根据目标值(mk)、综合距离(dist)以及各层距离(d_MS, d_TW, d_TF, d_OS)，计算当前状态编号。
        """
        # 1. q_tier: 质量层 (0: 优秀, 1: 中等, 2: 较差)
        q_tier = 0 if mk <= q1 else (2 if mk >= q2 else 1)
        
        # 2. n_tier: 新颖层 (0: 同质化, 1: 一般, 2: 新颖)
        n_tier = 2 if dist >= d2 else (0 if dist <= d1 else 1)
        
        # 3. s_tier: 差异来源层 (0: MS主导, 1: TW主导, 2: TF主导, 3: OS主导)
        max_d = max(d_MS, d_TW, d_TF, d_OS)
        
        if max_d == 0: s_tier = 3         # 若完全相同或无差异，默认归为 OS 差异兜底
        elif max_d == d_MS: s_tier = 0
        elif max_d == d_TW: s_tier = 1
        elif max_d == d_TF: s_tier = 2
        else: s_tier = 3                  # OS 主导
            
        # 返回映射的 36 宫格状态 (3 * 3 * 4)
        # 索引计算: q_tier * 12 (因为后面有3*4=12种组合) + n_tier * 4 + s_tier
        return q_tier * 12 + n_tier * 4 + s_tier

    def compute_explorer_reward(self, Qgain: float, NovGain: float) -> float:
        return 0.5 * Qgain + 0.5 * NovGain

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon: 
            return random.randint(0, self.n_actions - 1)
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(random.choice(best_actions.tolist()))

    def update(self, state: int, action: int, reward: float, next_state: int):
        old_q = self.q_table[state, action]
        max_next = np.max(self.q_table[next_state])
        self.q_table[state, action] = old_q + self.alpha * (reward + self.gamma * max_next - old_q)

    def decay_end_of_generation(self, decay_rate=0.01, min_epsilon=0.01, min_alpha=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * (1.0 - decay_rate))
        self.alpha   = max(min_alpha, self.alpha * (1.0 - decay_rate))
