# # -*- coding: utf-8 -*-
# import onnx
# import onnxruntime as rt
# import numpy as  np
#
# # Load the ONNX model
# model = onnx.load("dino-main/weights/dino_resnet50.onnx")
#
# # Check that the IR is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))
#
#
# data = np.array(np.random.randn(1,3,224,224))
# sess = rt.InferenceSession("dino-main/weights/dino_resnet50.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
#
# pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
# print(pred_onx)
# print(np.argmax(pred_onx))

import heapq


def astar(graph, start, goal):
    open_list = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_list:
        current_f, current_node = heapq.heappop(open_list)

        if current_node == goal:
            return reconstruct_path(came_from, goal)

        for neighbor, cost in graph[current_node].items():
            tentative_g_score = g_score[current_node] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None


def reconstruct_path(came_from, current_node):
    total_path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        total_path.insert(0, current_node)
    return total_path


def heuristic(node, goal):
    # 这里可以使用启发式函数来估计节点到目标的距离，例如欧氏距离
    return 0


# 定义节点和边
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 5, 'D': 10},
    'C': {'A': 2, 'B': 5, 'D': 3},
    'D': {'B': 10, 'C': 3}
}

# 选择起点和终点
start_node = 'D'
goal_node = 'A'

# 执行A*算法
path = astar(graph, start_node, goal_node)
print("最佳路径:", path)

import math
import torch
from torch import nn


class EfficientCA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(EfficientCA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # [b, c, 1, 1]
        print(y.shape)
        print(y.squeeze(-1).shape)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [b, 1, c]
        print(y.shape)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]
        print(y.shape)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == "__main__":
    t = torch.ones((32, 256, 24, 24))
    eca = EfficientCA(256)
    out = eca(t)
    print(out.shape)










