graph = {}
graph["a"] = {}
graph["a"]["c"] = 4
graph["a"]["d"] = 2
graph["b"] = {}
graph["b"]["a"] = 8
graph["b"]["d"] = 7
graph["c"] = {}
graph["c"]["d"] = 6
graph["c"]["fin"] = 3
graph["d"] = {}
graph["d"]["fin"] = 1
graph["fin"] = {}

infinity = float("inf")
costs = {}
costs["a"] = 5
costs["b"] = 2
costs["c"] = infinity
costs["d"] = infinity
costs["fin"] = infinity


parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["c"] = None
parents["d"] = None
parents["fin"] = None

processed = []

def find_lowest_cost_node(costs): 
    lowest_cost = float("inf") 
    lowest_cost_node = None
    for node in costs: # 遍历所有的节点
        cost = costs[node]
        if cost < lowest_cost and node not in processed: # 如果当前节点的开销更低 且未处理过
            lowest_cost = cost # 就将其视为开销最低的节点
            lowest_cost_node = node 
    return lowest_cost_node

node = find_lowest_cost_node(costs) # 在未处理的节点中找出开销最小的节点
while node is not None: # 循环在所有节点都被处理过后结束
    cost = costs[node] 
    neighbors = graph[node] 
    for n in neighbors.keys(): # 遍历当前节点的所有邻居
        new_cost = cost + neighbors[n] 
        if costs[n] > new_cost: # 如果经当前节点前往该邻居更近
            costs[n] = new_cost # 就更新该邻居的开销
            parents[n] = node # 同时将该邻居的父节点设置为当前节点
    processed.append(node) # 将当前节点标记为处理过
    node = find_lowest_cost_node(costs) # 找出接下来要处理的节点，并循环