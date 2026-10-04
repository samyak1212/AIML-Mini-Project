"""Smart Retail Intelligence - Part 2
Apriori recommendations and BFS, DFS, UCS, Greedy and A*.
Run segmentation_prediction.py first.
"""
from collections import deque
import heapq
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(df = pd.read_csv("data/retail_transactions_clean.csv"))

# 1. Convert transactions into invoice-product baskets.
basket = df.groupby(["Invoice", "Description"])["Quantity"].sum().unstack(fill_value=0)
basket = (basket > 0).astype(int)
print("Invoices:", len(basket), "Products:", len(basket.columns))

# 2. Apriori finds products that are frequently bought together.
frequent = apriori(basket, min_support=.02, use_colnames=True, low_memory=True)
rules = association_rules(frequent, metric="confidence", min_threshold=.3)
rules = rules.sort_values("lift", ascending=False)
print("\nTOP PRODUCT RULES")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# 3. Build a graph: products are nodes and association rules are edges.
graph = {}
for _, r in rules.iterrows():
    cost = 1 / max(float(r.lift), .0001)
    for a in r.antecedents:
        for b in r.consequents:
            graph.setdefault(a, {})
            graph.setdefault(b, {})
            if b not in graph[a] or cost < graph[a][b]:
                graph[a][b] = cost
                graph[b][a] = cost

# 4. BFS: explore level by level.
def bfs(start, goal):
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft(); node = path[-1]
        if node == goal: return path
        for n in graph.get(node, {}):
            if n not in seen:
                seen.add(n); q.append(path + [n])
    return None

# 5. DFS: explore one branch deeply before another.
def dfs(start, goal):
    stack = [[start]]; seen = set()
    while stack:
        path = stack.pop(); node = path[-1]
        if node == goal: return path
        if node in seen: continue
        seen.add(node)
        for n in graph.get(node, {}):
            if n not in seen: stack.append(path + [n])
    return None

# 6. UCS: choose the lowest total path cost.
def ucs(start, goal):
    heap = [(0, [start])]; best = {start: 0}
    while heap:
        cost, path = heapq.heappop(heap); node = path[-1]
        if node == goal: return path, cost
        for n, edge in graph.get(node, {}).items():
            new = cost + edge
            if new < best.get(n, float("inf")):
                best[n] = new; heapq.heappush(heap, (new, path + [n]))
    return None, float("inf")

# 7. Simple heuristic used by Greedy and A*.
def h(node, goal):
    return graph.get(node, {}).get(goal, 1.0)

# 8. Greedy Best First: choose the node with the best heuristic.
def greedy(start, goal):
    heap = [(h(start, goal), [start])]; seen = set()
    while heap:
        _, path = heapq.heappop(heap); node = path[-1]
        if node == goal: return path
        if node in seen: continue
        seen.add(node)
        for n in graph.get(node, {}):
            if n not in seen: heapq.heappush(heap, (h(n, goal), path + [n]))
    return None

# 9. A*: combine path cost and heuristic.
def astar(start, goal):
    heap = [(h(start, goal), 0, [start])]; best = {start: 0}
    while heap:
        _, cost, path = heapq.heappop(heap); node = path[-1]
        if node == goal: return path, cost
        for n, edge in graph.get(node, {}).items():
            new = cost + edge
            if new < best.get(n, float("inf")):
                best[n] = new
                heapq.heappush(heap, (new + h(n, goal), new, path + [n]))
    return None, float("inf")

# 10. Demonstrate all five search algorithms.
products = list(graph)
if len(products) >= 2:
    start, goal = products[0], products[1]
    print("\nSEARCH")
    print("Start:", start)
    print("Goal :", goal)
    print("BFS:", bfs(start, goal))
    print("DFS:", dfs(start, goal))
    p, c = ucs(start, goal); print("UCS:", p, "Cost:", round(c, 3))
    print("Greedy:", greedy(start, goal))
    p, c = astar(start, goal); print("A*:", p, "Cost:", round(c, 3))

# 11. Simple recommendation function.
def recommend(product, n=5):
    return sorted(graph.get(product, {}).items(), key=lambda x: x[1])[:n]

if products:
    print("\nRecommendations for:", products[0])
    for product, cost in recommend(products[0]):
        print(" ->", product, "cost:", round(cost, 3))

print("\nDone.")
