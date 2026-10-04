"""
Smart Retail Intelligence - Part 2
Apriori recommendations and BFS, DFS, UCS, Greedy and A*.
Streamlit frontend version.
"""

import heapq
from collections import deque

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Retail Intelligence - Part 2",
    page_icon="🛍️",
    layout="wide"
)

st.title("Smart Retail Intelligence - Product Intelligence")
st.write(
    "Apriori product recommendations with BFS, DFS, UCS, "
    "Greedy Best-First Search and A*."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    possible_paths = [
        "data/retail_transactions_clean.csv",
        "retail_transactions_clean.csv",
    ]

    data_path = next(
        (path for path in possible_paths if __import__("os").path.exists(path)),
        None
    )

    if data_path is None:
        raise FileNotFoundError(
            "retail_transactions_clean.csv was not found. "
            "Put it inside the data/ folder."
        )

    df = pd.read_csv(data_path)

    required = {"Invoice", "Description", "Quantity"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["Invoice", "Description", "Quantity"]).copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Quantity"])
    df = df[df["Quantity"] > 0]

    df["Invoice"] = df["Invoice"].astype(str)
    df["Description"] = df["Description"].astype(str).str.strip()

    df = df[df["Description"] != ""]

    return df


@st.cache_data
def create_basket(df):
    basket = (
        df.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    return (basket > 0).astype(int)


# ============================================================
# APRIORI
# ============================================================

@st.cache_data
def run_apriori(basket, min_support, min_confidence):
    frequent = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        low_memory=True
    )

    if frequent.empty:
        return frequent, pd.DataFrame()

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    return frequent, rules


# ============================================================
# GRAPH
# ============================================================

def build_graph(rules):
    graph = {}

    for _, row in rules.iterrows():
        lift = float(row["lift"])
        cost = 1 / max(lift, 0.0001)

        for a in row["antecedents"]:
            graph.setdefault(a, {})

            for b in row["consequents"]:
                graph.setdefault(b, {})

                if b not in graph[a] or cost < graph[a][b]:
                    graph[a][b] = cost
                    graph[b][a] = cost

    return graph


# ============================================================
# SEARCH ALGORITHMS
# ============================================================

def bfs(graph, start, goal):
    queue = deque([[start]])
    seen = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(path + [neighbor])

    return None


def dfs(graph, start, goal):
    stack = [[start]]
    seen = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node in seen:
            continue

        seen.add(node)

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                stack.append(path + [neighbor])

    return None


def ucs(graph, start, goal):
    heap = [(0.0, [start])]
    best = {start: 0.0}

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path, cost

        if cost > best.get(node, float("inf")):
            continue

        for neighbor, edge_cost in graph.get(node, {}).items():
            new_cost = cost + edge_cost

            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heapq.heappush(
                    heap,
                    (new_cost, path + [neighbor])
                )

    return None, float("inf")


def heuristic(graph, node, goal):
    # Uses the same heuristic idea as the original program:
    # direct graph edge cost, otherwise 1.0.
    return graph.get(node, {}).get(goal, 1.0)


def greedy(graph, start, goal):
    heap = [(heuristic(graph, start, goal), [start])]
    seen = set()

    while heap:
        _, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path

        if node in seen:
            continue

        seen.add(node)

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                heapq.heappush(
                    heap,
                    (
                        heuristic(graph, neighbor, goal),
                        path + [neighbor]
                    )
                )

    return None


def astar(graph, start, goal):
    heap = [(heuristic(graph, start, goal), 0.0, [start])]
    best = {start: 0.0}

    while heap:
        _, cost, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path, cost

        if cost > best.get(node, float("inf")):
            continue

        for neighbor, edge_cost in graph.get(node, {}).items():
            new_cost = cost + edge_cost

            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                priority = new_cost + heuristic(
                    graph, neighbor, goal
                )

                heapq.heappush(
                    heap,
                    (priority, new_cost, path + [neighbor])
                )

    return None, float("inf")


def recommend(graph, product, n=5):
    return sorted(
        graph.get(product, {}).items(),
        key=lambda x: x[1]
    )[:n]


def path_cost(graph, path):
    if not path or len(path) < 2:
        return 0.0

    return sum(
        graph[path[i]][path[i + 1]]
        for i in range(len(path) - 1)
    )


# ============================================================
# LOAD
# ============================================================

try:
    df = load_data()
    basket = create_basket(df)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Controls")

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.005,
    max_value=0.20,
    value=0.02,
    step=0.005
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05
)

top_n = st.sidebar.slider(
    "Top Rules / Recommendations",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)


# ============================================================
# RUN APRIORI
# ============================================================

with st.spinner("Running Apriori and generating association rules..."):
    frequent, rules = run_apriori(
        basket,
        min_support,
        min_confidence
    )

graph = build_graph(rules)


# ============================================================
# OVERVIEW
# ============================================================

st.header("Dataset & Apriori Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions", len(basket))
c2.metric("Products", len(basket.columns))
c3.metric("Frequent Itemsets", len(frequent))
c4.metric("Association Rules", len(rules))

st.divider()


# ============================================================
# APRIORI RESULTS
# ============================================================

st.header("Apriori Association Rules")

if rules.empty:
    st.warning(
        "No association rules were generated. "
        "Try lowering Minimum Support or Minimum Confidence."
    )
else:
    display_rules = rules.head(top_n).copy()

    display_rules["antecedents"] = display_rules[
        "antecedents"
    ].apply(lambda x: ", ".join(sorted(map(str, x))))

    display_rules["consequents"] = display_rules[
        "consequents"
    ].apply(lambda x: ", ".join(sorted(map(str, x))))

    display_rules = display_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]

    display_rules.columns = [
        "Antecedents",
        "Consequents",
        "Support",
        "Confidence",
        "Lift"
    ]

    st.dataframe(
        display_rules,
        use_container_width=True,
        hide_index=True
    )

    # Lift chart
    chart_data = display_rules.head(10).copy()
    chart_data["Rule"] = (
        chart_data["Antecedents"]
        + " → "
        + chart_data["Consequents"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        chart_data["Rule"].iloc[::-1],
        chart_data["Lift"].iloc[::-1]
    )
    ax.set_xlabel("Lift")
    ax.set_title("Top Association Rules by Lift")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PRODUCT GRAPH
# ============================================================

st.header("Product Association Graph")

graph_nodes = list(graph.keys())

g1, g2 = st.columns(2)
g1.metric("Graph Nodes", len(graph_nodes))
g2.metric(
    "Graph Edges",
    sum(len(v) for v in graph.values()) // 2
)

if graph_nodes:
    st.write(
        "Each product is a node. Products connected by association "
        "rules are linked with an edge. Edge cost = 1 / Lift."
    )

    graph_rows = []

    for product, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            if str(product) <= str(neighbor):
                graph_rows.append({
                    "Product A": product,
                    "Product B": neighbor,
                    "Cost": round(cost, 4),
                    "Lift": round(1 / cost, 4)
                })

    graph_df = pd.DataFrame(graph_rows)
    st.dataframe(
        graph_df.sort_values("Lift", ascending=False).head(50),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "The product graph is empty. Lower the Apriori thresholds."
    )


# ============================================================
# SEARCH SECTION
# ============================================================

st.header("Search Algorithms")

if len(graph_nodes) < 2:
    st.warning(
        "At least two products connected by association rules "
        "are required to run the search algorithms."
    )
else:
    s1, s2 = st.columns(2)

    start = s1.selectbox(
        "Start Product",
        graph_nodes,
        key="start_product"
    )

    goal_options = [
        product for product in graph_nodes
        if product != start
    ]

    goal = s2.selectbox(
        "Goal Product",
        goal_options,
        key="goal_product"
    )

    if st.button(
        "Run All Search Algorithms",
        type="primary",
        use_container_width=True
    ):
        bfs_path = bfs(graph, start, goal)
        dfs_path = dfs(graph, start, goal)
        ucs_path, ucs_cost = ucs(graph, start, goal)
        greedy_path = greedy(graph, start, goal)
        astar_path, astar_cost = astar(graph, start, goal)

        st.subheader("Search Results")

        results = [
            {
                "Algorithm": "BFS",
                "Path": " → ".join(map(str, bfs_path))
                if bfs_path else "No path",
                "Steps": len(bfs_path) - 1
                if bfs_path else None,
                "Cost": path_cost(graph, bfs_path)
                if bfs_path else None
            },
            {
                "Algorithm": "DFS",
                "Path": " → ".join(map(str, dfs_path))
                if dfs_path else "No path",
                "Steps": len(dfs_path) - 1
                if dfs_path else None,
                "Cost": path_cost(graph, dfs_path)
                if dfs_path else None
            },
            {
                "Algorithm": "UCS",
                "Path": " → ".join(map(str, ucs_path))
                if ucs_path else "No path",
                "Steps": len(ucs_path) - 1
                if ucs_path else None,
                "Cost": ucs_cost
                if ucs_path else None
            },
            {
                "Algorithm": "Greedy",
                "Path": " → ".join(map(str, greedy_path))
                if greedy_path else "No path",
                "Steps": len(greedy_path) - 1
                if greedy_path else None,
                "Cost": path_cost(graph, greedy_path)
                if greedy_path else None
            },
            {
                "Algorithm": "A*",
                "Path": " → ".join(map(str, astar_path))
                if astar_path else "No path",
                "Steps": len(astar_path) - 1
                if astar_path else None,
                "Cost": astar_cost
                if astar_path else None
            }
        ]

        results_df = pd.DataFrame(results)

        results_df["Cost"] = results_df["Cost"].apply(
            lambda x: round(x, 4) if pd.notna(x) else None
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # Individual algorithm outputs
        tabs = st.tabs([
            "BFS",
            "DFS",
            "UCS",
            "Greedy",
            "A*"
        ])

        paths = [
            bfs_path,
            dfs_path,
            ucs_path,
            greedy_path,
            astar_path
        ]

        costs = [
            path_cost(graph, bfs_path) if bfs_path else None,
            path_cost(graph, dfs_path) if dfs_path else None,
            ucs_cost if ucs_path else None,
            path_cost(graph, greedy_path) if greedy_path else None,
            astar_cost if astar_path else None
        ]

        for tab, algorithm, path, cost in zip(
            tabs,
            ["BFS", "DFS", "UCS", "Greedy", "A*"],
            paths,
            costs
        ):
            with tab:
                if path:
                    st.success(
                        f"{algorithm} found a path with "
                        f"{len(path) - 1} steps."
                    )
                    st.write(" → ".join(map(str, path)))
                    st.metric("Path Cost", round(cost, 4))
                else:
                    st.error(
                        f"{algorithm} could not find a path."
                    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.header("Product Recommendations")

if graph_nodes:
    selected_product = st.selectbox(
        "Select a product",
        graph_nodes,
        key="recommendation_product"
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        1,
        20,
        5
    )

    recommendations = recommend(
        graph,
        selected_product,
        recommendation_count
    )

    if recommendations:
        recommendation_df = pd.DataFrame(
            [
                {
                    "Recommended Product": product,
                    "Edge Cost": round(cost, 4),
                    "Association Strength (Lift)": round(
                        1 / cost, 4
                    )
                }
                for product, cost in recommendations
            ]
        )

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            f"Recommendations for: {selected_product}"
        )

        for rank, (product, cost) in enumerate(
            recommendations,
            start=1
        ):
            st.write(
                f"**{rank}. {product}** — "
                f"Lift: {1 / cost:.3f}"
            )
    else:
        st.info(
            "No recommendations are available for this product."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.write("Smart Retail Intelligence System")
st.sidebar.write("Apriori + Graph Search")
