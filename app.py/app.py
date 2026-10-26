import heapq
import math
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="GPS Navigation System", page_icon="🗺️", layout="wide")

# Sample locations and positions for drawing the educational map.
LOCATIONS = {
    "College": (0, 3),
    "Bus Stand": (2, 4),
    "Hospital": (5, 3),
    "Mall": (1, 1),
    "Railway Station": (3, 2),
    "Airport": (6, 0),
    "Park": (3, 0),
    "City Center": (0, 0),
}

# Each road is an undirected weighted edge. Weight = sample distance in kilometres.
ROADS = [
    ("College", "Bus Stand", 4),
    ("College", "Mall", 3),
    ("Bus Stand", "Hospital", 5),
    ("Bus Stand", "Mall", 2),
    ("Bus Stand", "Railway Station", 4),
    ("Hospital", "Railway Station", 3),
    ("Hospital", "Airport", 7),
    ("Mall", "Railway Station", 3),
    ("Mall", "City Center", 5),
    ("Mall", "Park", 6),
    ("Railway Station", "Airport", 4),
    ("Railway Station", "Park", 3),
    ("Airport", "Park", 5),
    ("Park", "City Center", 2),
]


def make_graph():
    graph = nx.Graph()
    for name, pos in LOCATIONS.items():
        graph.add_node(name, pos=pos)
    for u, v, distance in ROADS:
        graph.add_edge(u, v, weight=distance)
    return graph


def floyd_warshall(graph):
    """Floyd–Warshall: shortest distances between all pairs of nodes."""
    nodes = list(graph.nodes)
    dist = {u: {v: math.inf for v in nodes} for u in nodes}
    for u in nodes:
        dist[u][u] = 0
    for u, v, data in graph.edges(data=True):
        dist[u][v] = data["weight"]
        dist[v][u] = data["weight"]

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return nodes, dist


def prim_mst(graph):
    """Prim's algorithm: build a minimum spanning tree."""
    start = next(iter(graph.nodes))
    visited = {start}
    mst_edges = []
    total = 0

    while len(visited) < graph.number_of_nodes():
        candidate = None
        for u in visited:
            for v, data in graph[u].items():
                if v not in visited:
                    weight = data["weight"]
                    if candidate is None or weight < candidate[2]:
                        candidate = (u, v, weight)
        if candidate is None:
            break
        u, v, weight = candidate
        visited.add(v)
        mst_edges.append((u, v, weight))
        total += weight

    return mst_edges, total


def a_star(graph, start, goal):
    """A*: shortest route using road distance and a consistent scaled heuristic."""
    positions = nx.get_node_attributes(graph, "pos")

    # The coordinates are only illustrative. Scale the straight-line estimate
    # conservatively so it remains no larger than the sample road-cost scale.
    def heuristic(node):
        x1, y1 = positions[node]
        x2, y2 = positions[goal]
        return math.hypot(x1 - x2, y1 - y2) * 0.5

    open_heap = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {node: math.inf for node in graph.nodes}
    g_score[start] = 0
    closed = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[goal]

        closed.add(current)
        for neighbor, data in graph[current].items():
            tentative = g_score[current] + data["weight"]
            if tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (tentative + heuristic(neighbor), tentative, neighbor),
                )
    return None, math.inf


def draw_graph(graph, route=None, mst_edges=None):
    route = route or []
    mst_edges = mst_edges or []
    route_pairs = {frozenset((route[i], route[i + 1])) for i in range(len(route) - 1)}
    mst_pairs = {frozenset((u, v)) for u, v, _ in mst_edges}
    positions = nx.get_node_attributes(graph, "pos")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#f8fbff")
    # Draw roads one at a time so route/MST edges can be highlighted.
    for u, v, data in graph.edges(data=True):
        pair = frozenset((u, v))
        if pair in route_pairs:
            color, width = "#16803c", 4.5
        elif pair in mst_pairs:
            color, width = "#d97706", 4.0
        else:
            color, width = "#94a3b8", 2.0
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(u, v)], ax=ax,
            edge_color=color, width=width
        )
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.12,
            str(data["weight"]) + " km",
            fontsize=8, ha="center", color="#334155",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
        )

    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_size=1500,
        node_color="#2563eb", edgecolors="white", linewidths=2
    )
    nx.draw_networkx_labels(
        graph, positions, ax=ax, font_size=8, font_color="white",
        font_weight="bold"
    )
    ax.set_title("Sample Road Network (edge labels are distances in km)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


st.title("🗺️ GPS Navigation System")
st.caption("Data Structures project | Floyd–Warshall • Prim's MST • A* Pathfinding")
st.info("This is an educational simulation using sample locations and distances. It does not use live GPS or real road traffic.")

graph = make_graph()
left, right = st.columns([1, 2])

with left:
    st.subheader("Choose a route")
    start = st.selectbox("Starting location", list(LOCATIONS.keys()), index=0)
    destination = st.selectbox("Destination", list(LOCATIONS.keys()), index=5)
    algorithm = st.radio(
        "Choose algorithm to display",
        ["A* Route Finder", "Floyd–Warshall All-Pairs Distances", "Prim's Minimum Spanning Tree"],
    )
    run_button = st.button("Run algorithm", type="primary", use_container_width=True)

if "last_algorithm" not in st.session_state:
    st.session_state.last_algorithm = "A* Route Finder"
    st.session_state.last_start = start
    st.session_state.last_destination = destination
if run_button:
    st.session_state.last_algorithm = algorithm
    st.session_state.last_start = start
    st.session_state.last_destination = destination

active_algorithm = st.session_state.last_algorithm
active_start = st.session_state.last_start
active_destination = st.session_state.last_destination
route = []
mst_edges = []

with right:
    st.subheader("Road map")
    if active_algorithm == "A* Route Finder":
        route, distance = a_star(graph, active_start, active_destination)
        st.pyplot(draw_graph(graph, route=route), use_container_width=True)
    elif active_algorithm == "Floyd–Warshall All-Pairs Distances":
        nodes, distances = floyd_warshall(graph)
        st.pyplot(draw_graph(graph), use_container_width=True)
    else:
        mst_edges, total = prim_mst(graph)
        st.pyplot(draw_graph(graph, mst_edges=mst_edges), use_container_width=True)

st.divider()
st.subheader("Algorithm output")

if active_algorithm == "A* Route Finder":
    route, distance = a_star(graph, active_start, active_destination)
    c1, c2 = st.columns(2)
    c1.metric("Total distance", f"{distance:g} km")
    c2.metric("Locations visited in route", len(route))
    st.write("**Route:** " + " → ".join(route))
    st.write("**How it works:** A* uses `f(n) = g(n) + h(n)`, where `g(n)` is distance travelled and `h(n)` estimates the remaining distance.")
elif active_algorithm == "Floyd–Warshall All-Pairs Distances":
    nodes, distances = floyd_warshall(graph)
    st.write("The table gives the shortest distance between every pair of locations.")
    st.dataframe(
        {node: [distances[row][node] for row in nodes] for node in nodes},
        use_container_width=True,
    )
    st.caption("Rows are starting locations and columns are destination locations. Values are kilometres.")
else:
    mst_edges, total = prim_mst(graph)
    st.metric("Total length of selected MST roads", f"{total} km")
    st.write("**Selected roads:**")
    for u, v, weight in mst_edges:
        st.write(f"- {u} ↔ {v}: {weight} km")
    st.write("**How it works:** Prim's algorithm repeatedly chooses the cheapest road that connects the current network to a new location.")
    st.caption("An MST is useful for planning a low-cost connected road/network layout. It is not the same as the shortest route between two locations.")

with st.expander("Data Structures used"):
    st.markdown(
        "- **Graph:** locations are vertices and roads are weighted edges.\n"
        "- **Adjacency structure:** NetworkX stores each location's neighbours and road distances.\n"
        "- **Priority queue:** A* uses Python's `heapq` to process the most promising location.\n"
        "- **Distance matrix:** Floyd–Warshall stores best-known distances between every pair.\n"
        "- **Visited set and edge list:** Prim's algorithm builds the Minimum Spanning Tree."
    )
