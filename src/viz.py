# src/main.py
import os
import time
import osmnx as ox
from shapely.geometry import Point
from data_processing import load_osm_data, project_graph, get_convex_hull
from shortest_path_twoq import two_q
from shortest_path_dijkstra import dijkstra
from visualization import visualize_dual_route
from utils.helpers import get_osm_file_path, find_nearest_node, print_route_info
from comparison_text import save_algorithm_comparison  # Newly added import

def main():
    # Get the path to the OSM file
    osm_file = get_osm_file_path()

    # Load OSM data and create the graph
    muc_graph = load_osm_data(osm_file)

    # Project the graph to a suitable coordinate reference system (CRS)
    muc_graph_proj = project_graph(muc_graph)

    # Get nodes and edges from the projected graph
    nodes_proj, edges_proj = ox.graph_to_gdfs(muc_graph_proj, nodes=True, edges=True)

    # Calculate the convex hull and centroid
    convex_hull = get_convex_hull(edges_proj)
    centroid = convex_hull.centroid

    # Define source and target nodes
    source_point = Point(centroid.x, centroid.y)  # Centroid as the source node
    target_point = nodes_proj.loc[nodes_proj['x'] == nodes_proj['x'].min(), 'geometry'].values[0]  # Westernmost node

    # Find the nearest nodes for source and target points
    source_node = find_nearest_node(muc_graph_proj, source_point)
    target_node = find_nearest_node(muc_graph_proj, target_point)

    # Calculate the shortest path using the Two-Q algorithm and measure the time
    start_time = time.time()
    shortest_path_two_q = two_q(muc_graph_proj, source_node, target_node)
    two_q_time = time.time() - start_time

    # Calculate the shortest path using Dijkstra's algorithm and measure the time
    start_time = time.time()
    shortest_path_dijkstra = dijkstra(muc_graph_proj, source_node, target_node)
    dijkstra_time = time.time() - start_time

    # If no shortest path is found
    if not shortest_path_two_q or not shortest_path_dijkstra:
        print("Target node could not be reached!")
    else:
        # Print route information for both algorithms
        print_route_info(shortest_path_two_q, "Two-Q")
        print_route_info(shortest_path_dijkstra, "Dijkstra")

        # Save the algorithm comparison results
        save_algorithm_comparison(shortest_path_two_q, shortest_path_dijkstra, two_q_time, dijkstra_time, muc_graph_proj)

        # Get the nodes and edges for the routes found by both algorithms
        route_nodes_two_q = nodes_proj.loc[shortest_path_two_q]
        route_edges_two_q = edges_proj.loc[shortest_path_two_q]

        route_nodes_dijkstra = nodes_proj.loc[shortest_path_dijkstra]
        route_edges_dijkstra = edges_proj.loc[shortest_path_dijkstra]

        # Visualize the routes and save as an HTML file
        visualize_dual_route(
            route_edges_two_q, route_nodes_two_q,
            route_edges_dijkstra, route_nodes_dijkstra,
            edges_proj, nodes_proj,
            source_node, target_node,
            output_html="outputs/dual_path_visualization.html"
        )

if __name__ == "__main__":
    main()