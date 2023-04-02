# import needed packages
import matplotlib.pyplot as plt
import networkx as nx
import sys

## Initial setup

# create our title
img = plt.imread(r"images/title.jpg")
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.imshow(img, extent=[-1.75, 2, -1.25, 1.5])
plt.show()

# our "space" background
img = plt.imread(r"images/space.jpg")
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.imshow(img, extent=[-1.75, 2, -1.25, 1.5])

# create the initial graph
edges = [['HOTH', 'NABOO'], ['HOTH', 'DAGOBAH'], ['MUSTAFAR', 'NABOO'],
         ['MUSTAFAR', 'TATOOINE'], ['DAGOBAH', 'NABOO'],
         ['DAGOBAH', 'TATOOINE'], ['NABOO', 'TATOOINE']]
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)

nx.draw(G,
        pos,
        edge_color='white',
        width=1,
        linewidths=1,
        node_size=300,
        node_color='gray',
        alpha=0.9,
        font_color="yellow",
        font_size=9,
        labels={node: node
                for node in G.nodes()})

nx.draw_networkx_edge_labels(G,
                             pos,
                             edge_labels={
                               ('HOTH', 'NABOO'): '3',
                               ('HOTH', 'DAGOBAH'): '2',
                               ('MUSTAFAR', 'NABOO'): '4',
                               ('MUSTAFAR', 'TATOOINE'): '1',
                               ('DAGOBAH', 'NABOO'): '7',
                               ('DAGOBAH', 'TATOOINE'): '5',
                               ('NABOO', 'TATOOINE'): '6'
                             },bbox=dict(alpha=0.4),
                             font_color='yellow')
plt.axis('off')
plt.show()


# adapted from 
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html

class Graph(object):

  def __init__(self, nodes, init_graph):
    self.nodes = nodes
    self.graph = self.construct_graph(nodes, init_graph)

  def construct_graph(self, nodes, init_graph):
    '''
        This method makes sure that the graph is symmetrical. In other words, 
        if there's a path from node A to B with a value V, 
        there needs to be a path from node B to node A with a value V.
        '''
    graph = {}
    for node in nodes:
      graph[node] = {}

    graph.update(init_graph)

    for node, edges in graph.items():
      for adjacent_node, value in edges.items():
        if graph[adjacent_node].get(node, False) == False:
          graph[adjacent_node][node] = value

    return graph

  def get_nodes(self):
    "Returns the nodes of the graph."
    return self.nodes

  def get_outgoing_edges(self, node):
    "Returns the neighbors of a node."
    connections = []
    for out_node in self.nodes:
      if self.graph[node].get(out_node, False) != False:
        connections.append(out_node)
    return connections

  def value(self, node1, node2):
    "Returns the value of an edge between two nodes."
    return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
  color_map = ['gray','gray','gray','gray','gray']
  
  unvisited_nodes = list(graph.get_nodes())

  # We'll use this dict to save the cost of visiting each node and update 
  # it as we move along the graph
  shortest_path = {}

  # We'll use this dict to save the shortest known path to a node found so far
  previous_nodes = {}

  # We'll use max_value to initialize the "infinity" value of the unvisited nodes
  max_value = sys.maxsize
  for node in unvisited_nodes:
    shortest_path[node] = max_value
  # However, we initialize the starting node's value with 0
  shortest_path[start_node] = 0

  node_values = shortest_path.copy()

  # Change the sys.maxsize value to a string that says "inf"
  for i in G.nodes():
      number = shortest_path[i]
      if (number == max_value):
          node_values[i] = i + ": Inf"
      else:
          node_values[i] = i + ": " + str(number)
  
  # The algorithm executes until we visit all nodes
  while unvisited_nodes:
    # The code block below finds the node with the lowest score
    current_min_node = None
    for node in unvisited_nodes:  # Iterate over the nodes and update the visited node
      if current_min_node == None:
        current_min_node = node
      elif shortest_path[node] < shortest_path[current_min_node]:
        current_min_node = node

    # The code block below retrieves the current node's neighbors and updates their distances
    neighbors = graph.get_outgoing_edges(current_min_node)

    print(current_min_node, "has been visited.")

    # After visiting its neighbors, we mark the node as "visited"
    unvisited_nodes.remove(current_min_node)  
    
    # Add a line to color the node green for "visited"
    color_map = []
    for node in nodes:
      if node in unvisited_nodes:
        color_map.append('gray')
      else:
        color_map.append('blue')
          
    # this janky block of code update our visited nodes with the color
    # gray for unvisited and blue for visited    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.imshow(img, extent=[-1.5, 1.75, -1.75, 1.75])

    nx.draw(
        G,
      pos,
      edge_color='white',
      width=1,
      linewidths=1,
      node_size=300,
      node_color=color_map,
      alpha=0.9,
      font_color="yellow",
      font_size=9,
      labels = node_values)

    nx.draw_networkx_edge_labels(G,
                                 pos,
                                 edge_labels={
                                   ('HOTH', 'NABOO'): '3',
                                   ('HOTH', 'DAGOBAH'): '2',
                                   ('MUSTAFAR', 'NABOO'): '4',
                                   ('MUSTAFAR', 'TATOOINE'): '1',
                                   ('DAGOBAH', 'NABOO'): '7',
                                   ('DAGOBAH', 'TATOOINE'): '5',
                                   ('NABOO', 'TATOOINE'): '6'
                                 },bbox=dict(alpha=0.4),
                                 font_color='yellow')

    plt.axis('off')
    plt.show()
    
    # this block of code shows the main logic behind the algorithm
    for neighbor in neighbors:
      print('\n')
      print('Edge:', current_min_node, '-', neighbor, 'has weight value:',
            graph.value(current_min_node, neighbor))
      tentative_value = shortest_path[current_min_node] + graph.value(
        current_min_node, neighbor)
      
      print(neighbor, 'has tentative value:', tentative_value, ". We compare this to:",
            shortest_path[neighbor])

      if tentative_value < shortest_path[neighbor]:
          
        print(tentative_value, "<",
                  shortest_path[neighbor], "so we accept it.")
        
        shortest_path[neighbor] = tentative_value
        # We also update the best path to the current node
        previous_nodes[neighbor] = current_min_node
        
        
        
        node_values = shortest_path.copy()
        for i in G.nodes():
            number = shortest_path[i]
            if (number == max_value):
                node_values[i] = i + ": Inf"
            else:
                node_values[i] = i + ": " + str(number)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.imshow(img, extent=[-1.5, 1.75, -1.75, 1.75])

        nx.draw(
          G,
        pos,
        edge_color='white',
        width=1,
        linewidths=1,
        node_size=300,
        node_color=color_map,
        alpha=0.9,
        font_color="yellow",
        font_size=9,
            labels = node_values)
            #labels={node: node
                  # for node in G.nodes()})
        nx.draw_networkx_edge_labels(G,
                                       pos,
                                       edge_labels={
                                         ('HOTH', 'NABOO'): '3',
                                         ('HOTH', 'DAGOBAH'): '2',
                                         ('MUSTAFAR', 'NABOO'): '4',
                                         ('MUSTAFAR', 'TATOOINE'): '1',
                                         ('DAGOBAH', 'NABOO'): '7',
                                         ('DAGOBAH', 'TATOOINE'): '5',
                                         ('NABOO', 'TATOOINE'): '6'
                                       },bbox=dict(alpha=0.4),
                                       font_color='yellow')

        plt.axis('off')
        plt.show()
        
      else:
        print(tentative_value, ">",
                  shortest_path[neighbor], "so we deny it.")
        
    

  return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
  path = []
  node = target_node

  while node != start_node:
    path.append(node)
    node = previous_nodes[node]

  # Add the start node manually
  path.append(start_node)

  print("We found the following best path with a value of {}.".format(
    shortest_path[target_node]))
  print(" -> ".join(reversed(path)))

# start using the Graph class
nodes = ["HOTH", "NABOO", "DAGOBAH", "MUSTAFAR", "TATOOINE"]

init_graph = {}
for node in nodes:
  init_graph[node] = {}

init_graph["HOTH"]["NABOO"] = 3
init_graph["HOTH"]["DAGOBAH"] = 2
init_graph["MUSTAFAR"]["NABOO"] = 4
init_graph["MUSTAFAR"]["TATOOINE"] = 1
init_graph["DAGOBAH"]["NABOO"] = 7
init_graph["DAGOBAH"]["TATOOINE"] = 5
init_graph["NABOO"]["TATOOINE"] = 6

graph = Graph(nodes, init_graph)

print('Our list of planets', nodes)

start = input("Where is Han and Chewy right now? ")
end = input("Where do they need to go? ")

previous_nodes, shortest_path = dijkstra_algorithm(graph=graph,
                                                   start_node=start.upper())

print_result(previous_nodes, shortest_path, start_node=start.upper(), target_node=end.upper())
