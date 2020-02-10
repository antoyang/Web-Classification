import os

with open("train.csv", 'r') as f:
    train_nodes = []
    for line in f:
        train_nodes.append(line.split(',')[0])

edge_list = []
node_set = set()

if os.path.exists('reduced_edgelist.txt'):
    print("Files already exist")
    exit(0)

with open('reduced_edgelist.txt', 'w') as newlist:
    with open("edgelist.txt", 'r') as edgelist:
        for line in edgelist:
            nodes = line.split(' ')
            if nodes[0] in train_nodes and nodes[1] in train_nodes:
                newlist.write(line)
                node_set.add(nodes[0])
                node_set.add(nodes[1])

print(len(node_set))

with open('reduced_train.csv', 'w') as trainfile:
    with open("train.csv", 'r') as f:
        for line in f:
            nodes = line.split(',')
            if nodes[0] in node_set:
               trainfile.write(line)
               node_set.remove(nodes[0])