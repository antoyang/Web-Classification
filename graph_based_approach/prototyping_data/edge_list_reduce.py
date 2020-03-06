import os


def create_reduced_files(type="train"):
    if type != "train" and type != "test":
        print("Invalid type")
        exit(-1)

    og_file = type + ".csv"
    reduced_file = "reduced_" + type + "2.csv"
    reduced_edge_file = "reduced_" + type + "_edgelist.txt"

    with open(og_file, 'r') as f:
        node_list = []
        for line in f:
            node_list.append(line.replace("\n", "").split(',')[0])

    node_set = set()

    if os.path.exists(reduced_edge_file):
        print("File", reduced_edge_file, "already exist")
        return

    with open(reduced_edge_file, 'w') as newlist:
        with open("edgelist.txt", 'r') as edgelist:
            for line in edgelist:
                node0, node1, weight = line.split(' ')
                #if node0 in node_list and node1 in node_list:
                #    newlist.write(line)
                #   node_set.add(node0)
                #   node_set.add(node1)
                if int(weight) < 4:
                    if node0 in node_list and not node0 in node_set:
                        pass
                    elif node1 in node_list and not node1 in node_set:
                        pass
                    else:
                        continue
                if node0 in node_list or node1 in node_list:
                    newlist.write(line)
                if node0 in node_list:
                    node_set.add(node0)
                if node1 in node_list:
                    node_set.add(node1)
    print(len(node_set))

    with open(reduced_file, 'w') as reduced:
        with open(og_file, 'r') as f:
            for line in f:
                nodes = line.replace("\n","").split(',')
                if nodes[0] in node_set:
                    reduced.write(line)
                    node_set.remove(nodes[0])


if __name__ == "__main__":
    create_reduced_files(type="train")
    create_reduced_files(type="test")
