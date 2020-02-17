import os


def create_reduced_files(type="train"):
    if type != "train" and type != "test":
        print("Invalid type")
        exit(-1)

    og_file = type + ".csv"
    reduced_file = "reduced_" + type + ".csv"
    reduced_edge_file = "reduced_" + type + "_edgelist.txt"

    with open(og_file, 'r') as f:
        node_list = []
        for line in f:
            node_list.append(line.replace("\n", "").split(',')[0])

    node_set = set()

    if os.path.exists(reduced_edge_file):
        print("File" + reduced_edge_file + "already exist")
        return

    with open(reduced_edge_file, 'w') as newlist:
        with open("edgelist.txt", 'r') as edgelist:
            for line in edgelist:
                nodes = line.split(' ')
                if nodes[0] in node_list and nodes[1] in node_list:
                    newlist.write(line)
                    node_set.add(nodes[0])
                    node_set.add(nodes[1])
                elif nodes[0] in node_list:
                    newlist.write(line)
                    node_set.add(nodes[0])
                elif nodes[1] in node_list:
                    newlist.write(line)
                    node_set.add(nodes[1])

    print(len(node_set))

    with open(reduced_file, 'w') as trainfile:
        with open(og_file, 'r') as f:
            for line in f:
                nodes = line.split(',')
                if nodes[0] in node_set:
                    trainfile.write(line)
                    node_set.remove(nodes[0])


if __name__ == "__main__":
    create_reduced_files(type="train")
    create_reduced_files(type="test")