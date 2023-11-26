from LSH import ApproximateNearestNeighborSearch


# driver for creating index
def driver():
    # number of layers
    l = int(input("Please select number of Layers, L\n"))
    # number of hashes per layer
    h = int(input("Please number of hashes per layer, h\n"))
    ann = ApproximateNearestNeighborSearch(l, h)
    ann.train()
    # ann.describe()


if __name__ == "__main__":
    driver()
