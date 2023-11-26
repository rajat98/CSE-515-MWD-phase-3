from LSH import ApproximateNearestNeighborSearch


def driver():
    # l = int(input("Please select numer of Layers, L\n"))
    # h = int(input("Please number of hashes per layer, h\n"))
    l = 3
    h = 3
    ann = ApproximateNearestNeighborSearch(l, h)
    ann.train()
    ann.describe()


if __name__ == "__main__":
    driver()
