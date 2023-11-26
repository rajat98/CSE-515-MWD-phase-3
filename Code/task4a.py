from LSH import ApproximateNearestNeighborSearch


def driver():
    # l = int(input("Please select numer of Layers, L\n"))
    # h = int(input("Please number of hashes per layer, h\n"))
    l = 10
    h = 10
    ann = ApproximateNearestNeighborSearch(l, h)
    ann.train()
    ann.describe()


if __name__ == "__main__":
    driver()
