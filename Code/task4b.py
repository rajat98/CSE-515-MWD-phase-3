from LSH import ApproximateNearestNeighborSearch


def driver():
    # input_image_id_or_path = input("Please select an image id or image path\n")
    # t = int(input("Please select t to find t similar images\n"))
    t = 10
    ann = ApproximateNearestNeighborSearch()
    for id in [1, 881, 2501, 5123, 8675]:
        input_image_id_or_path = str(id)
        ann.find_t_nearest_neighbor(input_image_id_or_path, t)
    # ann.describe()


if __name__ == "__main__":
    driver()
