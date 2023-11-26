from LSH import ApproximateNearestNeighborSearch


# driver function for searching t nearest neighbor using LSH index created in task 4a
def driver():
    # input image id or path
    input_image_id_or_path = input("Please select an image id or image path\n")
    # the parameter t
    t = int(input("Please select t to find t similar images\n"))
    ann = ApproximateNearestNeighborSearch()
    ann.find_t_nearest_neighbor(input_image_id_or_path, t)


if __name__ == "__main__":
    driver()
