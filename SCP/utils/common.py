def searchIndicesOfClass(searched_class, labels, n=0, initial_pos=0):
    """
    Function that outputs a list with all the indices of the searched_class in the labelsList
    If n is provided, only the first n coincidences are outputted
    If initial_pos is provided, the search starts by this position.
    searched_class, n, initial_pos -> integer
    labels -> array
    """
    indices = []
    if n == 0:
        # Case of searching for all the array
        for index, labels in enumerate(labels[initial_pos:]):
            if labels == searched_class:
                indices.append(initial_pos + index)
    else:
        # Case of searching only n number of indices
        i = 0
        for index, labels in enumerate(labels[initial_pos:]):
            if i >= n:
                break
            if labels == searched_class:
                indices.append(initial_pos + index)
                i += 1
    return indices
