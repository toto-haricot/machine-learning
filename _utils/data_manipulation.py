def divide_by_feature(X, feature, threshold):
    """This function will split a numpy array with on one side the 
    records greater the given threshold for the given feature. 
    """
    split_fct = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_fct = lambda x: x[feature] >= threshold
    else:
        split_fct = lambda x: x[feature] == threshold

    X_left = np.array([xi for xi in X if split_fct(xi)])
    X_right = np.array([xi for xi in X if not split_fct(xi)])

    return np.array([X_left, X_right])