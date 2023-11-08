def createTestSet(ListOfData):
    #Create test set
    Y_test_1 = []
    # Use the training data
    Y_test_1 += ["Flat"] * len(features_solar_1_flat_test[ListOfData[0]])
    Y_test_1 += ["Vertical"] * len(features_solar_1_vertical_test[ListOfData[0]])
    # X_train_up_1 = np.column_stack((features_solar_1_up[x_data],features_solar_1_up[y_data], features_solar_1_up[z_data]))
    # X_train_down_1 = np.column_stack((features_solar_1_down[x_data], features_solar_1_down[y_data], features_solar_1_down[z_data]))
    # X_train_up_down_1 = np.column_stack((features_solar_1_up_down[x_data],features_solar_1_up_down[y_data], features_solar_1_up_down[z_data]))
    # X_train_down_up_1 = np.column_stack((features_solar_1_down_up[x_data], features_solar_1_down_up[y_data], features_solar_1_down_up[z_data]))

    Y_test_1 += ["Flat Inverse"] * len(features_solar_1_flat_inverse_test[ListOfData[0]])
    Y_test_1 +=  ["Vertical Inverse"] * len(features_solar_1_vertical_inverse_test[ListOfData[0]])
    # Create empty columns for columnStack
    X_test_flat_inverse_1 = np.array([features_solar_1_flat_inverse_test[key] for key in ListOfData]).T
    X_test_vertical_1 = np.array([features_solar_1_vertical_test[key] for key in ListOfData]).T
    X_test_flat_1 = np.array([features_solar_1_flat_test[key] for key in ListOfData]).T
    X_test_vertical_inverse_1 = np.array([features_solar_1_vertical_inverse_test[key] for key in ListOfData]).T
    
    X_test_1 = np.concatenate((X_test_flat_1, X_test_vertical_1, X_test_flat_inverse_1, X_test_vertical_inverse_1), axis=0)
    
    return X_test_1, Y_test_1