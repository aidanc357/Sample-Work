import numpy as np

def get_distance(arr_1, arr_2):
    """
    
    Parameters
    ----------
    arr_1 : np.array (float)
        Coordinate values of the some point
    arr_2 : np.array (float)
        Coordinate values of another point (dim must match)

    Returns
    -------
    Euclidean distance between the two points (float)
    
    """
    if len(arr_1) != len(arr_2):
        raise Exception("Both arrays must be of the same dimension")
    new_arr = np.subtract(arr_1, arr_2)
    new_arr = np.square(new_arr)
    distance_squared = np.sum(new_arr)
    return(np.sqrt(distance_squared))

class kNN_classifier:
    """
    
    This class creates a k-Nearest-Neighbors classifier.
    
    Attributes
    ----------
    data : m*n np.array (float)
        Matrix containing data for m observations of n dimensions each
    no_neighbors : int
        Number of neighbors to considered when assigning class id
    classes : list (int)
        Class id's of original observations
        
    Methods
    -------
    get_distances
        Calculates the Euclidean distances between a point and those in the
        data
    choose_class
        Assigns a class id to a given observation based on previously
        calculated distances
    
    """
    
    def __init__(self, d, n, c):
        """
        
        Parameters
        ----------
        d : m*n np.array (float)
            Matrix containing data for m observations of n dimensions each
        n : int
            Number of neighbors to considered when assigning class id
        c : list (int)
            Class id's of original observations
        """
        if len(d) != len(c):
            raise Exception("The no. of entries must equal the no. of id's")
        elif n % 2 != 1:
            raise Exception("The no. of nearest neigbhors must be odd")
        self.data = d
        self.no_neighbors = n
        self.classes = c
    
    def get_distances(self, arr):
        """
        
        Parameters
        ----------
        arr : np.array (float)
            Data for a single observation with same number of attributes as
            the original matrix

        Returns
        -------
        distances : list (float)
            List of Euclidean distances between the array and all observations
            in the original matrix
        """
        distances = []
        for i in range(0, len(self.data)):
            distances += [get_distance(arr, self.data[i])]
        return(distances)
    
    def choose_class(self, distances):
        """

        Parameters
        ----------
        distances : list (float)
            List of Euclidean distances between the array and all observations
            in the original matrix

        Returns
        -------
        max_index: int
            Preicted class id for a given observation

        """
        sorted_indices = np.argsort(distances)[0:self.no_neighbors]
        best_classes = np.bincount(self.classes[sorted_indices])
        max_occurence = 0
        max_index = 0
        for i in range(0, len(best_classes)):
            if best_classes[i] > max_occurence:
                max_index = i
                max_occurence = best_classes[i]
        return(max_index)
        
    def choose_classes(self, new_data):
        """
        
        Parameters
        ----------
        new_data : m*n np.array (float)
            Matrix containing data for m observations of n dimensions each

        Returns
        -------
        class_ids : list (int)
            Predicted class id's for the the new data

        """
        class_ids = []
        for i in range(0, len(new_data)):
            distances = self.get_distances(new_data[i])
            class_ids += [self.choose_class(distances)]
        return(class_ids)
    
# Sample Test
    
data = np.array([[1, 0, 0],
                 [0, 0.5, 0],
                 [1, 1, 0],
                 [2, 3, 3],
                 [3, 2.5, 2.5],
                 [0, 0, 0.5],
                 [1, 2, 1],
                 [3, 3, 3.5],
                 [4, 5, 3],
                 [2, 1, 0],
                 [1, 2, 1.5],
                 [3, 4, 4.5],
                 [2, 5, 4]], dtype = object)

classes = np.array([1, 1, 1, 2, 2, 1, 1, 2, 3, 1, 2, 3, 3], dtype = int)

new_data = np.array([[1, 1, 0.5],
                     [0, 1, 1.5],
                     [3, 2, 3],
                     [1, 1.5, 2]], dtype = object)

classifier = kNN_classifier(data, 3, classes)
print(classifier.choose_classes(new_data))