"""Dataset base class

"""
class DataSet(object):
    """Base class

    """
    def __init__(self, common_params, dataset_params):
        """

        :param common_params: A params dict
        :param dataset_params: A params dict
        """
        raise NotImplementedError

    def batch(self):
        """Get batch

        :return: batch of data
        """
        raise NotImplementedError