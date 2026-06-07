"""Custom exception for when a 'predict' method has been called with 'fit' being called first."""


class ModelNotFittedError(Exception):
    pass