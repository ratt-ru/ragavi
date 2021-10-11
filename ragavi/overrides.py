from collections import UserDict

class rdict(dict):
    """Add a function to set multiple keys' defaults if they don't already exist"""
    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
    
    def set_multiple_defaults(self, *args, **kwargs):
        """
        Initialise multiple default values for this dictionary
        new_defs: dict
            Contains defaults for various keys
        """
        if args:
            kwargs = args[0]
        for key, value in kwargs.items():
            self.data.setdefault(key, value)
        


def set_multiple_defaults(in_dict, defaults):
    for key, value in defaults.items():
        in_dict.setdefault(key, value)
    
    return in_dict
