# Adapted from https://stackoverflow.com/a/32107024 - Accessed 02/05/2020
class ParameterMap(dict):
    """
    Example:
    m = ParameterMap({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(ParameterMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    
    def set_with_cast(self, key, value):
        if self[key] is None:
            return False
        else:
            t = type(self[key])
            if t is bool:
                val = value == 'True'
            elif t is int:
                val = int(value)
            elif t is float:
                val = float(value)
            else:
                val = str(value)
            self[key] = val
            return True

    def __setitem__(self, key, value):
        super(ParameterMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ParameterMap, self).__delitem__(key)
        del self.__dict__[key]

    def clone(self):
        new_map = ParameterMap()
        for k in self.keys():
            new_map[k] = self[k]
        return new_map

    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            arg_split = line.split(': ')
            assert len(arg_split) == 2
            k = arg_split[0].strip()
            v = arg_split[1].strip()
            if not self.set_with_cast(k, v):
                raise AttributeError('Unknown parameter [{}]'.format(k))

    def __str__(self):
        rep = ''
        for key in self:
            rep += '{}: {}\n'.format(key, self[key])
        return rep


def override_arguments(args, additional_args):
    for arg in additional_args:
        assert arg[:2] == '--'
        arg_split = arg.split('=')
        assert len(arg_split) == 2
        k = arg_split[0][2:]
        v = arg_split[1]
        if args.set_with_cast(k, v):
            print('Overriding argument [{}] with [{}]'.format(k, v))
        else:
            raise AttributeError('Unknown parameter [{}]'.format(k))
