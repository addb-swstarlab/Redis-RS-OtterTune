class BaseType(object):
    TYPE_NAMES = {}

    @classmethod
    def choices(cls):
        return list(cls.TYPE_NAMES.items())

    @classmethod
    def name(cls, ctype):
        return cls.TYPE_NAMES[ctype]

    @classmethod
    def type(cls, name):
        return [k for k, v in list(cls.TYPE_NAMES.items()) if
                v.lower() == name.lower()][0]


class VarType(BaseType):
    STRING = 1
    INTEGER = 2
    REAL = 3
    BOOL = 4
    ENUM = 5
    TIMESTAMP = 6

    TYPE_NAMES = {
        STRING: 'STRING',
        INTEGER: 'INTEGER',
        REAL: 'REAL',
        BOOL: 'BOOL',
        ENUM: 'ENUM',
        TIMESTAMP: 'TIMESTAMP',
    }