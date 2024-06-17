import gc


def trace(func):
    def wrapper(*args, **kwargs):
        print(f'{func.__name__} start, obj len - {len(gc.get_objects())}')
        ret = func(*args, **kwargs)
        gc.collect()
        print(f'{func.__name__} end, obj len - {len(gc.get_objects())}')
        return ret
    return wrapper
