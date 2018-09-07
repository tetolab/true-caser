def batch_from_generator(generator_func, number=1):
    batched = []
    for item in generator_func():
        batched.append(item)
        if len(batched) == number:
            yield batched
            batched = []
    yield batched

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]