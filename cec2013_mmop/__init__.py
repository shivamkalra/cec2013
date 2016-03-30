from cec2013_mmop import _benchmark as b


def get_benchmark(fno):
    return b._cec2013Benchmark_suite[fno - 1]
