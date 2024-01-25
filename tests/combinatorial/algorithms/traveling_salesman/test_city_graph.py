from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from copy import deepcopy


def test_city_graph_equality():
    # check that city graphs are equal
    cg1 = CityGraph(num_cities=100)
    cg2 = deepcopy(cg1)
    assert cg1 == cg2, 'City graphs are supposed to be equal. '\
        + f'\nCity graph 1: {cg1}.\nCity graph 2: {cg2}'

    # check that city graphs are not equal
    cg2 = CityGraph(num_cities=50)
    assert cg1 != cg2, 'City graphs are not supposed to be equal. '\
        + f'\nCity graph 1: {cg1}.\nCity graph 2: {cg2}'

    # check that city graphs are not equal
    cg2 = CityGraph(num_cities=100)
    assert cg1 != cg2, 'City graphs are not supposed to be equal. '\
        + f'\nCity graph 1: {cg1}.\nCity graph 2: {cg2}'
