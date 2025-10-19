# Hash sets: simpler form of hash maps
# store only the keys
from collections import defaultdict

from boltons.setutils import complement
from doc.source.conf import numpydoc_class_members_toctree
from patsy.mgcv_cubic_splines import test__map_cyclic_errors
from pygments.lexers.textfmts import TodotxtLexer
from statsmodels.tsa.arima.estimators.statespace import statespace


#Given an array of integers, return the indexes of any two numbers that add up to a target
#The order of indexes in the results doesn't matter

def pair_sum_unsorted_two_pass(nums: List[int], target: int) -> List[int]:

    num_map{}
    for i, num enumerate(nums):
        num_map[num] = i
    for i, num in enumerate(num):
        complement = target - num
        if complement in num_map and num_map[complement] != i
            return [i, num_map[complement]]
    return []

def pair_sum_unsorted_one_pass(nums: List[int], target: int) -> List[int]:
    hashmap = {}
    for i, x in enumerate(nums):
        if target - x in hashmap:
            return[hashmap[target - x], 1]
        hashmap[x] = i
    return[]

# given a partially completed 9x9 sudoku board, determine if the current state
# of the board adheres to the rules of the game
# each row and column must contain unique numbers between 1 and 9 or be empty (0)
# each of the nine 3x3 subgrids that compose the grid must contain unique numbers

def verify_sudoku_board(board: List[List[int]]) -> bool:
    row_sets = [set() for _ in range(9)]
    column_sets = [set() for _ in range(9)]
    subgrid_sets = [[set() for _ in range(3)] for _ in range(3)]
    for r in range(9):
        for c in range(9):
            num = board [r][c]
            if num == 0:
                continue
            if num in row_sets[r]:
                return False
            if num in column_sets[c]:
                return False
            if num in subgrid_sets[r // 3][c // 3]:
                return False
            row_sets[r].add(num)
            column_sets[c].add(num)
            subgrid_sets[r//3][c//3].add(num)
    return True

def zero_striping_hash_sets(matrix: List[List[int]]) -> None:
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set()
    #pass1: traverse through the matrix to identify r and c containing zeros
    for r in range(m):
        for c in range(n):
            if matrix[r][c] == 0:
                zero_rows.add(r)
                zero_cols.add(c)
    #pass 2: set any cell in matrix to zero
    for r in range(m):
        for c in range(n):
            if r in zero_rows or c in zero_cols:
                matrix[r][c] = 0
