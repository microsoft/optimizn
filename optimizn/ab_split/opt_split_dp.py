import numpy as np


# Taken from: https://www.geeksforgeeks.org/subset-sum-problem-dp-25/
# A Dynamic Programming solution for subset
# sum problem Returns true if there is a subset of
# set[] with sun equal to given sum


# Returns true if there is a subset of set[]
# with sum equal to given sum
def isSubsetSum(set, n, sum):
    # The value of subset[i][j] will be
    # true if there is a
    # subset of set[0..j-1] with sum equal to i
    subset = ([[False for i in range(sum + 1)]
               for i in range(n + 1)])

    # If sum is 0, then answer is true
    for i in range(n + 1):
        subset[i][0] = True

    # If sum is not 0 and set is empty,
    # then answer is false
    for i in range(1, sum + 1):
        subset[0][i] = False

    # Fill the subset table in bottom up manner
    for i in range(1, n + 1):
        for j in range(1, sum + 1):
            if j < set[i-1]:
                subset[i][j] = subset[i-1][j]
            if j >= set[i-1]:
                subset[i][j] = (subset[i-1][j] or
                                subset[i - 1][j-set[i-1]])

    return subset


# Driver code
if __name__ == '__main__':
    arr = [3, 34, 4, 12, 5, 2]
    sum = 9
    n = len(arr)
    if (isSubsetSum(set, n, sum)[n, sum]):
        print("Found a subset with given sum")
    else:
        print("No subset with given sum")
 
# This code is contributed by
# sahil shelangia.
