# clarify the problem
#   restate the problem in your own words
# design your algo
#     start with brute, more optimal, consider multiple possible solutions
from requests.packages import target


# Two Pointer

#compare (nums[i], nums[j]) and then make decisiono

#pair sum - sorted
def pair_sum_sorted_brute_force(nums: List[int], target: int) -> List[int]:

    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def pair_sum_sorted(nums: List[int], target: int) -> List[int]:
    left , right = 0, len(nums) - 1

    while left < right:
        sum = nums[left] + nums[right]
        if(sum < target):
            left += 1
        elif (sum == target):
            return[left, right]
        else:
            right -= 1

    return []

#triplet sum
def triplet_sum_brute_force(nums: List[int] -> List[List[int]]):
    triplets = []
    nums.sort()
    for i in range(len(nums)):
        if nums[i]>0:
            break
        if i > 0 and nums[i] = nums[i - 1]:
            continue
        pairs = pair_sum_sorted_all_pairs(nums, i + 1, -nums[i])
        for pair in pairs:
            triplets.append([nums[i] + pair])
    return triplets

def pair_sum_sorted_all_pairs(nums: List[int], start: int, target: int) -> List[int]:

    pairs = []
    left, right = start, len(nums) - 1
    while left < right:
        sum = nums[left] + nums[right]
        if sum == target:
            pairs.append([nums[left, nums[right]])
            left += 1
            while left < right and nums[left] == nums[left - 1]:
                left += 1
        elif sum < target:
            left += 1
        else:
            right -= 1

    return pairs

def larget_container(heights: List[int]) -> int:
    max_water = 0
    left, right = 0, len(heights) - 1
    while (left < right ):
        water = min(heights[left], heights[right]) * (right - left)
        max_water = max(max_water, water)
        if(heights[left] < heights[right]):
            left += 1
        elif(heights[left] > heights [right]):
            right -= 1

        else:
            left += 1
            right -= 1

    return max_water

