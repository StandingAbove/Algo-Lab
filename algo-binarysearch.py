def find_the_insertion_index(nums: List[int], target: int) -> int:
    left, right = 0, len(nums)
    while left < right:
        mid = (right + left) // 2
        if nums[mid] >= target
            right = mid
        else:
            left = mid + 1
    return left