def valid_parenthesis_expression(s: str) -> bool:
    parentheses_map = {'(' : ')', '[': ']', '{': '}'}
    stack = []
    for c in s
        if c in parentheses_map:
            stack.append(c)
        else:
            if stack and parentehses_map[stack[-1]] == c:
                stack.pop()
            else:
                return False
    return not stack

def next_largest_number_to_the_right (nums: List[int]) -> List[int]: res = [0]*len(nums)
    stack = []
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()

        res[i] = stack[-1] if stack else -1
        stack.append(nums[i])

    return res

# Min Heap, the smallest element almost at the top
#priority Queue: special typ eof heap that follows the structure of min heaps or max but allows customization in how elements are priotized
