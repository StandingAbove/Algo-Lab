def jump_to_end(nums: List[int]) -> bool:
    destination = len(nums) - 1
    #traverse the array in reverse to see if the destination can
    #be reched by earlier indexes.
    for i in range(len(nums) -1 , - 1, -1):
        if i + nums[i] >= destination:
            destination = i
    return desintation == 0