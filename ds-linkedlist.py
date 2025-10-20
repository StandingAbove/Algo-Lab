class ListNode:
    def __init__(self, val: int, next: ListNode):
        self.val = val
        self.next = next

    def linked_list_reversal(head: ListNode) -> ListNode:
        curr_node, prev_node = head, None
        while curr_node:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node

        return prev_node

    def remove_kth_last_node (head: ListNode, k: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        trailer = leader = dummy
        for _ in range(k):
            leader = leader.next

            if not leader:
                return head
        while leader.next:
            leader = leader.next
            trailer = trailer.next

        trailer.next = trailer.next.next
        return dummy.next


