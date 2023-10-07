class DNode:
    def __init__(self, elem, prev=None, next=None) -> None:
        self.data = elem
        self.llink = prev
        self.rlink = next

class DoublyLinkedDeque:
    def __init__(self) -> None:
        self.front: DNode = None
        self.rear: DNode = None
    
    def isEmpty(self) -> bool:
        return self.front == None

    def getNode(self, pos: int) -> DNode:
        if pos < 0: return None
        node = self.front
        while pos > 0 and node != None:
            node = node.rlink
            pos -= 1
        return node

    def insert(self, pos:int, item):
        before = self.getNode(pos-1)
        if (self.isEmpty()):
            self.front = self.rear = DNode(item)
        elif before == None:
            node = DNode(item, None, self.front)
            self.front.llink = node
            self.front = node
        elif before == self.rear:
            node = DNode(item, self.rear, None)
            self.rear.rlink = node
            self.rear = node
        else:
            node = DNode(item, before, before.rlink)
            before.rlink.llink = node
            before.rlink = node

s = DoublyLinkedDeque()

print(s.isEmpty())

print(s.insert(0, "밥"))
print(s.isEmpty())
print(s.getNode(0).data)