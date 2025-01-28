class State:
    def __init__(self, id: int):
        self.pre = None
        self.post = None
        self.id = id

    @property
    def pre(self):
        return self._pre
    
    @pre.setter
    def pre(self, value):
        self._pre = value

    @property
    def post(self):
        return self._post
    
    @post.setter
    def post(self, value):
        self._post = value

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        self._id = value