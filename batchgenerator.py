
__author__ = "cheayoung jung <che9992@gmail.com>"
__version__ = "2018-06-27"


class BatchGenerator():
    where = 0
    
    
        '''
    usage
    
    var = BatchGenerator(xdata, ydata, batch_size = 100)
    var.x 
    var.y
    var.next()
    
    '''
    
    def __init__(self, x, y, batch_size, one_hot = False, nb_classes = 0):
        self.nb_classes = nb_classes
        self.one_hot = one_hot
        self.x_ = x
        self.y_ = y
        self.batch_size = batch_size
        
        self.total_batch = int(len(x) / batch_size)
        self.x = self.x_[:batch_size,]
        self.y = self.y_[:batch_size,]
        self.where = batch_size
        
        if self.one_hot :
            self.set_one_hot()

    def next_batch(self):
        if self.where + self.batch_size > len(self.x_) :
            left = len(self.x_) - self.where
            self.x = self.x_[len(self.x_) - left:]
            self.y = self.y_[len(self.y_) - left:]
            self.where = 0
            
            if self.one_hot:
                self.set_one_hot()
            
            
        else:
            self.x = self.x_[self.where:self.where+self.batch_size,]
            self.y = self.y_[self.where:self.where+self.batch_size,]
            self.where += self.batch_size
        
            if self.one_hot:
                self.set_one_hot()
        
    def set_one_hot(self):
        one_hot = np.array(self.y).reshape(-1).astype(np.int)
        self.y = np.eye(self.nb_classes)[one_hot]
