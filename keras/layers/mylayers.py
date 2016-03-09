from keras.layers import containers
from keras.layers.convolutional import Convolution2D

    
class SplitTensor(Layer):
    '''Repeat the input n times.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
    # Output shape
        4D tensor with shape:
        `(samples, selected_channels, rows, cols)`
    # Arguments
        axis: integer, axis of the selection.
        ratio_split: int, number of parts dividing the input
        id_split: int, number of the selected part
    '''
    def __init__(self, axis=1, ratio_split=1, id_split = 0, **kwargs):
        super(SplitTensor, self).__init__(**kwargs)
        self.axis=axis
        self.ratio_split = ratio_split
        self.id_split = id_split

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_shape = input_shape
        output_shape[self.axis] = output_shape / self.ratio_split
        return output_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        id_split = self.id_split
        ratio_split = self.ratio_split
        if axis == 0:
            output =  X[id_split*ratio_split:(id_split+1)*ratio_split,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*ratio_split:(id_split+1)*ratio_split, :, :]
        elif axis == 2:
            output = X[:,:,id_split*ratio_split:(id_split+1)*ratio_split,:]
        elif axis == 3:
            output == X[:,:,:,id_split*ratio_split:(id_split+1)*ratio_split]
        else:
            raise ValueError("This axis is not possible")
        
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'axis': self.axis,
                  'ratio_split': self.ratio_split,
                  'id_split': self.id_split}
        base_config = super(SplitTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def ConvGroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    layer = containers.Graph()
    layer.add_input(name='input')

    for i in range(n_group):
        graph.add_node(SplitTensor(axis=2,ratio_split=n_group,id_split=i),
                       name='split'+str(i),
                       input='input')
        graph.add_node(Convolution2D(nb_filter / n_group,
                                     nb_row,nb_col,**kwargs),
                       name='conv'+str(i),
                       input='split'+str(i))

    layer.add_output(name='output',
                     inputs=['conv'+str(i) for i in range(n_group)],
                     merge_mode='concat',
                     concat_axis=1)
    

    return layer
        
