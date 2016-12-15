from rule import IndexRule
import theano
from theano import tensor as T

class IndexMul(theano.Op):
    """
    An arbitrarily generalized Fibbonacci sequence
    """
    __props__ = ("rule",)

    def __init__(self, rule_path , vocab, sen_len, ltp_model_path):
        self.rule=IndexRule(rule_path=rule_path, vocab=vocab, sen_len=sen_len, ltp_model_path=ltp_model_path)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self,
            inputs=[x],
            outputs=[x.type()])
        # using x_.type() is dangerous, it copies x's broadcasting behaviour

    def perform(self, node, inputs, output_storage):
        '''
        Op(index) return a fvector
        '''
        x = inputs[0]
        y=output_storage[0] 
        y[0]=self.rule.transform(x)

    
    def infer_shape(self, node, i0_shapes):
        return i0_shapes
    
    def grad(self, inputs, output_grads):
        return [T.zeros_like(inputs[0],dtype='float32')]
    ''' 
    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        y, = onames
        fail = sub['fail']
        return """
Py_XDECREF(%(y)s);
%(y)s = (PyArrayObject*)PyArray_FromArray(
            %(x)s, 0, NPY_ARRAY_ENSURECOPY);
if (!%(y)s)
  %(fail)s;
{//New scope needed to make compilation work
  dtype_%(y)s * y = (dtype_%(y)s*)PyArray_DATA(%(y)s);
  dtype_%(x)s * x = (dtype_%(x)s*)PyArray_DATA(%(x)s);
  for (int i = 2; i < PyArray_DIMS(%(x)s)[0]; ++i)
    y[i] = y[i-1]*y[i-2] + x[i];
}
        """ % locals()
    '''
    '''
    def c_code_cache_version(self):
        return (1,)
    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    '''
