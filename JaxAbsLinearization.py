import jax
import jax.numpy as jnp
from jax import core
from jax._src.core import ClosedJaxpr
from jax._src.lax.lax import add_p
import math




class AbsLinearForm():
    def __init__(self, f, x0):
        self.f = f
        self.x0 = x0
        # full flatten x 
        if isinstance(self.x0, (list, tuple)):
            self.xlist, self.treedef = jax.tree_util.tree_flatten(self.x0)
            self.xshapes = [x.shape for x in self.xlist]
            self.x = flatten_and_concat(self.xlist)
            self.jaxpr = jax.make_jaxpr(lambda x: f(jax.tree_util.tree_unflatten(self.treedef, split_by_shapes(x, self.xshapes))))(self.x)
        else:
            self.xshapes = [self.x0.shape]
            self.x = self.x0.flatten()
            self.jaxpr = jax.make_jaxpr(lambda x: f(x.reshape(self.xshapes[0])))(self.x)
        self.abslin_jaxpr, self.dummy_shapes = self.modify_jaxpr()
        #TODO figure out dtypes
        self.dummy_var = jnp.zeros(sum([math.prod(shape) for shape in self.dummy_shapes]), dtype=self.x.dtype)
        self.obj_and_switching = jax.jit(lambda x,z: concat_except_first(jax.core.eval_jaxpr(self.abslin_jaxpr.jaxpr, self.abslin_jaxpr.consts, x, *split_by_shapes(z, self.dummy_shapes)))) 
        self.obj_and_switching_jac = jax.jit(jax.jacobian(self.obj_and_switching, argnums=(0,1)))    
    
    def __call__(self, xin):
        
        if isinstance(xin, (list, tuple)):
            self.xlist, self.treedef = jax.tree_util.tree_flatten(xin)
            x = flatten_and_concat(self.xlist)
        else:
            x = xin.flatten()
        ders = self.obj_and_switching_jac( x, self.dummy_var)
        vals = self.obj_and_switching(x, self.dummy_var)  
        a = ders[0][0]
        b = ders[0][1]
        Z = ders[1][0]
        L = ders[1][1]
        y = vals[0]
        z = vals[1]
        return y, z, a, b, Z, L

    def modify_jaxpr(self):
        closedjaxpr = self.jaxpr
        jaxpr = closedjaxpr.jaxpr
        consts = closedjaxpr.consts
        gensym = core.gensym()
        new_eqns = []
        dummy_shapes = []
        new_invars = [invar for invar in jaxpr.invars]
        new_outvars = [outvar for outvar in jaxpr.outvars]
        for eqn in self.jaxpr.jaxpr.eqns:
            if eqn.primitive.name == 'abs':
                invar = eqn.invars[0]
                outvar = eqn.outvars[0]
                stopvar = gensym(outvar.aval)
                absvar = gensym(outvar.aval)
                dummy_invar = gensym(outvar.aval)
                # jax will require dummy inputs for these new input variables
                # dummy_inputs.append(jnp.zeros(outvar.aval.shape, dtype=outvar.aval.dtype))
                dummy_shapes.append(outvar.aval.shape)
                # prevent differentiation through abses by injecting stop_gradient. original outvar will be used later by other functions
                stop_eqn = core.new_jaxpr_eqn(invars=[invar], outvars=[stopvar], primitive=jax.lax.stop_gradient_p, params={}, effects=[])
                new_eqns.append(stop_eqn)
                #compute abs using additional variable
                abs_eqn = core.new_jaxpr_eqn(invars=[stopvar], outvars=[absvar], primitive=eqn.primitive, params=dict(eqn.params), effects=[])
                new_eqns.append(abs_eqn)
                # add dummy input to abs output to obtain same dependencies as abs variale (but you can differentiate wrt to this dummy)
                # make sure the dummy variable is 0 when evaluating the derivative!!
                add_eqn = core.new_jaxpr_eqn(invars=[absvar, dummy_invar], outvars=[outvar], primitive=add_p, params=dict(eqn.params), effects=[])
                new_eqns.append(add_eqn)
                # register inputs of abses as outputs of the function (switching eq)
                new_outvars.append(invar)
                new_invars.append(dummy_invar)
            else:
                new_eqns.append(eqn)
            
        
            JaxprClass = type(jaxpr)              
            new_jaxpr = JaxprClass(invars=new_invars, outvars=new_outvars, constvars=jaxpr.constvars, eqns=new_eqns)

        return ClosedJaxpr(new_jaxpr, consts), dummy_shapes
    
def split_by_shapes(big, shapes):
    out = []
    idx = 0
    for shape in shapes:
        out.append(big[idx:idx+math.prod(shape)].reshape(shape))
        idx += math.prod(shape)
    return out

def concat_except_first(input):
    return input[0],jnp.concatenate([jnp.atleast_1d(jnp.ravel(var)) for var in input[1:]])


def flatten_and_concat(xs):
    return jnp.concatenate([jnp.ravel(x) for x in xs])

    