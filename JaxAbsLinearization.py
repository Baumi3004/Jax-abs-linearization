import jax
import jax.numpy as jnp
from jax import core
from jax._src.core import ClosedJaxpr, Literal
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
        if self.dummy_var.size >= 1: 
            a = ders[0][0]
            b = ders[0][1]
            Z = ders[1][0]
            L = ders[1][1]
            y = vals[0]
            z = vals[1]
        else: 
            b, Z, L, z = None, None, None, None
            y, a = vals, ders[0]
        return y, z, a, b, Z, L

    def modify_jaxpr(self):
        closedjaxpr = self.jaxpr
        jaxpr = closedjaxpr.jaxpr
        consts = closedjaxpr.consts
        gensym = core.gensym()
        new_eqns = jaxpr.eqns.copy()
        dummy_shapes = []
        new_invars = [invar for invar in jaxpr.invars]
        new_outvars = [outvar for outvar in jaxpr.outvars]
        i = 0
        while i < len(new_eqns):
            if new_eqns[i].primitive.name == 'abs':
                i = replace_abs(new_eqns, i, new_invars, new_outvars, dummy_shapes)
            if new_eqns[i].primitive.name == 'max' or new_eqns[i].primitive.name == 'min':
                i = replace_maxmin(new_eqns, i)
            #if new_eqns[i].primitive.name == 'reduce_max' or new_eqns[i].primitive.name == "reduce_min": 
            #   i = replace_maxmin_red(new_eqns, i)
            i += 1
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
    if len(input) > 1:
        return input[0],jnp.concatenate([jnp.atleast_1d(jnp.ravel(var)) for var in input[1:]])
    return input[0]



def flatten_and_concat(xs):
    return jnp.concatenate([jnp.ravel(x) for x in xs])
    
def replace_abs(eqns, i, invars, outvars, dummy_shapes):
    gensym = core.gensym()
    eqn = eqns[i]
    assert eqn.primitive.name == 'abs'
    del eqns[i]
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
    eqns.insert(i,stop_eqn)
    #compute abs using additional variable
    abs_eqn = core.new_jaxpr_eqn(invars=[stopvar], outvars=[absvar], primitive=jax.lax.abs_p, params=dict(eqn.params), effects=[])
    eqns.insert(i+1, abs_eqn)
    # add dummy input to abs output to obtain same dependencies as abs variale (but you can differentiate wrt to this dummy)
    # make sure the dummy variable is 0 when evaluating the derivative!!
    add_eqn = core.new_jaxpr_eqn(invars=[absvar, dummy_invar], outvars=[outvar], primitive=jax.lax.add_p, params=dict(eqn.params), effects=[])
    eqns.insert(i+2, add_eqn)
    # register inputs of abses as outputs of the function (switching eq)
    outvars.append(invar)
    invars.append(dummy_invar)
    return i + 2

def replace_maxmin(eqns, i):
    gensym = core.gensym()
    eqn = eqns[i]
    primname = eqn.primitive.name
    assert primname == 'max' or primname == 'min'
    del eqns[i]
    #change max or min(a,b) to 0.5(a+b +-|a-b|)
    invars = eqn.invars
    outvar = eqn.outvars[0] 
    diffvar = gensym(outvar.aval)
    absvar = gensym(outvar.aval)
    sumvar = gensym(outvar.aval)
    combvar = gensym(outvar.aval)

    # compute a+b
    sum_eqn = core.new_jaxpr_eqn(invars=invars, outvars=[sumvar], primitive=jax.lax.add_p, params={}, effects=[])
    eqns.insert(i+0,sum_eqn)
    # compute argument of abs 
    diff_eqn = core.new_jaxpr_eqn(invars=invars, outvars=[diffvar], primitive=jax.lax.sub_p, params={}, effects=[])
    eqns.insert(i+1,diff_eqn)
    # compute abs
    abs_eqn = core.new_jaxpr_eqn(invars=[diffvar], outvars=[absvar], primitive=jax.lax.abs_p, params={}, effects=[])
    eqns.insert(i+2,abs_eqn)
    # compute + or - depeneding on max or min
    combprim = jax.lax.add_p if primname == 'max' else jax.lax.sub_p
    comb_eqn = core.new_jaxpr_eqn(invars=[sumvar, absvar], outvars=[combvar], primitive=combprim, params={}, effects=[])
    eqns.insert(i+3,comb_eqn)
    #scale by 0.5
    scale_eqn = core.new_jaxpr_eqn(invars=[combvar, Literal(0.5, aval=combvar.aval)], outvars=[outvar], primitive=jax.lax.mul_p, params={}, effects=[])
    eqns.insert(i+4,scale_eqn)
       
    return i


def replace_maxmin_red(eqns, i):
    gensym = core.gensym()
    eqn = eqns[i]
    primname = eqn.primitive.name
    assert primname == 'reduce_max' or primname == 'reduce_min'
    #del eqns[i]
    num_epements = math.prod([eqn.invars[0].aval.shape[ax] for ax in eqn.params['axes']])
    from IPython import embed; embed()
    return i
    