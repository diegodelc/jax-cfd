# # Copyright 2021 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Examples of defining equations."""
# import functools
# from typing import Callable, Optional

# import jax
# import jax.numpy as jnp
# import numpy as np

# from jax_cfd.base import advection
# from jax_cfd.base import diffusion
# from jax_cfd.base import grids
# from jax_cfd.base import pressure
# from jax_cfd.base import time_stepping
# from jax_cfd.base import boundaries
# import tree_math

# # from jax_cfd.ml.diego_train_functions import reshapeData

# # Specifying the full signatures of Callable would get somewhat onerous
# # pylint: disable=g-bare-generic

# GridArray = grids.GridArray
# GridArrayVector = grids.GridArrayVector
# GridVariable = grids.GridVariable
# GridVariableVector = grids.GridVariableVector
# ConvectFn = Callable[[GridVariableVector], GridArrayVector]
# DiffuseFn = Callable[[GridVariable, float], GridArray]
# ForcingFn = Callable[[GridVariableVector], GridArrayVector]



# def reshapeData(mydata,mygrid,
#                 offsets=[(1.0,0.5),
#                          (0.5,1.0)],
#                 bcs = [boundaries.channel_flow_boundary_conditions(ndim=2),
#                       boundaries.channel_flow_boundary_conditions(ndim=2)]):
#     """defaults to channel flow settings"""
#     return (grids.GridArray(data = mydata[:,:,0],offset=offsets[0],grid=mygrid),
#           grids.GridArray(data = mydata[:,:,1],offset=offsets[1],grid=mygrid))


# def sum_fields(*args):
#   return jax.tree_map(lambda *a: sum(a), *args)


# def stable_time_step(
#     max_velocity: float,
#     max_courant_number: float,
#     viscosity: float,
#     grid: grids.Grid,
#     implicit_diffusion: bool = False,
# ) -> float:
#   """Calculate a stable time step for Navier-Stokes."""
#   dt = advection.stable_time_step(max_velocity, max_courant_number, grid)
#   if not implicit_diffusion:
#     diffusion_dt = diffusion.stable_time_step(viscosity, grid)
#     if diffusion_dt < dt:
#       raise ValueError(f'stable time step for diffusion is smaller than '
#                        f'the chosen timestep: {diffusion_dt} vs {dt}')
#   return dt


# def dynamic_time_step(v: GridVariableVector,
#                       max_courant_number: float,
#                       viscosity: float,
#                       grid: grids.Grid,
#                       implicit_diffusion: bool = False) -> float:
#   """Pick a dynamic time-step for Navier-Stokes based on stable advection."""
#   v_max = jnp.sqrt(jnp.max(sum(u.data ** 2 for u in v)))
#   return stable_time_step(
#       v_max, max_courant_number, viscosity, grid, implicit_diffusion)


# def _wrap_term_as_vector(fun, *, name):
#   return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


# def navier_stokes_explicit_terms(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
#     forcing: Optional[ForcingFn] = None,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""
#   del grid  # unused

#   if convect is None:
#     def convect(v):  # pylint: disable=function-redefined
#       return tuple(
#           advection.advect_van_leer_using_limiters(u, v, dt) for u in v)
  
#   def diffuse_velocity(v, *args):
#     out = tuple(diffuse(u, *args) for u in v) #this assumes v is a tuple of u and v, if we dstack this it becomes a valid input for the CNN
# #     print(out)
#     return out
    
  
#   convection = _wrap_term_as_vector(convect, name='convection')
  
#   diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  


  
#   if forcing is not None:
#     forcing = _wrap_term_as_vector(forcing, name='forcing')

#   @tree_math.wrap
#   @functools.partial(jax.named_call, name='navier_stokes_momentum')
#   def _explicit_terms(v):

#     dv_dt = convection(v)

#     if viscosity is not None:
#       dv_dt += diffusion_(v, viscosity / density)
#     if forcing is not None:
#       dv_dt += forcing(v) / density
#     return dv_dt

#   def explicit_terms_with_same_bcs(v):
#     dv_dt = _explicit_terms(v)
#     return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

#   return explicit_terms_with_same_bcs


# # TODO(shoyer): rename this to explicit_diffusion_navier_stokes
# def semi_implicit_navier_stokes(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
#     pressure_solve: Callable = pressure.solve_fast_diag,
#     forcing: Optional[ForcingFn] = None,
#     time_stepper: Callable = time_stepping.forward_euler,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""

#   explicit_terms = navier_stokes_explicit_terms(
#       density=density,
#       viscosity=viscosity,
#       dt=dt,
#       grid=grid,
#       convect=convect,
#       diffuse=diffuse,
#       forcing=forcing)

#   pressure_projection = jax.named_call(pressure.projection, name='pressure')

#   # TODO(jamieas): Consider a scheme where pressure calculations and
#   # advection/diffusion are staggered in time.
#   ode = time_stepping.ExplicitNavierStokesODE(
#       explicit_terms,
#       lambda v: pressure_projection(v, pressure_solve)
#   )
#   step_fn = time_stepper(ode, dt)
#   return step_fn

    
# def corrected_semi_implicit_navier_stokes(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
#     pressure_solve: Callable = pressure.solve_fast_diag,
#     forcing: Optional[ForcingFn] = None,
#     time_stepper: Callable = time_stepping.forward_euler,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""

#   explicit_terms = corrected_navier_stokes_explicit_terms(
#       density=density,
#       viscosity=viscosity,
#       dt=dt,
#       grid=grid,
#       convect=convect,
#       diffuse=diffuse,
#       forcing=forcing)

#   pressure_projection = jax.named_call(pressure.projection, name='pressure')

#   # TODO(jamieas): Consider a scheme where pressure calculations and
#   # advection/diffusion are staggered in time.
#   ode = time_stepping.ExplicitNavierStokesODE(
#       explicit_terms,
#       lambda v: pressure_projection(v, pressure_solve)
#   )
#   step_fn = time_stepper(ode, dt)
#   return step_fn

# def corrected_navier_stokes_explicit_terms(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
#     forcing: Optional[ForcingFn] = None,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""
#   del grid  # unused

# #   if convect is None:
#   def convect(v):  # pylint: disable=function-redefined
#     return tuple(
#           advection.advect_van_leer(u, v, dt) for u in v)
# #   else:
# #   def convect_velocity(v):
# #       vels = []
# #       for vel in v:
# #         vels.append(vel.array.data)
        
# #       out = convect(jnp.dstack(vels))

# #       out = reshapeData(out,v[0].grid,
# #                     offsets=[v[0].offset,v[1].offset],
# #                     bcs = [v[0].bc,v[1].bc]
# #                          )

# #       return out
  
#   def diffuse_velocity(v, *args):
    
#     vels = []
#     for vel in v:
#         vels.append(vel.array.data)
#     out = diffuse(jnp.dstack(vels),*args)

#     out = reshapeData(out,v[0].grid,
#                 offsets=[v[0].offset,v[1].offset],
#                 bcs = [v[0].bc,v[1].bc]
#                      )

#     return out


#   convection = _wrap_term_as_vector(convect, name='convection')
  
#   diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  
  
#   if forcing is not None:
#     forcing = _wrap_term_as_vector(forcing, name='forcing')

#   @tree_math.wrap
#   @functools.partial(jax.named_call, name='navier_stokes_momentum')
#   def _explicit_terms(v):
#     dv_dt = convection(v)
#     if viscosity is not None:
#       dv_dt += diffusion_(v, viscosity / density)
#     if forcing is not None:
#       dv_dt += forcing(v) / density
#     return dv_dt

#   def explicit_terms_with_same_bcs(v):
#     dv_dt = _explicit_terms(v)
#     return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

#   return explicit_terms_with_same_bcs

# def sampling(data,factor):
#     return data[0::factor,0::factor,:]

# def HYBMOD2_semi_implicit_navier_stokes(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
# #     convect: Optional[ConvectFn] = None,
    
#     factor: float,
#     diffuse : lambda x: x,
#     superresFun = lambda x: x,
#     pressure_solve: Callable = pressure.solve_fast_diag,
#     forcing: Optional[ForcingFn] = None,
#     time_stepper: Callable = time_stepping.forward_euler,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""

#   explicit_terms = HYBMOD2_navier_stokes_explicit_terms(
#       density=density,
#       viscosity=viscosity,
#       dt=dt,
#       grid=grid,
#       diffuse=diffuse,
#       factor = factor,
#       applySuperresolution = superresFun,
#       forcing=forcing)

#   pressure_projection = jax.named_call(pressure.projection, name='pressure')

#   # TODO(jamieas): Consider a scheme where pressure calculations and
#   # advection/diffusion are staggered in time.
#   ode = time_stepping.ExplicitNavierStokesODE(
#       explicit_terms,
#       lambda v: pressure_projection(v, pressure_solve)
#   )
#   step_fn = time_stepper(ode, dt)
#   return step_fn

# def HYBMOD2_navier_stokes_explicit_terms(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
# #     convect: Optional[ConvectFn] = None,
#     factor: float,
#     diffuse: lambda x: x,
#     applySuperresolution: lambda x: x,
#     forcing: Optional[ForcingFn] = None,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""
#   del grid  # unused

# #   if convect is None:
# #     def convect(v):  # pylint: disable=function-redefined
# #       return tuple(
# #           advection.advect_van_leer_using_limiters(u, v, dt) for u in v)
  
# #   def diffuse_velocity(v, *args):
# #     vels = []
# #     for vel in v:
# #         vels.append(vel.array.data)
# #     out = diffuse(jnp.dstack(vels),*args)

# #     out = reshapeData(out,v[0].grid,
# #                 offsets=[v[0].offset,v[1].offset],
# #                 bcs = [v[0].bc,v[1].bc]
# #                      )

# #     return out

#   def superresolution(v):

#     flat = v.tree_flatten()[0][0]
    
#     u = flat[0].array.data
#     v = flat[1].array.data
#     reshapedV = jnp.dstack([u,v])
# #     print(np.shape(reshapedV))
#     high_def = applySuperresolution(reshapedV,factor)
# #     print(np.shape(reshapedV))
    
#     prev_grid = flat[0].grid
#     prev_size = prev_grid.shape
# #     print(prev_size)
#     new_size = (prev_size[0]*factor,prev_size[1]*factor)
#     mygrid = grids.Grid(new_size, domain = flat[0].grid.domain)
#     bcs = boundaries.channel_flow_boundary_conditions(ndim=2)
    
#     arrU = grids.GridArray(data = high_def[:,:,0],
#                      offset = flat[0].offset,
#                      grid = mygrid)
#     arrV = grids.GridArray(data = high_def[:,:,1],
#                      offset = flat[1].offset,
#                      grid = mygrid)
#     U = grids.GridVariable(array= arrU,bc = bcs)
#     V = grids.GridVariable(array= arrV,bc = bcs)
    

   
#     return tree_math.Vector([U,V])

  
#   def diffuse_velocity(v, *args):
#     out = tuple(diffuse(u, *args) for u in v) 
#     return out
    
  

    
#   def convect(v):  # pylint: disable=function-redefined
#     return tuple(
#       advection.advect_van_leer(u, v, dt) for u in v)

#   def downsample(v,factor):

#     u = v[0]
#     v = v[1]
    
#     prev_grid = u.grid
#     prev_size = prev_grid.shape
# #     print(prev_size)
#     new_size = (int(prev_size[0]/factor),int(prev_size[1]/factor))
#     mygrid = grids.Grid(new_size, domain = u.grid.domain)
    
#     bcs = boundaries.channel_flow_boundary_conditions(ndim=2)
    
#     arrU = grids.GridArray(data = u.data[::factor,::factor],
#                      offset = u.offset,
#                      grid = mygrid)
#     arrV = grids.GridArray(data = v.data[::factor,::factor],
#                      offset = v.offset,
#                      grid = mygrid)
    

#     return (arrU,arrV)

#   downsample_ = _wrap_term_as_vector(downsample, name='downsample')
 
#   convection = _wrap_term_as_vector(convect, name='convection')
  
#   diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  
  
  
#   superresolution_ = _wrap_term_as_vector(superresolution, name='superresolution')
  
#   if forcing is not None:
#     forcing = _wrap_term_as_vector(forcing, name='forcing')

#   @tree_math.wrap
#   @functools.partial(jax.named_call, name='navier_stokes_momentum')
#   def _explicit_terms(v):
#     print(v)
#     vels = superresolution(v)
    
#     dv_dt = convection(vels)
#     dv_dt += diffusion_(vels,  density/viscosity) #viscosity/density
# #     print("before downsample")
#     dv_dt = downsample_(dv_dt,factor)
# #     print("after downsample")
# #     print(dv_dt)
#     if forcing is not None:
# #       print(forcing(v)/density)
#       dv_dt += forcing(v) / density
    
    
    
    
#     return dv_dt

#   def explicit_terms_with_same_bcs(v):
    
#     dv_dt = _explicit_terms(v)

#     return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

#   return explicit_terms_with_same_bcs


# def implicit_diffusion_navier_stokes(
#     density: float,
#     viscosity: float,
#     dt: float,
#     grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffusion_solve: Callable = diffusion.solve_fast_diag,
#     pressure_solve: Callable = pressure.solve_fast_diag,
#     forcing: Optional[ForcingFn] = None,
# ) -> Callable[[GridVariableVector], GridVariableVector]:
#   """Returns a function that performs a time step of Navier Stokes."""
#   del grid  # unused
#   if convect is None:
#     def convect(v):  # pylint: disable=function-redefined
#       return tuple(
#           advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

#   convect = jax.named_call(convect, name='convection')
#   pressure_projection = jax.named_call(pressure.projection, name='pressure')
#   diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

#   # TODO(shoyer): refactor to support optional higher-order time integators
#   @jax.named_call
#   def navier_stokes_step(v: GridVariableVector) -> GridVariableVector:
#     """Computes state at time `t + dt` using first order time integration."""
#     convection = convect(v)
#     accelerations = [convection]
#     if forcing is not None:
#       # TODO(shoyer): include time in state?
#       f = forcing(v)
#       accelerations.append(tuple(f / density for f in f))
#     dvdt = sum_fields(*accelerations)
#     # Update v by taking a time step
#     v = tuple(
#         grids.GridVariable(u.array + dudt * dt, u.bc)
#         for u, dudt in zip(v, dvdt))
#     # Pressure projection to incompressible velocity field
#     v = pressure_projection(v, pressure_solve)
#     # Solve for implicit diffusion
#     v = diffusion_solve(v, viscosity, dt)
#     return v
#   return navier_stokes_step


























































































# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Examples of defining equations."""
import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import time_stepping
from jax_cfd.base import boundaries
import tree_math

# from jax_cfd.ml.diego_train_functions import reshapeData

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]



def reshapeData(mydata,mygrid,
                offsets=[(1.0,0.5),
                         (0.5,1.0)],
                bcs = [boundaries.channel_flow_boundary_conditions(ndim=2),
                      boundaries.channel_flow_boundary_conditions(ndim=2)]):
    """defaults to channel flow settings"""
    return (grids.GridArray(data = mydata[:,:,0],offset=offsets[0],grid=mygrid),
          grids.GridArray(data = mydata[:,:,1],offset=offsets[1],grid=mygrid))


def sum_fields(*args):
  return jax.tree_map(lambda *a: sum(a), *args)


def stable_time_step(
    max_velocity: float,
    max_courant_number: float,
    viscosity: float,
    grid: grids.Grid,
    implicit_diffusion: bool = False,
) -> float:
  """Calculate a stable time step for Navier-Stokes."""
  dt = advection.stable_time_step(max_velocity, max_courant_number, grid)
  if not implicit_diffusion:
    diffusion_dt = diffusion.stable_time_step(viscosity, grid)
    if diffusion_dt < dt:
      raise ValueError(f'stable time step for diffusion is smaller than '
                       f'the chosen timestep: {diffusion_dt} vs {dt}')
  return dt


def dynamic_time_step(v: GridVariableVector,
                      max_courant_number: float,
                      viscosity: float,
                      grid: grids.Grid,
                      implicit_diffusion: bool = False) -> float:
  """Pick a dynamic time-step for Navier-Stokes based on stable advection."""
  v_max = jnp.sqrt(jnp.max(sum(u.data ** 2 for u in v)))
  return stable_time_step(
      v_max, max_courant_number, viscosity, grid, implicit_diffusion)


def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


def navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

  if convect is None:
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)
  
  def diffuse_velocity(v, *args):
    out = tuple(diffuse(u, *args) for u in v) #this assumes v is a tuple of u and v, if we dstack this it becomes a valid input for the CNN
#     print(out)
    return out
    
  
  convection = _wrap_term_as_vector(convect, name='convection')
  
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  


  
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):

    dv_dt = convection(v)

    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    if forcing is not None:
      dv_dt += forcing(v) / density
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs


# TODO(shoyer): rename this to explicit_diffusion_navier_stokes
def semi_implicit_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  pressure_projection = jax.named_call(pressure.projection, name='pressure')

  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve)
  )
  step_fn = time_stepper(ode, dt)
  return step_fn

    
def corrected_semi_implicit_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = corrected_navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  pressure_projection = jax.named_call(pressure.projection, name='pressure')

  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve)
  )
  step_fn = time_stepper(ode, dt)
  return step_fn

def corrected_navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

#   if convect is None:
  def convect(v):  # pylint: disable=function-redefined
    return tuple(
          advection.advect_van_leer(u, v, dt) for u in v)
#   else:
#   def convect_velocity(v):
#       vels = []
#       for vel in v:
#         vels.append(vel.array.data)
        
#       out = convect(jnp.dstack(vels))

#       out = reshapeData(out,v[0].grid,
#                     offsets=[v[0].offset,v[1].offset],
#                     bcs = [v[0].bc,v[1].bc]
#                          )

#       return out
  
  def diffuse_velocity(v, *args):
    
    vels = []
    for vel in v:
        vels.append(vel.array.data)
    out = diffuse(jnp.dstack(vels),*args)

    out = reshapeData(out,v[0].grid,
                offsets=[v[0].offset,v[1].offset],
                bcs = [v[0].bc,v[1].bc]
                     )

    return out


  convection = _wrap_term_as_vector(convect, name='convection')
  
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  
  
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    if forcing is not None:
      dv_dt += forcing(v) / density
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs

def sampling(data,factor):
    return data[0::factor,0::factor,:]

def HYBMOD2_semi_implicit_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
    factor: float,
    superresFun = lambda x: x,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = HYBMOD2_navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      factor = factor,
      applySuperresolution = superresFun,
      forcing=forcing)

  pressure_projection = jax.named_call(pressure.projection, name='pressure')

  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve)
  )
  step_fn = time_stepper(ode, dt)
  return step_fn

def HYBMOD2_navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
#     convect: Optional[ConvectFn] = None,
#     diffuse: DiffuseFn = diffusion.diffuse,
    factor: float,
    applySuperresolution: lambda x: x,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

#   if convect is None:
#     def convect(v):  # pylint: disable=function-redefined
#       return tuple(
#           advection.advect_van_leer_using_limiters(u, v, dt) for u in v)
  
#   def diffuse_velocity(v, *args):
#     vels = []
#     for vel in v:
#         vels.append(vel.array.data)
#     out = diffuse(jnp.dstack(vels),*args)

#     out = reshapeData(out,v[0].grid,
#                 offsets=[v[0].offset,v[1].offset],
#                 bcs = [v[0].bc,v[1].bc]
#                      )

#     return out

  def superresolution(v):
    flat = v.tree_flatten()[0][0]
    
    u = flat[0].array.data
    v = flat[1].array.data
    reshapedV = jnp.dstack([u,v])
    
    high_def = applySuperresolution(reshapedV,factor)
    u = high_def[:,:,0]
    v = high_def[:,:,1]
    [dudy,dudx] = np.gradient(u)
    [dvdy,dvdx] = np.gradient(v)
    lapu = np.gradient(dudx,axis=1) + np.gradient(dudy,axis=0)
    lapv = np.gradient(dvdx,axis=1) + np.gradient(dvdy,axis=0)
    laps = jnp.dstack([
        lapu,
        lapv
    ])
    diffusion = sampling(laps,factor)*viscosity/density
    convection = jnp.dstack([
        u*dudx + u*dudy,
        v*dvdx + v*dvdy
    ])
    convection = sampling(convection,factor)
    out = diffusion #-convection
    
    out = reshapeData(out,flat[0].grid,
                offsets=[flat[0].offset,flat[1].offset],
                bcs = [flat[0].bc,flat[1].bc]
                     )
    return tree_math.Vector(out)
  
    
  def convect(v):  # pylint: disable=function-redefined
    return tuple(
      advection.advect_van_leer(u, v, dt) for u in v)
  
 
    
  
  convection = _wrap_term_as_vector(convect, name='convection')
  
  
  



  
  superresolution_ = _wrap_term_as_vector(superresolution, name='superresolution')
  
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = superresolution(v)
    dv_dt += convection(v)
    if forcing is not None:
      dv_dt += forcing(v) / density
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs


def implicit_diffusion_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffusion_solve: Callable = diffusion.solve_fast_diag,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused
  if convect is None:
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  convect = jax.named_call(convect, name='convection')
  pressure_projection = jax.named_call(pressure.projection, name='pressure')
  diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

  # TODO(shoyer): refactor to support optional higher-order time integators
  @jax.named_call
  def navier_stokes_step(v: GridVariableVector) -> GridVariableVector:
    """Computes state at time `t + dt` using first order time integration."""
    convection = convect(v)
    accelerations = [convection]
    if forcing is not None:
      # TODO(shoyer): include time in state?
      f = forcing(v)
      accelerations.append(tuple(f / density for f in f))
    dvdt = sum_fields(*accelerations)
    # Update v by taking a time step
    v = tuple(
        grids.GridVariable(u.array + dudt * dt, u.bc)
        for u, dudt in zip(v, dvdt))
    # Pressure projection to incompressible velocity field
    v = pressure_projection(v, pressure_solve)
    # Solve for implicit diffusion
    v = diffusion_solve(v, viscosity, dt)
    return v
  return navier_stokes_step