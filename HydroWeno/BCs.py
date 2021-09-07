from .SimSettings import Prim, Gas

def reflecting_bc(position='Lower'):
    if position == 'Lower':
        def reflecting_lower(grid, V):
            for i in range(grid.numGhost):
                k0 = 2 * grid.numGhost - 1

                V[:, i] = V[:, k0-i]
                V[Prim.Velo, i] = -V[Prim.Velo, k0-i]
        return reflecting_lower
    elif position == 'Upper':
        def reflecting_upper(grid, V):
            for i in range(grid.numGhost):
                k0 = 2 * grid.numGhost - 1
                k1 = grid.griEnd - grid.numGhost

                V[:, k1+k0-i] = V[:, k1+i]
                V[Prim.Velo, k1+k0-i] = -V[Prim.Velo, k1+i]
        return reflecting_upper
    
    raise ValueError('Unknown position argument %s. Expected: "Lower"|"Upper"' % position)

def diode_bc(position='Lower'):
    if position == 'Lower':
        def diode_lower(grid, V):
            for i in range(grid.numGhost):
                k0 = 2 * grid.numGhost - 1
                k1 = grid.griEnd - grid.numGhost

                V[:, i] = V[:, k0-i]
                V[Prim.Velo, i] = min(0.0, V[Prim.Velo, k0-i])
        return diode_lower
    elif position == 'Upper':
        def diode_upper(grid, V):
            for i in range(grid.numGhost):
                k0 = 2 * grid.numGhost - 1
                k1 = grid.griEnd - grid.numGhost

                V[:, k1+k0-i] = V[:, k1+i]
                V[Prim.Velo, k1+k0-i] = max(0.0, V[Prim.Velo, k1+i])
        return diode_upper
    
    raise ValueError('Unknown position argument %s. Expected: "Lower"|"Upper"' % position)

def fixed_bc(values, position='Lower'):
    if position == 'Lower':
        def fixed_lower(grid, V):
            V[:, :grid.numGhost] = values
        return fixed_lower
    elif position == 'Upper':
        def fixed_upper(grid, V):
            V[:, -grid.numGhost:] = values
        return fixed_upper
    
    raise ValueError('Unknown position argument %s. Expected: "Lower"|"Upper"' % position)

def zero_grad_bc(position='Lower'):
    if position == 'Lower':
        def zero_grad_lower(grid, V):
            for i in range(grid.numGhost):
                V[:, i] = V[:, grid.numGhost]
        return zero_grad_lower
    elif position == 'Upper':
        def zero_grad_upper(grid, V):
            for i in range(grid.numGhost):
                V[:, -(i+1)] = V[:, -grid.numGhost-1]
        return zero_grad_upper
    
    raise ValueError('Unknown position argument %s. Expected: "Lower"|"Upper"' % position)


def apply_bcs(grid, V):
    typ = 'Reflect'
    if typ == 'Reflect':
        for i in range(grid.numGhost):
            k0 = 2 * grid.numGhost - 1
            k1 = grid.griEnd - grid.numGhost

            V[:, i] = V[:, k0-i]
            V[Prim.Velo, i] = -V[Prim.Velo, k0-i]

            V[:, k1+k0-i] = V[:, k1+i]
            V[Prim.Velo, k1+k0-i] = -V[Prim.Velo, k1+i]
    if typ == 'LowerDiode':
        for i in range(grid.numGhost):
            k0 = 2 * grid.numGhost - 1
            k1 = grid.griEnd - grid.numGhost

            V[:, i] = V[:, k0-i]
            V[Prim.Velo, i] = min(0.0, V[Prim.Velo, k0-i])

            V[:, k1+k0-i] = V[:, k1+i]
            V[Prim.Velo, k1+k0-i] = -V[Prim.Velo, k1+i]
    elif typ == 'Fixed':
        V[Prim.Dens, :grid.numGhost] = 3.857143
        V[Prim.Velo, :grid.numGhost] = 2.629369
        V[Prim.Pres, :grid.numGhost] = 10.33333333
    
        for i in range(1, grid.numGhost+1):
            V[:, -i] = V[:, -grid.numGhost-1]
    
        V[Prim.Eint] = V[Prim.Pres] / ((Gas.Gamma-1.0) * V[Prim.Dens])
    elif typ == 'ZeroGrad':
        for i in range(grid.numGhost):
            V[:, i] = V[:, grid.numGhost]
            V[:, -(i+1)] = V[:, -grid.numGhost-1]
    
        V[Prim.Eint] = V[Prim.Pres] / ((Gas.Gamma-1.0) * V[Prim.Dens])