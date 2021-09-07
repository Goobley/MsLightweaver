import numpy as np

XStart = 0.0
XEnd = 1.0
NumGhost = 4
NumCells = 100
GriBeg = NumGhost
GriEnd = NumCells + NumGhost
GriMax = NumCells + 2 * NumGhost
GrDx = (XEnd - XStart) / NumCells
XPos = np.linspace(XStart - NumGhost * GrDx, XEnd + NumGhost * GrDx, GriMax)