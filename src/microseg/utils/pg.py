'''
Misc tools for pyqtgraph rendering
Mirrors the functionality of mpl_tools.py
'''

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy import QtGui, QtWidgets
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt
import numpy as np
import pdb
import sys

from typing import List, Tuple, Callable

from matgeo.ellipsoid import Ellipsoid
from matgeo.plane import Plane, PlanarPolygon
from matgeo.triangulation import FaceTriangulation, VoronoiTriangulation, Triangulation

from .colors import *
from asrvsn_math.vectors import vec_angle
from microseg.widgets.pg_gl import GLMakeSynced, GrabbableGLViewWidget, GrabbableGLViewWindow
# from pg_widgets import *
# from pg_seg_widgets import *

''' Constants '''

rel_figsize = (500,500)

gl_mesh_default_opts = {
    'shader': 'balloon',
    'glOptions': 'opaque',
    'drawEdges': True,
    'smooth': False,
}

''' State '''

plots = []

''' Common API (mixed 2D/3D plots) '''

def plot(index: Tuple[int, int], func, nd: int, *args):
    global plots
    assert nd in [2, 3], 'nd must be 2 or 3'
    max_y = max(index[0], len(plots)-1)
    max_x = max(index[1], len(plots[0])-1 if len(plots) > 0 else 0)
    for y in range(max_y+1):
        if len(plots) <= y:
            plots.append([])
        for x in range(max_x+1):
            if len(plots[y]) <= x:
                plots[y].append(None)
    plots[index[0]][index[1]] = ([func], [args], nd)
    assert all(len(plots[y]) == len(plots[0]) for y in range(len(plots))), 'plots array is ragged'

def modify(index: Tuple[int, int], func, *args):
    global plots
    assert index[0] < len(plots), 'y axis out of range'
    assert index[1] < len(plots[index[0]]), 'x axis out of range'
    assert plots[index[0]][index[1]] != None, 'no existing plot to modify'
    plots[index[0]][index[1]][0].append(func)
    plots[index[0]][index[1]][1].append(args)

def render(synced=False) -> pg.GraphicsLayoutWidget:
    global plots
    assert all(len(plots[y]) == len(plots[0]) for y in range(len(plots))), 'plots array is ragged'
    if len(plots) > 0:
        nr, nc = len(plots), len(plots[0])
        win_size = (rel_figsize[1]*nc, rel_figsize[0]*nr)
        win = pg.GraphicsLayoutWidget(size=win_size, show=True)
        layout = QtWidgets.QGridLayout()
        win.setLayout(layout)
        ws_3d = []
        ws_2d = []
        
        for y, row in enumerate(plots):
            for x, element in enumerate(row):
                if element != None:
                    (funcs, argss, nd) = element
                    assert nd in [2, 3], 'nd must be 2 or 3'
                    if nd == 2:
                        widget = PlotWidget()
                        if synced:
                            for other in ws_2d:
                                widget.setXLink(other)
                                widget.setYLink(other)
                                other.setXLink(widget)
                                other.setYLink(widget)
                            ws_2d.append(widget)
                    elif nd == 3:
                        if synced:
                            widget = GLMakeSynced(GrabbableGLViewWidget)()
                            for other in ws_3d:
                                widget.sync_camera_with(other)
                                other.sync_camera_with(widget)
                            ws_3d.append(widget)
                        else:
                            widget = GrabbableGLViewWidget()
                    for f, args in zip(funcs, argss):
                        f(widget, *args)
                    layout.addWidget(widget, y, x)
        return win
    else:
        print('No axes found, not plotting.')

def show(synced=False, fullscreen=False):
    win = render(synced)
    if win != None:
        if fullscreen:
            win.showFullScreen()
        QtWidgets.QApplication.instance().exec()

def run_gl(fun: Callable):
    """
    Run a PyQtGraph window, handling Qt application lifecycle.
    For Jupyter notebooks: Creates/reuses QApplication and shows window.
    For scripts: Runs event loop until window is closed.
    """

    # Create/get QApplication instance
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    # Close any existing windows
    for w in QApplication.topLevelWindows():
        w.close()

    # Setup new window
    win = GrabbableGLViewWindow()
    fun(win._vw)
    # win.setAttribute(Qt.WA_DeleteOnClose)
    win.show()
    return app.exec_()

''' 2D plotting API '''

def plot_2d(index: Tuple[int, int], func, *args):
    plot(index, func, 2, *args)

def image_2d(w: pg.PlotWidget, img: np.ndarray, invert_y: bool=True, aspect_locked: bool=True, **kwargs):
    ''' 
    Plot image. Inverts y-axis by default.
    '''
    img = pg.ImageItem(img, **kwargs)
    w.addItem(img)
    if invert_y:
        w.invertY(True)
    if aspect_locked:
        w.setAspectLocked(True)

''' 3D OpenGL plotting API '''

def plot_3d(index: Tuple[int, int], func, *args):
    plot(index, func, 3, *args)

def xy_grid_3d(vw: gl.GLViewWidget, pts: np.ndarray=None):
    gr = gl.GLGridItem()
    if not (pts is None):
        # Add grid item sized 2x in radius in xy plane as points
        assert pts.ndim == 2
        assert pts.shape[1] >= 2
        xy_max = np.max(np.abs(pts[:, :2]))
        gr.scale(xy_max/10, xy_max/10, 1)
    vw.addItem(gr)

def axes_3d(vw: gl.GLViewWidget, pts: np.ndarray=None, scale: float=None):
    ax = gl.GLAxisItem()
    if not (scale is None):
        ax.setSize(x=scale, y=scale, z=scale)
    elif not (pts is None):
        # Add axes item scaled to fit points
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        max_ = np.max(np.abs(pts))
        ax.setSize(x=max_, y=max_, z=max_)
    vw.addItem(ax)

def scatter_3d(vw: gl.GLViewWidget, pts: np.ndarray, **kwargs):
    sp1 = gl.GLScatterPlotItem(pos=pts, pxMode=True, **kwargs)
    sp1.setGLOptions('translucent') # Fixes issue with white backgrounds
    vw.addItem(sp1)

def tri_mesh_3d(vw: gl.GLViewWidget, pts: np.ndarray, simplices: np.ndarray, colors=None, grid=True, **kwargs):
    assert simplices.ndim == 2
    assert simplices.shape[1] == 3
    kwargs = gl_mesh_default_opts | kwargs

    if grid:
        xy_grid_3d(vw, pts)

    # Differentiate faces using random face colors from colorcet (see cc_glasbey_rgba above)
    if colors is None:
        colors = cc_glasbey_01_rgba[1:]
        colors = colors[np.arange(simplices.shape[0]) % colors.shape[0]]
    else:
        assert colors.shape[0] == simplices.shape[0]
    md1 = gl.MeshData(vertexes=pts, faces=simplices, faceColors=colors)
    sp1 = gl.GLMeshItem(meshdata=md1, **kwargs)
    vw.addItem(sp1)

def triangulation_3d(vw: gl.GLViewWidget, tri: Triangulation, **kwargs):
    tri_mesh_3d(vw, tri.pts, tri.simplices, **kwargs)

def ellipsoid_3d(vw: gl.GLViewWidget, ell: Ellipsoid, grid=False, **kwargs):
    kwargs = gl_mesh_default_opts | kwargs
    md = gl.MeshData.sphere(rows=20, cols=40)
    pts = md.vertexes()
    pts = ell.map_sphere(pts)

    if grid:
        xy_grid_3d(vw, pts)
    
    md = gl.MeshData(vertexes=pts, faces=md.faces())
    sp = gl.GLMeshItem(meshdata=md, **kwargs)
    vw.addItem(sp)

def plane_3d(vw: gl.GLViewWidget, plane: Plane, **kwargs):
    ''' Plot a plane using grid item '''
    assert plane.ndim == 3
    gr = gl.GLGridItem()
    n = np.array([0, 0, 1])
    axis = np.cross(plane.n, n)
    angle = vec_angle(plane.n, n)
    gr.rotate(angle, *axis)
    gr.translate(*plane.v)
    gr.scale(10, 10, 1)
    vw.addItem(gr)

def face_triangulation_3d(vw: gl.GLViewWidget, tri: FaceTriangulation, colormap: np.ndarray=None, scheme='categorical', **kwargs):
    labels = tri.label_simplices()
    if not (colormap is None):
        assert colormap.shape[0] == len(tri.faces)
        labels = colormap[labels]
    colors = map_colors(labels, scheme, rgba=True)
    tri_mesh_3d(vw, tri.pts, tri.simplices, colors=colors, **kwargs)

def polygons_3d(vw: gl.GLViewWidget, polygons: List[np.ndarray], centers: np.ndarray, colormap: np.ndarray=None, scheme='categorical', **kwargs):
    assert all([p.shape[1] == 3 for p in polygons]), 'All polygons must be 3D'
    assert centers.shape[1] == 3, 'Centers must be 3D'
    labels = np.arange(len(polygons)) if colormap is None else colormap
    colors = map_colors(labels, scheme, rgba=True)
    all_vertices = []
    all_simplices = []
    all_colors = []
    n_vert = 0
    for p, c, o in zip(polygons, centers, colors):
        n = p.shape[0]
        vertices = np.vstack([p, c])
        simplices = np.vstack([
            np.arange(n),
            np.roll(np.arange(n), 1),
            np.full(n, n)
        ]).T + n_vert
        assert simplices.shape == (n, 3)
        o_ = np.full((n, 4), o)
        all_vertices.append(vertices)
        all_simplices.append(simplices)
        all_colors.append(o_)
        n_vert += vertices.shape[0]
    vertices = np.vstack(all_vertices)
    simplices = np.vstack(all_simplices)
    colors = np.vstack(all_colors)
    tri_mesh_3d(vw, vertices, simplices, grid=False, colors=colors)

def ppolygons_3d(vw: gl.GLViewWidget, polygons: List[PlanarPolygon], centers: np.ndarray, *args, **kwargs):
    assert all([p.ndim == 3 for p in polygons]), 'All polygons must be 3D'
    polygons = [
        p.vertices_nd for p in polygons
    ]
    polygons_3d(vw, polygons, centers, *args, **kwargs)

def image_3d(vw: gl.GLViewWidget, img: np.ndarray, origin=np.zeros(3), grid=True, **kwargs):
    '''
    Plot an image in a 3D coordinate system in the XY plane centered at "center"
    image may be 2d or 3d, in which case the last dimension is interpreted as RGB
    '''
    Nx, Ny = img.shape[:2]
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    pts = np.stack([x, y, np.zeros_like(x)], axis=-1)
    pts = pts.reshape(-1, 3)
    pts = pts + origin
    scatter_3d(vw, pts, color=img.reshape(-1, img.shape[-1]), size=1, **kwargs)
    if grid:
        xy_grid_3d(vw, pts)
