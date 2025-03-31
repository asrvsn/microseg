'''
Pyqtgraph OpenGL widgets
'''
from typing import List, Tuple
import numpy as np
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSizePolicy, QShortcut
from qtpy.QtGui import QKeySequence
import pyqtgraph.opengl as gl
import skimage
import skimage.measure
import math

from matgeo import Triangulation
from .base import MainWindow
from microseg.utils.colors import *

class GrabbableGLViewWidget(gl.GLViewWidget):
    '''
    Screen grabs on press enter key
    '''
    def _grab(self):
        self.grabFramebuffer().save('screenshot.png')
        print('Screenshot saved as screenshot.png')

    def keyPressEvent(self, ev):
        # Grab on enter key

        if ev.key() == Qt.Key.Key_Return:
            self._grab()
        else:
            super().keyPressEvent(ev)

class GrabbableGLViewWindow(MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vw = GrabbableGLViewWidget()
        self._vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self._vw)

class GLPlaneItem(gl.GLMeshItem):
    """ 
    A 3D XY-plane rendered at a specific Z position in a PyQtGraph OpenGL view.
    It is created as a quadrilateral (two triangles) and can be moved dynamically.
    """
    def __init__(self, x_size, y_size, z_pos=0, color=(1, 1, 1, 0.3), draw_edges=False):
        """
        Initializes the plane.

        Parameters:
        - x_size (float): Width of the plane in the X direction.
        - y_size (float): Height of the plane in the Y direction.
        - z_pos (float): Z position where the plane should be placed.
        - color (tuple): RGBA color of the plane.
        - draw_edges (bool): Whether to draw edges around the plane.
        """
        self.x_size = x_size
        self.y_size = y_size
        self.z_pos = z_pos
        self.color = color
        self.draw_edges = draw_edges
        meshdata = self._create_plane_mesh()
        super().__init__(meshdata=meshdata, smooth=False, drawEdges=draw_edges, glOptions='translucent')

    def _create_plane_mesh(self):
        """Create a quadrilateral plane from two triangles."""
        vertices = np.array([
            [0, 0, self.z_pos],           # Bottom-left (origin)
            [self.x_size, 0, self.z_pos],  # Bottom-right
            [self.x_size, self.y_size, self.z_pos],  # Top-right
            [0, self.y_size, self.z_pos]   # Top-left
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        colors = np.full((2, 4), self.color)
        return gl.MeshData(vertexes=vertices, faces=faces, faceColors=colors)

    def setZ(self, z_new):
        """Update the Z position of the plane."""
        self.z_pos = z_new
        self.resetTransform()  
        self.translate(0, 0, z_new)  

class GLZStackItem(gl.GLVolumeItem):
    MAX_TEXTURE_SIZE = 2048
    
    def __init__(self, z_stack: np.ndarray, *args, xyz_voxel_size: Tuple[float, float, float]=(1, 1, 1), glOptions='additive', alpha=0.5, **kwargs):
        '''
        Z-stack is ZXY format
        '''
        self._z_stack = z_stack
        self._xyz_voxel_size = np.array(xyz_voxel_size)
        self._xyz_scale = self._xyz_voxel_size
        self._alpha = alpha
        super().__init__(*args, z_stack, glOptions=glOptions, **kwargs)

    def setData(self, z_stack: np.ndarray, alpha=0.5):
        '''
        Z-stack is ZXY format
        '''
        self._z_stack = z_stack
        self._alpha = alpha
        data = self._process_stack()
        super().setData(data)
        self.resetTransform()
        if not np.allclose(self._xyz_scale, (1, 1, 1)):
            self.scale(*self._xyz_scale)

    def _process_stack(self) -> np.ndarray:
        if self._z_stack.ndim == 3:
            # Intensity data only provided
            # Prepare volume data (x,y,z,RGBA)
            vol = self._z_stack.transpose(2, 1, 0)  
            vol = vol.astype(np.float32)
            vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize to [0,1]
            
            # # Create alpha channel - transparent where intensity is 0, otherwise proportional to intensity
            # alpha = vol.copy() * self._alpha
            # # Optional: Set a minimum threshold to remove noise
            # alpha[vol < 0.01] = 0  # Black regions (near zero) become fully transparent
            alpha = np.full_like(vol, self._alpha)
            alpha[vol < 0.01] = 0
            
            # Create final RGBA volume - maintain original intensities for RGB channels
            vol_rgba = np.stack([vol] * 3 + [alpha], axis=-1)
            vol_rgba = (vol_rgba * 255).astype(np.ubyte)
        else:
            # RGB already provided
            assert self._z_stack.ndim == 4 
            assert self._z_stack.shape[3] in [3, 4]
            if self._z_stack.shape[3] == 3:
                alpha = np.full_like(self._z_stack[..., 0], int(255 * self._alpha))
                alpha[self._z_stack[..., 0] == 0] = 0
                vol_rgba = np.concatenate([self._z_stack, alpha[..., None]], axis=-1)
            else:
                vol_rgba = self._z_stack
            vol_rgba = vol_rgba.astype(np.ubyte)

        # Downscale to fit within OpenGL texture size limitations as needed 
        if max(vol_rgba.shape[:2]) > self.MAX_TEXTURE_SIZE:
            ds_x = math.ceil(vol_rgba.shape[0] / self.MAX_TEXTURE_SIZE)
            ds_y = math.ceil(vol_rgba.shape[1] / self.MAX_TEXTURE_SIZE)
            ds = np.array([ds_x, ds_y] + [1] * (vol_rgba.ndim - 2))
            vol_rgba = skimage.measure.block_reduce(vol_rgba, tuple(ds), np.max)
            self._xyz_scale = self._xyz_voxel_size * ds[:3]
            
        print(f'Volume size: {vol_rgba.shape}')

        return vol_rgba

    
class GLTriangulationItem(gl.GLMeshItem):
    FACE_COLORS = cc_glasbey_01_rgba
    
    def __init__(self, tri: Triangulation, *args, color_mode=None, only_watertight=False, md_kwargs=dict(), **kwargs):
        assert color_mode in [None, 'cc', 'face']
        self._color_mode = color_mode
        super().__init__(*args, **kwargs)
        self.setData(tri, only_watertight=only_watertight, **md_kwargs)

    def setData(self, tri: Triangulation, only_watertight: bool=False, **md_kwargs):
        if self._color_mode == 'cc':
            tris = tri.get_connected_components(only_watertight=only_watertight)
            labels = np.concatenate([np.full(len(t.simplices), i) for i, t in enumerate(tris)])
            tri = Triangulation.merge(tris)
            md_kwargs['faceColors'] = self.FACE_COLORS[labels % len(self.FACE_COLORS)]
        elif self._color_mode == 'face':
            md_kwargs['faceColors'] = self.FACE_COLORS[np.arange(len(tri.simplices)) % len(self.FACE_COLORS)]
        md = gl.MeshData(vertexes=tri.pts, faces=tri.simplices, **md_kwargs)
        self.setMeshData(meshdata=md)

class GLHoverableSurfaceViewWidget(gl.GLViewWidget):
    '''
    GLView + Triangulation with hoverable 3D points using raycasting
    '''
    sel_eps: float=1e-2 # Percentage of viewport diagonal to consider a selection
    size: int=20
    h_size: int=20
    h_alpha: float=0.75
    face_rgba: float=0.8
    mesh_opts: dict = {
        'shader': 'normalColor',
        'glOptions': 'opaque',
        'drawEdges': True,
        'smooth': False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._mesh = gl.GLMeshItem(**self.mesh_opts)
        self._normals = gl.GLLinePlotItem(color=(1,1,1,1), antialias=True)
        self._sp_hov = gl.GLScatterPlotItem(pxMode=True)
        self._points = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._mesh)
        self.addItem(self._normals)
        self.addItem(self._sp_hov)
        self.addItem(self._points)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Listeners
        self._esc_sc = QShortcut(QKeySequence('Escape'), self)
        self._esc_sc.activated.connect(self._escape)

        # State
        self.cam_motion_enabled = True
        self._show_normals = False
        self._show_points = False
        self._mat = None
        self._dragging_cam = False
        self._tri = None
        self._tri_normals = None
        self._tri_centroids = None
        self._md = None
        self._proj_pts = None
        self._width, self._height = None, None
        self._diag = None
        self._cam_facing = None # Camera-facing points
        self._reset_mouse()

    def setTriangulation(self, tri: Triangulation, *args, **kwargs):
        if self._tri is None:
            # If this is the first time receiving a mesh, move the camera point
            origin = tri.pts.mean(axis=0)
            self.pan(origin[0], origin[1], origin[2])
        self._tri = tri
        self._md = gl.MeshData(vertexes=tri.pts, faces=tri.simplices, faceColors=np.full((len(tri.simplices), 4), self.face_rgba))
        kwargs = dict(size=self.size) | kwargs
        self._mesh.setMeshData(meshdata=self._md)
        print(f'Set mesh with {len(tri.pts)} points and {len(tri.simplices)} faces')
        self._tri_normals = tri.compute_normals()
        self._tri_centroids = tri.compute_centroids()
        self._redrawNormals()
        self._redrawPoints()
        self._escape()
        self._project()

    def toggleNormals(self):
        self._show_normals = not self._show_normals
        self._redrawNormals()

    def togglePoints(self):
        self._show_points = not self._show_points
        self._redrawPoints()

    def _redrawNormals(self):
        if self._show_normals:
            print('Showing normals')
            simp = self._tri.simplices[np.random.choice(len(self._tri.simplices))] # Use a random triangle for normal length
            normal_length = np.linalg.norm(self._tri.pts[simp[1]] - self._tri.pts[simp[0]])
            starts = self._tri_centroids
            ends = starts + self._tri_normals * normal_length
            lines = np.empty((len(starts) * 2, 3), dtype=np.float32) # Interleave start and end lines
            lines[0::2] = starts
            lines[1::2] = ends
            self._normals.setData(pos=lines, width=1, mode='lines')
        else:
            self._normals.setData(pos=np.empty((0, 3), dtype=np.float32))

    def _redrawPoints(self):
        if self._show_points:
            print('Showing points')
            self._points.setData(pos=self._tri.pts, size=self.size, color=(0,1,0,1))
        else:
            self._points.setData(pos=np.empty((0, 3), dtype=np.float32))

    def _project(self):
        if not self._tri is None:
            pts = self._tri.pts
            
            # Get the current view and projection matrices
            proj = np.array(self.projectionMatrix(region=None).data()).reshape(4, 4)
            view = np.array(self.viewMatrix().data()).reshape(4, 4)
            mat = view @ proj

            # If the view has change, re-project
            if self._mat is None or not np.allclose(self._mat, mat):
                self._mat = mat

                # Project 3D points to normalized device coordinates
                pts_4d = np.column_stack((pts, np.ones(pts.shape[0])))
                proj_pts = pts_4d @ mat # mat is not transposed, which is weird, but correct.

                # Perform perspective division
                proj_pts[:, :3] /= proj_pts[:, 3, None]

                # Convert to viewport coordinates
                _, __, self._width, self._height = self.getViewport()
                self._diag = np.sqrt(self._width**2 + self._height**2)
                self._proj_pts = np.column_stack((
                    (proj_pts[:, 0] + 1) * self._width / 4, # This 4 is weird, but correct.
                    (1 - proj_pts[:, 1]) * self._height / 4
                ))

                # Compute camera-facing triangles
                self._compute_cam_facing()
                # print('ran projection')

    def _compute_cam_facing(self):
        cam_to_tri = np.array(self.cameraPosition()) - self._tri_centroids
        facing_cam = (self._tri_normals * cam_to_tri).sum(axis=1) > 0
        self._cam_facing = np.unique(self._tri.simplices[facing_cam])
        # print(f'Computed cam-facing')

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        self._project()

    def mouseMoveEvent(self, ev):
        if self.cam_motion_enabled:
            super().mouseMoveEvent(ev)
            if ev.buttons():  # Check if any mouse button is held
                self._dragging_cam = True
            else:
                self._dragging_cam = False
        if not self._proj_pts is None:
            pos = ev.pos()
            dists = np.linalg.norm(self._proj_pts[self._cam_facing] - np.array([pos.x(), pos.y()]), axis=1)
            if dists.size > 0:
                idx = np.argmin(dists)
                if dists[idx] < self.sel_eps * self._diag:
                    hovered = self._cam_facing[idx]
                else:
                    hovered = None
                if hovered != self._hovered:
                    self._hovered = hovered
                    self._redraw_hov()

    def _redraw_hov(self):
        hov = [] if self._hovered is None else [self._hovered]
        self._sp_hov.setData(pos=self._tri.pts[hov], size=self.h_size, color=(1,1,1,self.h_alpha))

    def _reset_mouse(self):
        self._hovered = None

    def _escape(self):
        self._reset_mouse()
        self._redraw_hov()

class GLSelectableSurfaceViewWidget(GLHoverableSurfaceViewWidget):
    '''
    Same as hover but supports (multiple) selection of the points
    '''
    s_alpha: float=1.0
    selectionChanged = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sp_sel = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._sp_sel)

    def _reset_mouse(self):
        super()._reset_mouse()
        self._selected = set()

    def _redraw_sel(self):
        sel = list(self._selected)
        self._sp_sel.setData(pos=self._tri.pts[sel], size=self.h_size, color=(1,1,1,self.s_alpha))
        # print(f'Selected these points: {sel}')
        self.selectionChanged.emit(sel)

    def _escape(self):
        super()._escape()
        self._selected = set()
        self._redraw_sel()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if not self._hovered is None:
            if ev.button() == Qt.MouseButton.LeftButton and not self._dragging_cam:
                print('selecting', self._hovered)
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._selected.add(self._hovered)
                else:
                    self._selected = set([self._hovered])
                self._redraw_sel()

class GLDrawableSurfaceViewWidget(GLHoverableSurfaceViewWidget):
    '''
    Same as hover but supports drawing embedded polygons
    '''
    mesh_opts: dict = GLHoverableSurfaceViewWidget.mesh_opts | {
        'shader': None
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Listeners
        self._edit_sc = QShortcut(QKeySequence('E'), self)
        self._edit_sc.activated.connect(self._edit)

        # State
        self._mask = None # For each triangle, associate a face ID (bit like a pixel mask); 0 is traditionally the unlabeled.
        self._reset_drawing_state()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._drawn_poly = None
        self._drawn_label = None
        # Enable viewport movement
        self.cam_motion_enabled = True

    def _reset_mouse(self):
        super()._reset_mouse()
        self._reset_drawing_state()

    def _edit(self):
        if not self._is_drawing:
            self._is_drawing = True
            self.cam_motion_enabled = False

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self._is_drawing:
            if not self._hovered is None:
                if Qt.MouseButton.LeftButton & ev.buttons():
                    if self._drawn_poly is None:
                        print('starting draw')
                        self._drawn_poly = [self._hovered]
                    else:
                        self._drawn_poly.append(self._hovered)

def GLMakeSynced(base_class):
    class SyncedGLViewWidget(base_class):
        '''
        Shamelessly taken from 
        https://stackoverflow.com/questions/70551355/link-cameras-positions-of-two-3d-subplots-in-pyqtgraph
        '''

        def __init__(self, *args, **kwargs):
            self.linked_views: List[GrabbableGLViewWidget] = []
            super().__init__(*args, **kwargs)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def wheelEvent(self, ev):
            """Update view on zoom event"""
            super().wheelEvent(ev)
            self._update_views()

        def mouseMoveEvent(self, ev):
            """Update view on move event"""
            super().mouseMoveEvent(ev)
            self._update_views()

        def mouseReleaseEvent(self, ev):
            """Update view on move event"""
            super().mouseReleaseEvent(ev)
            self._update_views()

        def _update_views(self):
            """Take camera parameters and sync with all views"""
            camera_params = self.cameraParams()
            # Remove rotation, we can't update all params at once (Azimuth and Elevation)
            camera_params["rotation"] = None
            for view in self.linked_views:
                view.setCameraParams(**camera_params)

        def sync_camera_with(self, view: GrabbableGLViewWidget):
            """Add view to sync camera with"""
            self.linked_views.append(view)

    return SyncedGLViewWidget

