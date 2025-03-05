'''
Base classes for building apps from ROI editors
'''
from typing import Dict, Optional
import pyqtgraph.opengl as gl # Has to be imported before qtpy
from matgeo import Triangulation
from scipy.spatial import cKDTree

from .base import *
from .pg import *
from .roi_image import *
from .pg_gl import *

class ImageSegmentorApp(SaveableApp):
    '''
    Simple usable app for segmenting single (or stacks) of images in ZXYC format
    '''
    def __init__(self, img_path: str, desc: str='rois', *args, **kwargs):
        # State
        self._z = 0
        self._img_path = img_path
        self._img = load_stack(img_path, fmt='ZXYC')
        self._zmax = self._img.shape[0]
        self._rois = [[] for _ in range(self._zmax)]
        
        # Widgets
        self._main = VLayoutWidget()
        self._creator = ROIsCreator()
        self._main._layout.addWidget(self._creator)
        self._z_slider = IntegerSlider(mode='scroll')
        self._main._layout.addWidget(self._z_slider)
        if self._zmax == 1:
            print(f'Received standard 2D image, disabling z-slider')
            self._z_slider.hide()
        else:
            print(f'Received z-stack with {self._zmax} slices, enabling z-slider')
            self._z_slider.setData(0, self._zmax-1, self._z)

        # Listeners
        self._creator.add.connect(self._add)
        self._creator.delete.connect(self._delete)
        self._z_slider.valueChanged.connect(lambda z: self._set_z(z))

        # Run data load and rest of initialization in superclass
        self._pre_super_init() # TODO: so ugly
        self._creator.setImage(self._img[self._z])
        super().__init__(
            f'Segmenting {desc} on image: {os.path.basename(img_path)}',
            f'{os.path.splitext(img_path)[0]}.{desc}',
        *args, **kwargs)
        self.setCentralWidget(self._main)

    ''' Overrides '''

    def copyIntoState(self, state: List[List[ROI]]):
        self._rois = [[r.copy() for r in subrois] for subrois in state]
        assert len(self._rois) == self._zmax, f'Expected {self._zmax} z-slices, got {len(self._rois)}'
        self.refreshROIs(push=False)

    def copyFromState(self) -> List[List[ROI]]:
        return [[r.copy() for r in subrois] for subrois in self._rois]

    def readData(self, path: str) -> List[List[ROI]]:
        rois = pickle.load(open(path, 'rb'))
        # Allow flat-list of ROIs for 1-stack images
        if not type(rois[0]) is list:
            rois = [rois]
        # Allow unlabeled ROIs
        lbl = max([max([r.lbl for r in subrois if type(r) is LabeledROI], default=0) for subrois in rois], default=0) + 1
        for subrois in rois:
            for i, r in enumerate(subrois):
                if not type(r) is LabeledROI:
                    subrois[i] = LabeledROI(lbl, r)
                    lbl += 1
        return rois
    
    def writeData(self, path: str, rois: List[List[ROI]]):
        # Write flat-list of ROIs for 1-stack images
        if self._zmax == 1:
            rois = rois[0]
        pickle.dump(rois, open(path, 'wb'))

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Left:
            self._set_z(self._z-1, set_slider=True)
        elif evt.key() == Qt.Key_Right:
            self._set_z(self._z+1, set_slider=True)
        else:
            super().keyPressEvent(evt)

    def refreshROIs(self, push: bool=True):
        self._creator.setROIs(self._rois[self._z])
        if push:
            self.pushEdit()

    ''' Private methods '''

    def _pre_super_init(self):
        pass

    def _set_z(self, z: int, set_slider: bool=False):
        z = max(0, min(z, self._zmax-1))
        if z != self._z:
            if set_slider:
                self._z_slider.setValue(z) # Callback will go to next branch
            else:
                self._z = z
                self._creator.setData(self._img[z], self._rois[z])

    @property
    def next_label(self) -> int:
        return max(
            [max([r.lbl for r in subrois], default=-1) 
             for subrois in self._rois], 
            default=-1
        ) + 1

    def _add(self, rois: List[ROI]):
        l = self.next_label
        lrois = [LabeledROI(l+i, r) for i, r in enumerate(rois)]
        self._rois[self._z].extend(lrois)
        self.refreshROIs(push=True)

    def _delete(self, rois: Set[int]):
        self._rois[self._z] = [
            r for r in self._rois[self._z] if not (r.lbl in rois)
        ]
        self.refreshROIs(push=True)

class ZStackObjectViewer(gl.GLViewWidget):
    '''
    3D viewer of multiple objects with current z-plane rendered
    '''
    nav_key_pressed = QtCore.Signal(object)
    start_proposing = QtCore.Signal()
    finish_proposing = QtCore.Signal(object)
    mesh_opts: dict = {
        # 'shader': 'normalColor',
        'glOptions': 'opaque',
        'drawEdges': False,
        'smooth': False,
    }
    cursor_opts: dict = {
        'pxMode': True,
        'color': (1.0, 1.0, 1.0, 1.0),
        'size': 10,
    }
    facecolors = cc_glasbey_01_rgba

    def __init__(self, imgsize: np.ndarray, voxsize: np.ndarray, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self._imgsize = imgsize # XYZ
        self._zmax = imgsize[2]
        self._voxsize = voxsize # XYZ
        self._z_aniso = voxsize[2] / voxsize[0] # Rendered in space where XY are unit-size pixels, scale z accordingly
        self.setWindowTitle('Z-Slice Object Viewer')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        MainWindow.resizeToScreen(self, offset=1) # Show on next avail screen
        viewsize = imgsize.copy() # Shape of viewport
        viewsize[2] *= self._z_aniso
        self.opts['center'] = pg.Vector(*(viewsize/2))
        self.setCameraPosition(distance=viewsize.max() * 1.1) # Zoom out sufficiently
        self._cursor_pt = gl.GLScatterPlotItem(**self.cursor_opts)
        self.addItem(self._cursor_pt)

        # State
        self._z = 0
        self._plane = None
        self._meshes = []
        self._rois = []
        self._is_proposing = False
        self._zpair_propose = (0, 0)
        self._proposed_map = dict()
        self._proposed_rois = []

    def setROIs(self, rois: List[List[ROI]]):
        assert not self._is_proposing
        self._rois = rois
        self._render_rois(rois)

    def setZ(self, z: int, img: np.ndarray):
        self._z = z
        if not self._plane is None:
            self.removeItem(self._plane)
        img = img.astype(np.float32)  # Convert to float for proper scaling
        img -= img.min()  # Shift min to 0
        img /= img.max()  # Normalize to [0, 1]
        img_8bit = (img * 255).astype(np.uint8)  # Scale to [0, 255]
        img_rgba = np.stack([img_8bit] * 3 + [np.full_like(img_8bit, 255)], axis=-1)  # Add RGBA channels
        self._plane = gl.GLImageItem(img_rgba)
        self._plane.translate(0, 0, z * self._z_aniso)
        self.addItem(self._plane)

    def setXY(self, xy: Tuple[int, int]):
        self._cursor_xy = xy
        self._cursor_pt.setData(pos=[xy])

    def closeEvent(self, evt):
        for widget in QApplication.instance().topLevelWidgets(): # Intercept the close evt and close the main application.
            if isinstance(widget, ZStackSegmentorApp):  
                widget.close()
                break  # Close only the first detected instance
        evt.accept()  

    def keyPressEvent(self, evt):
        if evt.key() in [Qt.Key_Left, Qt.Key_Right]:
            if evt.modifiers() & Qt.ShiftModifier:
                if not self._is_proposing:
                    delta = 1 if evt.key() == Qt.Key_Right else -1
                    z_ = self._z + delta
                    if 0 <= z_ < self._zmax:
                        self._start_proposing(self._z, z_)
            else:
                self.nav_key_pressed.emit(evt)
        else:
            super().keyPressEvent(evt)

    ''' Privates '''

    def _render_rois(self, rois: List[List[LabeledROI]]):
        # Remove existing meshes
        for mesh in self._meshes:
            self.removeItem(mesh)
        self._meshes = []
        # Compute 3d objects from 2d ROIs associated by their labels
        objs = dict()
        for z_i, level in enumerate(rois):
            z = z_i * self._z_aniso
            for roi in level:
                verts = roi.asPoly().vertices
                verts = np.hstack((verts, np.full((verts.shape[0], 1), z)))
                if roi.lbl in objs:
                    objs[roi.lbl].append((z, roi))
                else:
                    objs[roi.lbl] = [(z, roi)]
        # Triangulate the objects
        nfc = len(self.facecolors)
        for lbl, l_rois in objs.items():
            # Case 1: No solid object
            if len(l_rois) == 1:
                z, roi = l_rois[0]
                tri = Triangulation.from_polygon(roi.asPoly(n=20), z)
            # Case 2: Solid object
            else:
                verts = [(z, roi.asPoly(n=20).vertices) for _, roi in l_rois]
                verts = [np.hstack((v, np.full((v.shape[0], 1), z))) for z, v in verts]
                verts = np.concatenate(verts)
                tri = Triangulation.surface_3d(verts, method='advancing_front') # TODO: take options from GUI here
            colors = np.full((len(tri.simplices), 4), self.facecolors[lbl % nfc])
            md = gl.MeshData(vertexes=tri.pts, faces=tri.simplices, faceColors=colors)
            mesh = gl.GLMeshItem(meshdata=md, **self.mesh_opts)
            self.addItem(mesh)
            self._meshes.append(mesh)

    def _start_proposing(self, z1: int, z2: int):
        self._is_proposing = True
        self._zpair_propose = (z1, z2)
        self.start_proposing.emit()
        self._render_proposals()

    def _finish_proposing(self, accept: bool):
        self._is_proposing = False
        result = None
        if accept:
            result = (self._zpair_propose, self._proposed_rois) 
        self.finish_proposing.emit(result) # setROIs will be called by receiver
        self._proposed_rois = []

    def _render_proposals(self):
        self._proposed_map = self._compute_proposals()
        self._proposed_rois = [
            [r.relabel(self._proposed_map.get(r.lbl, r.lbl)) for r in self._rois[z]]
            for z in self._zpair_propose
        ]
        self._render_rois(self._proposed_rois)

    def _compute_proposals(self) -> Dict[int, int]:
        '''
        Returns a dictionary mapping from higher labels to lower labels for associated ROIs
        '''
        z1, z2 = self._zpair_propose
        rois1, rois2 = self._rois[z1], self._rois[z2]
        polys1, polys2 = [r.asPoly() for r in rois1], [r.asPoly() for r in rois2]
        
        # If either layer is empty, return empty mapping
        if not polys1 or not polys2:
            return {}
        
        # Build KDTree from centroids of first layer
        centroids1 = np.array([p.centroid() for p in polys1])
        tree = cKDTree(centroids1)
        
        # Get centroids and radii for second layer
        centroids2 = np.array([p.centroid() for p in polys2])
        radii1 = np.array([p.max_noncentrality() for p in polys1])
        radii2 = np.array([p.max_noncentrality() for p in polys2])
        
        # Find nearest neighbors and distances
        distances, indices = tree.query(centroids2)
        
        # Create label associations where distance is within threshold
        label_map = {}  # maps from higher label to lower label
        for poly2_idx, (poly1_idx, dist) in enumerate(zip(indices, distances)):
            threshold = min(radii1[poly1_idx], radii2[poly2_idx])
            
            if dist <= threshold:
                l1, l2 = rois1[poly1_idx].lbl, rois2[poly2_idx].lbl
                if l1 != l2:  # Only map if labels are different
                    label_map[min(l1, l2)] = max(l1, l2)
        
        return label_map

class ZStackSegmentorApp(ImageSegmentorApp):
    '''
    Segment 3-dimensional structures using iterated 2-dimensional segmentation and fusion/registration
    TODO:
    1. Add an auxiliary window which shows the current 3D structure alone
    2. Add a registration step between successive slices which allows:
        a. Manual association of ROIs to the same label
        b. Automated associations of ROIs
        c. Both in a kind of "proposer" mode as in the 2D case
        d. Proposer mode disables edits at the 2D level, but bubbles up selections to 3D for association
        e. Re-label ROIs when associated to min label
    3. Upon registration, re-compute and show the 3D structure
        c. Render position of cursor in z-plane (might need to bubble up from lower level)
    '''
    # TODO: better approach than overriding private
    def _pre_super_init(self):
        imgsize = np.array([self._img.shape[1], self._img.shape[2], self._img.shape[0]])
        voxsize = get_voxel_size(self._img_path, fmt='XYZ') # Physical voxel sizes
        self._viewer = ZStackObjectViewer(imgsize, voxsize)
        self._viewer.show()

        # Listeners
        self._creator.image_changed.connect(lambda img: self._viewer.setZ(self._z, img.T))
        self._viewer.nav_key_pressed.connect(self.keyPressEvent)
        self._viewer.start_proposing.connect(lambda: self.setEnabled(False))
        self._viewer.finish_proposing.connect(self._finish_proposing)

    def closeEvent(self, evt):
        if not self._viewer is None:
            self._viewer.close() 
        evt.accept()

    def refreshROIs(self):
        self._viewer.setROIs(self._rois)
        super().refreshROIs()

    ''' Privates ''' 

    def _finish_proposing(self, data: Optional[object]):
        edited = not (data is None)
        if edited:
            (z1, z2), (rois1, rois2) = data
            self._rois[z1] = rois1
            self._rois[z2] = rois2
        self.refreshROIs(push=edited)
