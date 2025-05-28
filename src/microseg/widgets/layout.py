import shortuuid
from qtpy import QtWidgets
from qtpy.QtCore import Qt

''' Layout widgets '''

class HLayout(QtWidgets.QHBoxLayout):
	'''
	HBoxLayout without margins
	'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setContentsMargins(0, 0, 0, 0)
		self.setSpacing(2)
		self.setAlignment(Qt.AlignmentFlag.AlignLeft)

class VLayout(QtWidgets.QVBoxLayout):
	'''
	VBoxLayout without margins
	'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setContentsMargins(0, 0, 0, 0)
		self.setSpacing(0)
		self.setAlignment(Qt.AlignmentFlag.AlignTop)

class StyledWidget(QtWidgets.QWidget):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._stylename = None

	def addStyle(self, style: str):
		# Lazily add style parameters
		if self._stylename is None:
			self._stylename = shortuuid.uuid()
			self.setAttribute(Qt.WA_StyledBackground)
			self.setObjectName(self._stylename)
		self.setStyleSheet(f'#{self._stylename} {{{style}}}')

	def clearStyle(self):
		self.setStyleSheet('')

class HLayoutWidget(StyledWidget):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._layout = HLayout()
		self.setLayout(self._layout)

	def addWidget(self, w: QtWidgets.QWidget):
		self._layout.addWidget(w)

	def insertWidget(self, i: int, w: QtWidgets.QWidget):
		self._layout.insertWidget(i, w)

	def addSpacing(self, s: int):
		self._layout.addSpacing(s)

	def addStretch(self):
		self._layout.addStretch()

class VLayoutWidget(StyledWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._layout = VLayout()
		self.setLayout(self._layout)

	def addWidget(self, w: QtWidgets.QWidget):
		self._layout.addWidget(w)

	def insertWidget(self, i: int, w: QtWidgets.QWidget):
		self._layout.insertWidget(i, w)

	def addSpacing(self, s: int):
		self._layout.addSpacing(s)

	def addStretch(self):
		self._layout.addStretch()

class VGroupBox(QtWidgets.QGroupBox):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._layout = VLayout()
		self.setLayout(self._layout)

	def addWidget(self, w: QtWidgets.QWidget):
		self._layout.addWidget(w)

	def insertWidget(self, i: int, w: QtWidgets.QWidget):
		self._layout.insertWidget(i, w)

	def addSpacing(self, s: int):
		self._layout.addSpacing(s)

	def addStretch(self):
		self._layout.addStretch()

class HGroupBox(QtWidgets.QGroupBox):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._layout = HLayout()
		self.setLayout(self._layout)

	def addWidget(self, w: QtWidgets.QWidget):
		self._layout.addWidget(w)

	def addSpacing(self, s: int):
		self._layout.addSpacing(s)

	def addStretch(self):
		self._layout.addStretch()

	def insertWidget(self, i: int, w: QtWidgets.QWidget):
		self._layout.insertWidget(i, w)