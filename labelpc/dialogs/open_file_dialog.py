from PyQt5.QtWidgets import QDialog, QInputDialog, QMessageBox, QLineEdit, QDialogButtonBox, QFormLayout

class OpenFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.maxPoints = QLineEdit(self)
        self.meshSize = QLineEdit(self)
        self.thickness = QLineEdit(self)
        self.maxPoints.setText("1000000")
        self.meshSize.setText("0.1")
        self.thickness.setText("0.2")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self);

        layout = QFormLayout(self)
        layout.addRow("Max Points", self.maxPoints)
        layout.addRow("Mesh Size", self.meshSize)
        layout.addRow("Thickness", self.thickness)
        layout.addWidget(buttonBox)


        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if not len(self.maxPoints.text()) or int(self.maxPoints.text()) <= 0:
            self.maxPoints.setText("10000000000")
        return (int(self.maxPoints.text()), float(self.meshSize.text()), float(self.thickness.text()))