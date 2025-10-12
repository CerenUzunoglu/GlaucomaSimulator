import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QDateTime
from ui_form import Ui_Widget
from PySide6.QtWidgets import QTableWidgetItem
from matplotlib import pyplot as plt
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QHeaderView
import os

class GlaucomaSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.ui.intraocularPressureSlider.setMinimum(10)
        self.ui.intraocularPressureSlider.setMaximum(50)
        self.ui.intraocularPressureSlider.setValue(20)
        self.ui.intraocularPressureSlider.valueChanged.connect(self.slider_changed)

        self.original_image = None
        self.processed_image = None
        self.frames = []

        self.ui.loadButton.clicked.connect(self.upload_image)
        self.ui.saveButton.clicked.connect(self.save_image)
        self.ui.resetButton.clicked.connect(self.reset_image)
        self.ui.simulateButton.clicked.connect(self.simulate_and_save)
        self.ui.tableTimeline.setColumnCount(4)
        self.ui.tableTimeline.setHorizontalHeaderLabels(["Date", "Pressure", "Risk", "Note"])
        self.ui.tableTimeline.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableTimeline.verticalHeader().setVisible(False)
        self.ui.tableTimeline.setAlternatingRowColors(True)

        self.update_pressure(20)

        self.ui.showGraphButton.clicked.connect(self.show_iop_graph)

        self.ui.savePatientButton.clicked.connect(self.save_patient_data)
        self.ui.loadPatientButton.clicked.connect(self.load_patient_data)

        self.resize(1200, 800)
        self.ui.originalLabel.setScaledContents(True)
        self.ui.processedLabel.setScaledContents(True)

        self.ui.originalLabel.setAlignment(Qt.AlignCenter)
        self.ui.processedLabel.setAlignment(Qt.AlignCenter)
        self.ui.spinBoxCSF.setRange(0, 30)
        self.ui.spinBoxCSF.setValue(12)
        self.ui.spinBoxCSF.valueChanged.connect(self._on_csf_changed)

        self.ui.labelTPD.setText("TPD: — mmHg")

        self.ui.btnShowTPD.clicked.connect(self.show_tpd_graph)

        self._on_csf_changed()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image.", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.original_image = img
                self.processed_image = img.copy()
                self.show_image(self.original_image, self.ui.originalLabel)
                self.show_image(self.processed_image, self.ui.processedLabel)
                self.update_datetime()
            else:
                print("No valid image selected.")

    def simulate_glaucoma_advanced(self):
        if self.processed_image is None:
            print("No image loaded. Cannot simulate.")
            self.ui.dateTimeEdit.setDateTime(QDateTime.currentDateTime())
            return


        if self.processed_image is not None:
            iop_value = self.ui.intraocularPressureSlider.value()

            kernel_size = (iop_value // 5) * 2 + 1
            blurred_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            masked = self.create_circular_mask(blurred_image.shape[:2], iop_value)
            masked_image = cv2.bitwise_and(blurred_image, blurred_image, mask=masked)

            with_spots = self.apply_blind_spots(masked_image, iop_value)
            final_image = self.apply_floaters_opacity(with_spots, iop_value)

            self.processed_image = final_image
            self.show_image(self.processed_image, self.ui.processedLabel)
            self.update_datetime()

            #notes = self.ui.textEditNotes.toPlainText()
            #self.add_to_timeline(iop_value, notes)

    def create_circular_mask(self, shape, iop):
        h, w = shape
        center = (w // 2, h // 2)
        max_radius = min(h, w) // 2
        radius = max_radius - ((iop - 10) * 3)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, max(5, radius), 255, -1)
        return mask

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save", "", "JPEG Files (*.jpg);;PNG Files (*.png)"
            )
            if file_path:
                if not (file_path.endswith(".jpg") or file_path.endswith(".png")):
                    file_path += ".png"
                success = cv2.imwrite(file_path, self.processed_image)
                if success:
                    print(f"Image saved: {file_path}")
                    self.update_datetime()
                else:
                    print("Save failed.")
        else:
            print("No image to save.")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.ui.intraocularPressureSlider.setValue(20)
            self.show_image(self.processed_image, self.ui.processedLabel)
            self.update_datetime()
            self.ui.tableTimeline.setRowCount(0)
        else:
            print("Please load an image first.")

    def show_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.repaint()

    def update_pressure(self, value):
        print(f"Intraocular Pressure: {value} mmHg")
        if value < 22:
            risk_text = "🟢 Normal"
        elif value < 30:
            risk_text = "🟡 High"
        else:
            risk_text = "🔴 Severe Risk"

        self.ui.labelRiskLevel.setText(risk_text)

        tpd = self.get_tpd(value)
        self.ui.labelTPD.setText(f"TPD: {tpd:.1f} mmHg")

    def add_to_timeline(self, iop_value, notes="—"):
        row = self.ui.tableTimeline.rowCount()
        self.ui.tableTimeline.insertRow(row)

        now = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm")
        risk = self.ui.labelRiskLevel.text()

        self.ui.tableTimeline.setItem(row, 0, QTableWidgetItem(now))
        self.ui.tableTimeline.setItem(row, 1, QTableWidgetItem(str(iop_value)))
        self.ui.tableTimeline.setItem(row, 2, QTableWidgetItem(risk))
        self.ui.tableTimeline.setItem(row, 3, QTableWidgetItem(notes))

    def apply_blind_spots(self, image, iop_value):
        rng = np.random.default_rng(seed=iop_value)
        output = image.copy()
        h, w = output.shape[:2]
        spot_count = (iop_value - 10) // 4

        for _ in range(spot_count):
            radius = np.random.randint(10, 25)
            center = (
                rng.integers(radius, w - radius),
                rng.integers(radius, h - radius)
            )
            color = (0, 0, 0)
            cv2.circle(output, center, radius, color, -1)

        return output

    def apply_floaters_opacity(self, image, iop_value):
        overlay = image.copy()
        h, w = image.shape[:2]
        alpha = min(0.5, (iop_value - 10) / 100)

        for _ in range((iop_value - 10) // 5):
            center = (
                np.random.randint(0, w),
                np.random.randint(0, h)
            )
            radius = np.random.randint(20, 40)
            color = (180, 180, 180)
            cv2.circle(overlay, center, radius, color, -1)

        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def slider_changed(self, value):
        self.update_pressure(value)
        self.simulate_glaucoma_advanced()
        self.update_datetime()

    def show_iop_graph(self):
        rows = self.ui.tableTimeline.rowCount()

        if rows == 0:
            print("No data available.")
            return

        timestamps = []
        iop_values = []

        for row in range(rows):
            time_item = self.ui.tableTimeline.item(row, 0)
            iop_item = self.ui.tableTimeline.item(row, 1)

            if time_item and iop_item:
                timestamps.append(time_item.text())
                try:
                    iop_values.append(int(iop_item.text()))
                except ValueError:
                    iop_values.append(0)

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, iop_values, marker="o", linestyle="-", color="blue")
        plt.title("Intraocular Pressure (IOP) over Time")
        plt.xlabel("Date")
        plt.ylabel("IOP (mmHg)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def simulate_step(self, base_img, iop_value):
        kernel_size = (iop_value // 5) * 2 + 1
        blurred = cv2.GaussianBlur(base_img, (kernel_size, kernel_size), 0)
        mask = self.create_circular_mask(blurred.shape[:2], iop_value)
        masked = cv2.bitwise_and(blurred, blurred, mask=mask)
        with_spots = self.apply_blind_spots(masked, iop_value)
        final_img = self.apply_floaters_opacity(with_spots, iop_value)

        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm")
        cv2.putText(final_img, f"IOP: {iop_value} mmHg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(final_img, ts, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return final_img

    def simulate_and_save(self):
        if self.original_image is None:
            print("Please load an image first.")
            return

        self.frames = []
        img = self.original_image.copy()
        for iop in range(10, 51, 5):
            frame = self.simulate_step(img, iop)
            self.frames.append(frame.copy())
            self.processed_image = frame
            self.update_pressure(iop)
            self.add_to_timeline(iop, f"Auto simulation step {iop}")

        self.show_image(self.processed_image, self.ui.processedLabel)
        self.update_datetime()

        msg = QMessageBox(self)
        msg.setWindowTitle("Save Video Format")
        msg.setText("Which format do you want to save the video in?")
        mp4_btn = msg.addButton("MP4", QMessageBox.AcceptRole)
        avi_btn = msg.addButton("AVI", QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()

        if msg.clickedButton() == mp4_btn:
            self.save_video_mp4()
        elif msg.clickedButton() == avi_btn:
            self.save_video_avi()

    def save_video_mp4(self):
        if not self.frames:
            print("No MP4 frames available.")
            return
        height, width, _ = self.frames[0].shape
        file_path, _ = QFileDialog.getSaveFileName(self, "Save MP4 Video", "", "MP4 Files (*.mp4)")
        if not file_path:
            return
        if not file_path.endswith(".mp4"):
            file_path += ".mp4"

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for frame in self.frames:
            out.write(frame)
        out.release()
        print(f"MP4 video saved: {file_path}")

    def save_video_avi(self):
        if not self.frames:
            print("No AVI frames available.")
            return
        h, w, _ = self.frames[0].shape
        file_path, _ = QFileDialog.getSaveFileName(self, "Save AVI Video", "", "AVI Files (*.avi)")
        if not file_path:
            return
        if not file_path.endswith(".avi"):
            file_path += ".avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(file_path, fourcc, 5, (w, h))
        for frame in self.frames:
            out.write(frame)
        out.release()
        print(f"AVI video saved: {file_path}")

    def save_patient_data(self):
        name = self.ui.lineEditName.text()
        surname = self.ui.lineEditSurname.text()
        birth_date = self.ui.dateEditBirth.date().toString("yyyy-MM-dd")
        gender = self.ui.comboBoxGender.currentText()
        patient_id = self.ui.lineEditID.text()
        note = self.ui.textEditNotes.toPlainText()
        patient_data = {
            "name": name,
            "surname": surname,
            "birth_date": birth_date,
            "gender": gender,
            "id": patient_id,
            "note": note
        }
        with open("patient.json", "w", encoding="utf-8") as f:
            import json
            json.dump(patient_data, f, ensure_ascii=False, indent=4)
        print("Patient saved:", patient_data)

    def load_patient_data(self):
        import json, os
        if os.path.exists("patient.json"):
            with open("patient.json", "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            self.ui.lineEditName.setText(patient_data.get("name", ""))
            self.ui.lineEditSurname.setText(patient_data.get("surname", ""))
            bd = patient_data.get("birth_date", "2000-01-01")
            dt = QDateTime.fromString(bd, "yyyy-MM-dd")
            if not dt.isValid():
                dt = QDateTime.fromString("2000-01-01", "yyyy-MM-dd")
            self.ui.dateEditBirth.setDate(dt.date())
            self.ui.comboBoxGender.setCurrentText(patient_data.get("gender", ""))
            self.ui.lineEditID.setText(patient_data.get("id", ""))
            self.ui.textEditNotes.setPlainText(patient_data.get("note", ""))
            print("Patient loaded:", patient_data)
        else:
            print("No saved patient file found.")

    def update_datetime(self):
        current_datetime = QDateTime.currentDateTime()
        self.ui.dateTimeEdit.setDateTime(current_datetime)

    def slider_changed(self, value):
        if self.original_image is None:
            self.update_pressure(value)
            return
        self.update_pressure(value)
        self.simulate_glaucoma_advanced()
        self.update_datetime()

    def get_tpd(self, iop_value: float) -> float:
        csf = float(self.ui.spinBoxCSF.value())
        return float(iop_value) - csf

    def _on_csf_changed(self):
        iop = self.ui.intraocularPressureSlider.value()
        tpd = self.get_tpd(iop)
        self.ui.labelTPD.setText(f"TPD: {tpd:.1f} mmHg")

        if self.original_image is not None:
            self.simulate_glaucoma_advanced()

    def show_tpd_graph(self):
        rows = self.ui.tableTimeline.rowCount()
        if rows == 0:
            print("No data available.")
            return

        timestamps = []
        tpd_values = []

        csf_now = float(self.ui.spinBoxCSF.value())  # şu anki CSF

        for row in range(rows):
            time_item = self.ui.tableTimeline.item(row, 0)
            iop_item = self.ui.tableTimeline.item(row, 1)
            if time_item and iop_item:
                timestamps.append(time_item.text())
                try:
                    iop = float(iop_item.text())
                except ValueError:
                    iop = 0.0
                tpd_values.append(iop - csf_now)

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, tpd_values, marker="s", linestyle="--", label="TPD (mmHg)")
        plt.title("TPD over Time")
        plt.xlabel("Date")
        plt.ylabel("TPD (mmHg)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlaucomaSimulator()
    window.show()
    sys.exit(app.exec())
