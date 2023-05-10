import os
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
import glob
from augmentation_techniques.back_translation import back_translation
from augmentation_techniques.contextual_insertion import contextual_insertion
from augmentation_techniques.contextual_replacement import contextual_replacement
from augmentation_techniques.eda import synonym_replacement
from augmentation_techniques.paraphraser import paraphraser
import pandas as pd


def apply_augm_method(thread, path, method):
    augm_texts = []

    output_path = path + '_output/'

    thread._signal.emit('Загрузка текстов')
    thread._signal.emit('0')

    texts = read_list_files(path, thread)

    if len(texts) != 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    thread._signal.emit('100')

    thread._signal.emit('Аугментация текстов')

    thread._signal.emit('0')

    for index, text in enumerate(texts):
        thread._signal.emit('Аугментация текстов' + '  (' + str(index+1) + '/' + str(len(texts)) + ')')
        if method == 'bt':
            text = back_translation([text], [-1])[0][0]
        elif method == 'pp':
            text = paraphraser([text], [-1])[0][0]
        elif method == 'cim':
            text = contextual_replacement([text], [-1])[0][0]
        elif method == 'cin':
            text = contextual_insertion([text], [-1])[0][0]
        elif method == 'eda':
            text = synonym_replacement(text)

        augm_texts.append(text.lower())

        thread._signal.emit(str(round((index + 1) / len(texts) * 100)))

    thread._signal.emit('100')

    return augm_texts


def read_file(file):
    with open(file, 'rt') as fd:
        lines = fd.readlines()
    return ('\n'.join(lines)).lower()


def read_list_files(folder_path, thread):
    result = []

    folder_path += "" if folder_path[-1] == "/" else "/"

    txt_files = glob.glob(folder_path + "*.txt")

    for index, txt_file in enumerate(sorted(txt_files)):
        text = read_file(txt_file)
        result.append(text)

        thread._signal.emit(str(round((index + 1) / len(txt_files) * 100)))

    return result


class Thread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()

    def run(self):
        ui.progress_bar.show()
        ui.result_label.show()
        ui.runButton.setEnabled(False)

        path = ui.foldLine.text()
        output_path = path + '_output/'

        augm_texts = apply_augm_method(self, path, ui.method)
        ui.runButton.setEnabled(True)
        ui.progress_bar.hide()

        for index, text in enumerate(augm_texts):
            append_file = open(output_path + 'augm_text' + str(index+1), 'w')
            append_file.write(text)
            append_file.close()

        df = pd.DataFrame({'augm texts': augm_texts})
        df.to_csv(output_path + 'augm_texts.csv')
        msg = "Аугментация завершена.  Аугментированные тексты сохранены в папку.  Ее путь: \n" + output_path
        self._signal.emit(msg)


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):

        super(UiMainWindow, self).__init__()
        self.method = None
        self.setupUi()

    def setupUi(self):

        self.resize(562, 350)
        self.setStyleSheet("")
        self.centralWidget = QtWidgets.QWidget(self)

        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 562, 350))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)

        self.foldLine = QtWidgets.QLineEdit(self)
        self.foldLine.setGeometry(QtCore.QRect(10, 30, 480, 21))
        self.foldLine.setReadOnly(True)

        self.foldLabel = QtWidgets.QLabel(self)
        self.foldLabel.setGeometry(QtCore.QRect(10, 10, 250, 20))

        self.foldDialogButton = QtWidgets.QToolButton(self)
        self.foldDialogButton.setGeometry(QtCore.QRect(497, 30, 40, 21))
        self.foldDialogButton.setStyleSheet("")

        self.nameLabel = QtWidgets.QLabel(self)
        self.nameLabel.setGeometry(QtCore.QRect(10, 75, 250, 20))

        self.back_trnl_radioButton = QtWidgets.QRadioButton(self)
        self.back_trnl_radioButton.setGeometry(QtCore.QRect(18, 100, 200, 20))

        self.back_trnl_radioButton.toggled.connect(self.back_trnl_selected)

        self.paraphr_radioButton = QtWidgets.QRadioButton(self)
        self.paraphr_radioButton.setGeometry(QtCore.QRect(18, 120, 200, 20))

        self.paraphr_radioButton.toggled.connect(self.paraphr_selected)

        self.context_implace_radioButton = QtWidgets.QRadioButton(self)
        self.context_implace_radioButton.setGeometry(QtCore.QRect(18, 140, 300, 20))

        self.context_implace_radioButton.toggled.connect(self.context_implace_selected)

        self.context_insert_radioButton = QtWidgets.QRadioButton(self)
        self.context_insert_radioButton.setGeometry(QtCore.QRect(18, 160, 300, 20))

        self.context_insert_radioButton.toggled.connect(self.context_insert_selected)

        self.eda_radioButton = QtWidgets.QRadioButton(self)
        self.eda_radioButton.setGeometry(QtCore.QRect(18, 180, 450, 20))

        self.eda_radioButton.toggled.connect(self.eda_selected)

        self.runButton = QtWidgets.QPushButton(self)
        self.runButton.setGeometry(QtCore.QRect(170, 220, 190, 32))

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(QtCore.QRect(10, 300, 531, 20))
        self.progress_bar.setProperty("value", 0)

        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setGeometry(QtCore.QRect(10, 260, 560, 40))

        self.progress_bar.hide()
        self.result_label.hide()

        self.setCentralWidget(self.centralWidget)
        self.retranslateUi(self)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.addFunctionsToButtons()

        self.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Аугментация текстовых данных"))
        self.foldLabel.setText(_translate("MainWindow", "Путь к папке с текстами(.txt):"))
        self.runButton.setText(_translate("MainWindow", "Начать аугментацию"))
        self.foldDialogButton.setText(_translate("MainWindow", "..."))
        self.back_trnl_radioButton.setText(_translate("MainWindow", "Обратный перевод"))
        self.paraphr_radioButton.setText(_translate("MainWindow", "Перефразирование"))
        self.context_implace_radioButton.setText(_translate("MainWindow", "Контекстная замена случайного токена"))
        self.context_insert_radioButton.setText(_translate("MainWindow", "Контекстная вставка случайного токена"))
        self.eda_radioButton.setText(_translate("MainWindow", "Easy Data Augmentation(замена слов синонинами(word2vec))"))
        self.nameLabel.setText(_translate("MainWindow", "Выбор метода аугментации:"))

    def addFunctionsToButtons(self):

        self.runButton.clicked.connect(self.handleRunBtn)
        self.foldDialogButton.clicked.connect(self.launchDialog)

    def launchDialog(self):

        response = QFileDialog.getExistingDirectory(self.centralWidget,
                                                    'Select folder')
        if response != '':
            self.foldLine.setText(response)

    def handleRunBtn(self):

        path = self.foldLine.text()
        if path != '':
            if self.method != None:
                self.thread = Thread()
                self.thread._signal.connect(self.acceptSignal)
                self.thread.start()
            else:
                self.handleError('Не выбран метод аугментации!')
        else:
            self.handleError('Не задан путь к папке с текстами!')

    def handleError(self, errorText):

        error = QMessageBox()
        error.setWindowTitle('Ошибка')
        error.setText(errorText)
        error.setIcon(QMessageBox.Warning)
        error.setStandardButtons(QMessageBox.Ok)
        error.exec_()

    def acceptSignal(self, message):

        if message.isdigit():
            self.progress_bar.setValue(int(message))
        else:
            self.result_label.setText(str(message))

    def back_trnl_selected(self):
        self.method = 'bt'

    def paraphr_selected(self):

        self.method = 'pp'

    def context_implace_selected(self):

        self.method = 'cim'

    def context_insert_selected(self):

        self.method = 'cin'

    def eda_selected(self):

        self.method = 'eda'


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    sys.exit(app.exec_())