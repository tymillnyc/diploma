import os
import pickle
import sys
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal
from preprocess import prepare_data, prepare_web_page
import bert_model

class Thread1(QThread):
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Thread1, self).__init__()

    def run(self):
        ui.progress_bar.show()
        ui.result_label.show()
        ui.threePageRunButton.setEnabled(False)

        path = ui.threePageFilePath.text()

        train_texts, valid_texts, train_labels, \
        valid_labels, category_index, reverse_category_index = prepare_data(path, thread=self)

        self._signal.emit('Загрузка модели и подготовка к обучению')
        self._signal.emit('0')
        bert = bert_model.BertClassifier('DeepPavlov/rubert-base-cased',
                                         'DeepPavlov/rubert-base-cased',
                                         category_index,
                                         reverse_category_index)
        self._signal.emit('50')
        bert.preparation(train_texts, train_labels, valid_texts, valid_labels)
        self._signal.emit('100')

        bert.train(self, ui)

        ui.threePageRunButton.setEnabled(True)
        ui.progress_bar.hide()
        ui.result_label.setText("Обучение завершено.  Модель сохранена.")


class Thread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()

    def run(self):
        ui.twoPageRunButton.setEnabled(False)

        path = ui.twoPageFilePath.text()

        bert = bert_model.BertClassifier(tokenizer_path=ui.twoModelPath.text())

        df = pd.read_csv(path, index_col=0)

        with open(ui.twoModelPath.text() + '/' + 'saved_dictionary.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)

        ui.progress_bar1.show()
        ui.result_label1.show()

        for index, row in df.iterrows():

            self._signal.emit('Извлечение текстовых данных и предобработка ' + str(index+1) + ' страницы')
            self._signal.emit('0')

            text = prepare_web_page(row['url'])

            self._signal.emit('Предсказание  ' + str(index+1) + ' страницы')
            self._signal.emit('50')

            predict_label = bert.predict(text)

            rowCount = ui.twoPageResultTable.rowCount()
            ui.twoPageResultTable.insertRow(rowCount)

            ui.twoPageResultTable.setItem(rowCount, 0, QTableWidgetItem(str(row['url'])))
            ui.twoPageResultTable.setItem(rowCount, 1, QTableWidgetItem(loaded_dict[predict_label]))

            self._signal.emit('100')

        ui.twoPageRunButton.setEnabled(True)

        ui.progress_bar1.hide()
        ui.result_label1.setText("Предсказание завершено.")


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UiMainWindow, self).__init__()
        self.setupUi()

    def setupUi(self):
        self.resize(562, 480)
        self.setStyleSheet("")
        self.centralWidget = QtWidgets.QWidget(self)

        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 562, 480))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setEnabled(True)

        self.URLAdress = QtWidgets.QLineEdit(self.tab1)
        self.URLAdress.setGeometry(QtCore.QRect(10, 30, 480, 21))
        self.URLAdress.setText("")
        self.URLAdress.setObjectName("URLAdress")
        self.URLLabel = QtWidgets.QLabel(self.tab1)
        self.URLLabel.setGeometry(QtCore.QRect(10, 10, 31, 20))
        self.URLLabel.setObjectName("label")
        self.onePageRunButton = QtWidgets.QPushButton(self.tab1)
        self.onePageRunButton.setGeometry(QtCore.QRect(170, 120, 113, 32))
        self.onePageResult = QtWidgets.QTextEdit(self.tab1)
        self.onePageResult.setGeometry(QtCore.QRect(10, 190, 531, 101))
        self.onePageResult.setReadOnly(True)
        self.resultLabel1 = QtWidgets.QLabel(self.tab1)
        self.resultLabel1.setGeometry(QtCore.QRect(10, 170, 531, 20))
        self.modelPath = QtWidgets.QLineEdit(self.tab1)
        self.modelPath.setGeometry(QtCore.QRect(10, 90, 480, 21))
        self.modelPath.setText("")
        self.modelPath.setObjectName("ModelPath")
        self.modelPath.setReadOnly(True)
        self.modelLabel = QtWidgets.QLabel(self.tab1)
        self.modelLabel.setGeometry(QtCore.QRect(10, 70, 480, 20))
        self.modelLabel.setObjectName("label")
        self.onePageFileDialogButton = QtWidgets.QToolButton(self.tab1)
        self.onePageFileDialogButton.setGeometry(QtCore.QRect(497, 90, 40, 21))
        self.onePageFileDialogButton.setStyleSheet("")


        self.tabWidget.addTab(self.tab1, "")





        self.tab2 = QtWidgets.QWidget()
        self.tab2.setEnabled(True)

        self.twoPageFilePath = QtWidgets.QLineEdit(self.tab2)
        self.twoPageFilePath.setGeometry(QtCore.QRect(10, 30, 480, 21))
        self.twoPageFilePath.setReadOnly(True)
        self.pathListURLLabel = QtWidgets.QLabel(self.tab2)
        self.pathListURLLabel.setGeometry(QtCore.QRect(10, 10, 400, 20))
        self.twoPageRunButton = QtWidgets.QPushButton(self.tab2)
        self.twoPageRunButton.setGeometry(QtCore.QRect(170, 120, 113, 32))
        self.twoPageResultTable = QtWidgets.QTableWidget(self.tab2)
        self.twoPageResultTable.setGeometry(QtCore.QRect(10, 240, 531, 200))
        self.twoPageResultTable.setColumnCount(2)
        self.twoPageSaveButton = QtWidgets.QPushButton(self.tab2)
        self.twoPageSaveButton.setGeometry(QtCore.QRect(290, 120, 113, 32))
        self.twoPageFileDialogButton = QtWidgets.QToolButton(self.tab2)
        self.twoPageFileDialogButton.setGeometry(QtCore.QRect(497, 30, 40, 21))
        self.twoPageFileDialogButton.setStyleSheet("")
        self.resultLabel2 = QtWidgets.QLabel(self.tab2)
        self.resultLabel2.setGeometry(QtCore.QRect(10, 220, 71, 16))
        self.twoModelPath = QtWidgets.QLineEdit(self.tab2)
        self.twoModelPath.setGeometry(QtCore.QRect(10, 90, 480, 21))
        self.twoModelPath.setText("")
        self.twoModelPath.setObjectName("ModelPath")
        self.twoModelPath.setReadOnly(True)
        self.twoModelLabel = QtWidgets.QLabel(self.tab2)
        self.twoModelLabel.setGeometry(QtCore.QRect(10, 70, 480, 20))
        self.twoModelLabel.setObjectName("label")
        self.twoPageFileDialogButton2 = QtWidgets.QToolButton(self.tab2)
        self.twoPageFileDialogButton2.setGeometry(QtCore.QRect(497, 90, 40, 21))
        self.twoPageFileDialogButton2.setStyleSheet("")
        self.progress_bar1 = QtWidgets.QProgressBar(self.tab2)
        self.progress_bar1.setGeometry(QtCore.QRect(10, 175, 531, 20))
        self.progress_bar1.setProperty("value", 0)
        self.result_label1 = QtWidgets.QLabel(self.tab2)
        self.result_label1.setGeometry(QtCore.QRect(10, 155, 531, 16))
        self.progress_bar1.hide()
        self.result_label1.hide()
        self.tabWidget.addTab(self.tab2, "")

        self.columnLabels = ['URL', 'Category']
        self.twoPageResultTable.setHorizontalHeaderLabels(self.columnLabels)
        header = self.twoPageResultTable.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        self.tab3 = QtWidgets.QWidget()
        self.tab3.setEnabled(True)
        self.threePageFilePath = QtWidgets.QLineEdit(self.tab3)
        self.threePageFilePath.setGeometry(QtCore.QRect(10, 30, 480, 21))
        self.threePageFilePath.setReadOnly(True)
        self.pathLabel = QtWidgets.QLabel(self.tab3)
        self.pathLabel.setGeometry(QtCore.QRect(10, 10, 400, 20))
        self.threePageLogs = QtWidgets.QTextEdit(self.tab3)
        self.threePageLogs.setReadOnly(True)
        self.threePageLogs.setGeometry(QtCore.QRect(10, 190, 531, 250))
        self.logsLabel = QtWidgets.QLabel(self.tab3)
        self.logsLabel.setGeometry(QtCore.QRect(10, 170, 531, 16))

        self.threePageRunButton = QtWidgets.QPushButton(self.tab3)
        self.threePageRunButton.setGeometry(QtCore.QRect(170, 70, 113, 32))
        self.threePageFileDialogButton = QtWidgets.QToolButton(self.tab3)
        self.threePageFileDialogButton.setGeometry(QtCore.QRect(497, 30, 40, 21))
        self.threePageFileDialogButton.setStyleSheet("")
        self.tabWidget.addTab(self.tab3, "")
        self.progress_bar = QtWidgets.QProgressBar(self.tab3)
        self.progress_bar.setGeometry(QtCore.QRect(10, 130, 531, 20))
        self.progress_bar.setProperty("value", 0)
        self.result_label = QtWidgets.QLabel(self.tab3)
        self.result_label.setGeometry(QtCore.QRect(10, 110, 531, 16))
        self.progress_bar.hide()
        self.result_label.hide()

        self.setCentralWidget(self.centralWidget)
        self.retranslateUi(self)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.addFunctionsToButtons()

        self.msg = QtWidgets.QMessageBox()

        self.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификатор веб-страниц"))
        self.URLLabel.setText(_translate("MainWindow", "URL"))
        self.onePageRunButton.setText(_translate("MainWindow", "Предсказать"))
        self.resultLabel1.setText(_translate("MainWindow", "Результат"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Одна веб-страница"))
        self.pathListURLLabel.setText(_translate("MainWindow", "Путь к файлу со списком URL-адресов(.csv)"))
        self.twoPageRunButton.setText(_translate("MainWindow", "Предсказать"))
        self.twoPageSaveButton.setText(_translate("MainWindow", "Сохранить"))
        self.twoPageFileDialogButton.setText(_translate("MainWindow", "..."))
        self.onePageFileDialogButton.setText(_translate("MainWindow", "..."))
        self.twoPageFileDialogButton2.setText(_translate("MainWindow", "..."))
        self.resultLabel2.setText(_translate("MainWindow", "Результат"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), _translate("MainWindow", "Список веб-страниц"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "Обучение"))
        self.pathLabel.setText(_translate("MainWindow", "Путь к файлу с данными для обучения(.csv)"))
        self.logsLabel.setText(_translate("MainWindow", "Промежуточные результаты обучения"))
        self.threePageRunButton.setText(_translate("MainWindow", "Обучить"))

        self.modelLabel.setText(_translate("MainWindow", "Путь к папке с моделью"))
        self.twoModelLabel.setText(_translate("MainWindow", "Путь к папке с моделью"))
        self.threePageFileDialogButton.setText(_translate("MainWindow", "..."))
        self.result_label.setText(_translate("MainWindow", ""))

    def addFunctionsToButtons(self):
        self.onePageRunButton.clicked.connect(self.handleOnePageRunBtn)
        self.onePageFileDialogButton.clicked.connect(self.launchDialog2)
        self.twoPageFileDialogButton2.clicked.connect(self.launchDialog3)
        self.twoPageFileDialogButton.clicked.connect(self.launchDialog)
        self.twoPageRunButton.clicked.connect(self.handleTwoPageRunBtn)
        self.twoPageSaveButton.clicked.connect(self.handleTwoPageSaveBtn)
        # self.threePageSaveButton.clicked.connect()
        self.threePageFileDialogButton.clicked.connect(self.launchDialog1)
        self.threePageRunButton.clicked.connect(self.handleThreePageRunBtn)

    def handleThreePageRunBtn(self):
        filePath = self.threePageFilePath.text()
        if filePath != '':
            if filePath.endswith('.csv'):
                if os.path.exists(filePath):
                    self.thread = Thread1()
                    self.thread._signal.connect(self.acceptSignal1)
                    self.thread.start()
                else:
                    self.handleError('Такого пути не существует', self.popupAction2)
            else:
                self.handleError('Файл некорректного формата', self.popupAction2)

    def handleTwoPageSaveBtn(self):
        pass
        # self.twoPageResultTable.clearContents()
        # self.twoPageResultTable.setRowCount(0)

    def handleTwoPageRunBtn(self):
        filePath = self.twoPageFilePath.text()
        if filePath != '':
            if filePath.endswith('.csv'):
                if os.path.exists(filePath):

                    if self.twoModelPath.text() == '':
                        self.handleError('Вы ничего не ввели!', self.popupAction1)

                    else:
                        self.thread = Thread()
                        self.thread._signal.connect(self.acceptSignal)
                        self.thread.start()
                else:
                    self.handleError('Такого пути не существует', self.popupAction1)
            else:
                self.handleError('Файл некорректного формата', self.popupAction1)

    def acceptSignal(self, message):
        if message.isdigit():
            self.progress_bar1.setValue(int(message))
        else:
            self.result_label1.setText(str(message))

    def acceptSignal1(self, message):
        if message.isdigit():
            self.progress_bar.setValue(int(message))
        else:
            self.result_label.setText(str(message))

    def handleOnePageRunBtn(self):
        url = self.URLAdress.text()
        if url != '':
                path = self.modelPath.text()
                with open(path + '/' + 'saved_dictionary.pkl', 'rb') as f:
                    loaded_dict = pickle.load(f)

                bert = bert_model.BertClassifier(tokenizer_path=path)
                text = prepare_web_page(url)
                predict_label = bert.predict(text)
                category = loaded_dict[predict_label]
                self.onePageResult.setText('Веб-страница: ' + str(url) + '\n' + 'Ее категория: ' + category)


    def handleError(self, errorText, handleBtnsInError):
        error = QMessageBox()
        error.setWindowTitle('Ошибка')
        error.setText(errorText)
        error.setIcon(QMessageBox.Warning)
        error.setStandardButtons(QMessageBox.Reset | QMessageBox.Ok)
        error.buttonClicked.connect(handleBtnsInError)
        error.exec_()

    def popupAction2(self, button):
        if button.text() == 'Reset':
            self.threePageFilePath.setText('')

    def popupAction1(self, button):
        if button.text() == 'Reset':
            self.twoPageFilePath.setText('')

    def popupAction(self, button):
        if button.text() == 'Reset':
            self.URLAdress.setText('')

    def launchDialog3(self):
        response = QFileDialog.getExistingDirectory(self.centralWidget,
                                               'Select folder')
        if response != '':
            self.twoModelPath.setText(response)

    def launchDialog2(self):
        response = QFileDialog.getExistingDirectory(self.centralWidget,
                                               'Select folder')
        if response != '':
            self.modelPath.setText(response)

    def launchDialog(self):
        response = QFileDialog.getOpenFileName(self.centralWidget,
                                               'Open file',
                                               directory=os.getcwd(),
                                               filter='File format (*.csv)',
                                               initialFilter='File format (*.csv)')
        if response[0] != '':
            self.twoPageFilePath.setText(response[0])

    def launchDialog1(self):
        response = QFileDialog.getOpenFileName(self.centralWidget,
                                               'Open file',
                                               directory=os.getcwd(),
                                               filter='File format (*.csv)',
                                               initialFilter='File format (*.csv)')
        if response[0] != '':
            self.threePageFilePath.setText(response[0])


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    sys.exit(app.exec_())