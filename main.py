#! /bin/python
# -*- coding: utf-8 -*-

from sys     import argv, stderr
from os      import environ
from os.path import split as splitpath, abspath, join

environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import cv2
import numpy  as np
import pandas as pd

from tensorflow import keras as ks
from argparse   import ArgumentParser, Namespace

SIZE_IMAGE = 60

ABS_PATH = splitpath(abspath(argv[0]))[0]

class Main:
    def run(self) -> int:
        try:
            self._runWithException()
        except KeyboardInterrupt:
            print("\n[INFO] The user triggered an interrupt! Termination of work.")
            return 0
        except Exception as err:
            print(f"\n[ERROR] {err}", file=stderr)
            return -1
    
    def _runWithException(self) -> None:
        args = self.parseArgument()

        self.__modelName   = args.model
        self.__datasetName = args.dataset

        if args.train:
            self.aiTrain(args.epochs_count, args.batch_size)
        elif args.predict:
            self.aiPredict()
        else:
            raise RuntimeError("No action selected! Please use -t/--train or -p/--predict")
        
        return 0
        
    def parseArgument(self) -> Namespace:
        argsParser = ArgumentParser()

        argsParser.add_argument("-t", "--train", action="store_true", help="train AI on a train dataset")
        argsParser.add_argument("-e", "--epochs-count", type=int, default=50, help="training epochs count")
        argsParser.add_argument("-b", "--batch-size",   type=int, default=32, help="training batch size")
        
        argsParser.add_argument("-p", "--predict", action="store_true", help="predict classes from test dataset")

        argsParser.add_argument("-m", "--model",   type=str, default="model", help="set model name")
        argsParser.add_argument("-d", "--dataset", type=str, help="set dataset name")

        return argsParser.parse_args()
    
    def aiTrain(self, epochs_count: int, batch_size: int) -> None:
        if not self.__datasetName:
            self.__datasetName = "train"
        
        (pathsToImages, classes) = self.readDatasetFile()
        images = self.loadImages(pathsToImages)

        normImages  = self.normalizationImages(images)
        normClasses = self.normalizationClasses(classes)

        imagesTest  = normImages[4000:]
        classesTest = normClasses[4000:]

        imagesTrain  = normImages[:4000]
        classesTrain = normClasses[:4000]

        model = self.generateModel()
        model.summary()

        model.fit(imagesTrain, classesTrain, epochs=epochs_count, batch_size=batch_size, 
                    validation_split=0.1)
        
        _, testAccuracy = model.evaluate(imagesTest, classesTest)
        print(f"Training Accuracy: {testAccuracy}")

        model.save(join(ABS_PATH, self.__modelName))
    
    def aiPredict(self) -> None:
        if not self.__datasetName:
            self.__datasetName = "test"
        
        (pathsToImages, _) = self.readDatasetFile()

        images = self.loadImages(pathsToImages)
        normImages = self.normalizationImages(images)

        model = ks.models.load_model(join(ABS_PATH, self.__modelName))

        predicts = model.predict(normImages)
        self.savePredict(predicts)
    
    def readDatasetFile(self) -> tuple[list[str], np.array]:
        datasetAbsPath = join(ABS_PATH, f"{self.__datasetName}/{self.__datasetName}")
        datasetFile    = pd.read_csv(f"{datasetAbsPath}.csv")

        absPathsToImages = datasetFile["ID_img"].map(lambda fileName: f"{datasetAbsPath}/{fileName}")
        classes          = np.array(datasetFile["class"])

        return (absPathsToImages, classes)
    
    def savePredict(self, predicts: np.array) -> None:
        datasetAbsPath = join(ABS_PATH, f"{self.__datasetName}/{self.__datasetName}")
        datasetFile    = pd.read_csv(f"{datasetAbsPath}.csv")

        for i, predict in enumerate(predicts):
            objClass = np.argmax(predict)
            datasetFile.at[i, "class"] = float(objClass)
        
        datasetFile.to_csv(f"{datasetAbsPath}.csv", index=False)
    
    def loadImages(self, paths: list[str]) -> list[cv2.Mat]:
        loadedImages = []
        countImages  = len(paths)
        countNumbers  = self.getCountNumber(countImages)
        
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            img = cv2.GaussianBlur(img, (3, 3), 1)
            img = cv2.resize(img, (SIZE_IMAGE, SIZE_IMAGE))

            loadedImages.append(img)
            print(f"Loading images... ({(i + 1):.>{countNumbers}}/{countImages:.>{countNumbers}})", end="\r")
        print()

        return loadedImages
    
    def getCountNumber(self, num: int) -> int:
        count = 0
        while num > 0:
            num //= 10
            count += 1
        
        return count
    
    def normalizationImages(self, images: np.array) -> np.array:
        return np.array(images, dtype="float32") / 255
    
    def normalizationClasses(self, classes: list) -> np.array:
        normClasses = []
        for i in classes:
            a = [0.] * 8
            a[int(i)] = 1.0

            normClasses.append(a)
        
        normClasses = np.array(normClasses)
        return normClasses
    
    def generateModel(self) -> ks.Sequential:
        model = ks.Sequential([
            ks.layers.Conv2D(32, (5, 5), input_shape=(SIZE_IMAGE, SIZE_IMAGE, 3), padding="same"),
            ks.layers.Activation("relu"),
            ks.layers.Dropout(0.05),
            ks.layers.BatchNormalization(),
            ks.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            ks.layers.Conv2D(64, (5, 5)),
            ks.layers.Activation("relu"),
            ks.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            ks.layers.Conv2D(64, (5, 5)),
            ks.layers.Activation("relu"),
            ks.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            ks.layers.Flatten(),

            ks.layers.Dense(64, activation="relu"),
            ks.layers.Dense(32, activation="relu"),

            ks.layers.Dense(8, activation="softmax")
        ])

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        return model

if __name__ == "__main__":
    main = Main()
    exit(main.run())