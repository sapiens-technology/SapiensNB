class SapiensNB:
    def __init__(self):
        from .core import SapiensNB
        self.__sapiensnb = SapiensNB()
    def fit(self, inputs=[], outputs=[]): self.__sapiensnb.fit(inputs=inputs, outputs=outputs)
    def saveModel(self, path=''): self.__sapiensnb.saveModel(path=path)
    def loadModel(self, path=''): self.__sapiensnb.loadModel(path=path)
    def transferLearning(self, transmitter_path='', receiver_path='', rescue_path=''): self.__sapiensnb.transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, rescue_path=rescue_path)
    def predict(self, inputs=[]): return self.__sapiensnb.predict(inputs=inputs)
    def test(self, inputs=[], outputs=[]): return self.__sapiensnb.test(inputs=inputs, outputs=outputs)
