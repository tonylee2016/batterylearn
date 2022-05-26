from pandas import dataframe

from pyens.elements import Base, Container


class Simulator(Container, Base):
    def __init__(self, name):
        Base.__init__(type="simulator", name=name)


class Datasoure(Base):
    def __init__(self, name, df: dataframe):
        Base.__init__(self, type="datasoure", name=name)
        self.df = df
