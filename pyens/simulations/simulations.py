import pyens.elements as elm


class simdata:
    pass


class simulation:
    def __init__(self):
        self.elements = []
        self.elements_pairs = []

    def add(self, element: elm.Dynamical):
        self.elements.append(element)

    def pair(self, pair_setting: tuple):

        # (flow_from, flow_to) = pair_setting

        for j in pair_setting:
            if j not in self.elements.index():
                raise ValueError(
                    "pair setting pointing to an element that does not exist !"
                )

        self.elements_pairs.append(pair_setting)

    # todo: implement a dynamical system connection method

    # def prepare_sim(self,
    #                 x,):
    #     for pair in self.elements_pairs:
    #         # data flow pair[0]->pair[1]
