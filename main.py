from math import log
from numbers import Real
import random
from typing import Self
import graphviz


class Value:
    def __init__(self: Self, value: Real, grad: Real = 1, name: str = ""):
        self.name = name
        self.value = value
        self.grad = grad
        self.children = []
        self._backward = lambda self: None

    def __str__(self) -> str:
        return f"{self.name}"

    def __add__(self, right: Self):
        result = Value(
            self.value + right.value, name=f"({self.__str__()} + {right.__str__()})"
        )

        def _backward(self):
            if self.children == []:
                return
            left, right = self.children
            left.grad = self.grad
            right.grad = self.grad

            left._backward(left)
            right._backward(right)

        result._backward = _backward
        result.children.append(self)
        result.children.append(right)

        return result

    def __mul__(self, right: Self):
        result = Value(
            self.value * right.value, name=f"({self.__str__()} * {right.__str__()})"
        )

        def _backward(self):
            if self.children == []:
                return
            left, right = self.children
            left.grad = right.value * self.grad
            right.grad = left.value * self.grad

            left._backward(left)
            right._backward(right)

        result._backward = _backward

        result.children.append(self)
        result.children.append(right)

        return result

    def __sub__(self, right: Self):
        result = Value(
            self.value - right.value, name=f"({self.__str__()} - {right.__str__()})"
        )

        def _backward(self):
            if self.children == []:
                return
            left, right = self.children
            left.grad = self.grad
            right.grad = -self.grad

            left._backward(left)
            right._backward(right)

        result._backward = _backward

        result.children.append(self)
        result.children.append(right)

        return result

    def __pow__(self, right: Self):
        result = Value(
            self.value**right.value, name=f"({self.__str__()} ** {right.__str__()})"
        )

        def _backward(self):
            if self.children == []:
                return
            left, right = self.children

            left.grad = self.grad * right.value * (left.value ** (right.value - 1))
            right.grad = self.grad * (left.value**right.value) * log(left.value)

            left._backward(left)
            right._backward(right)

        result._backward = _backward

        result.children.append(self)
        result.children.append(right)

        return result

    def relu(self):
        result = Value(0, name=f"relu({self.__str__()})")
        if self.value > 0:
            result = Value(self.value, name=f"relu({self.__str__()})")

        def _backward(self):
            if self.children == []:
                return
            child = self.children[0]
            child.grad = self.grad if self.value > 0 else 0

            if child._backward is not None:
                child._backward(child)

        result._backward = _backward

        result.children.append(self)

        return result

    def render_nodes(self, dot):
        dot.node(self.__str__())

        for child in self.children:
            child.render_nodes(dot)

    def render_edges(self, dot):

        for child in self.children:
            child.render_edges(dot)
            dot.edge(self.__str__(), child.__str__(), label=f"{child.grad:.2f}")

    def render(self, dot):
        self.render_nodes(dot)
        self.render_edges(dot)

    def backprop(self):
        self._backward(self)


import pandas as pd

csv = pd.read_csv("housing.csv")


head_csv = csv.head(1)


iterations = 40


x = []
x.append(head_csv["area"].values[0] / 1000)
x.append(head_csv["bedrooms"].values[0])
x.append(head_csv["bathrooms"].values[0])

w = [Value((2 * (random.random() - 0.5)), name=f"w[{i}]") for i in range(4)]
y = Value(head_csv["price"].values[0] / 1_000_000, name="y")


for i in range(iterations):
    yhat = (
        w[0]
        + w[1] * Value(x[0], name="x[1]")
        + w[2] * Value(x[1], name="x[2]")
        + w[3] * Value(x[2], name="x[3]")
    )

    loss = (y - yhat) ** Value(2, name="2")

    print(loss.value)
    dot = graphviz.Digraph()
    loss.backprop()
    loss.render(dot)
    dot.render(f"out/out", format="png")

    for j in range(4):
        w[j].value = w[j].value - w[j].grad * 0.001

print(w[0].grad, w[1].grad, w[2].grad, w[3].grad)
