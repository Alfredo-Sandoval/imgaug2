
import imgaug2 as ia


class Dummy1:
    @ia.deprecated(alt_func="Foo")
    def __init__(self):
        pass


class Dummy2:
    @ia.deprecated(alt_func="Foo", comment="Some example comment.")
    def __init__(self):
        pass


class Dummy3:
    def __init__(self):
        pass

    @ia.deprecated(alt_func="bar()",
                   comment="Some example comment.")
    def foo(self):
        pass


@ia.deprecated(alt_func="bar()", comment="Some example comment.")
def foo():
    pass


def main():
    Dummy1()
    Dummy2()
    Dummy3()
    foo()


if __name__ == "__main__":
    main()
